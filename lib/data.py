"""
Data preparation and storage services for GameMusicVAE
"""

import pretty_midi
# For plotting
import os
import pickle
import numpy as np
import math
import random

from lib.config import *

class DataPreparation(object):
    """
    prepare model-using data from raw midi files collected on internet
    """
    dataset = []
    trio_dataset = []
    def __init__(self):
        print("preparing data...")
        DataPreparation.dataset = DataPreparation.build_dataset(DATA_POOL_PATH, DATABASE_PATH)
        DataPreparation.trio_dataset = DataPreparation.generate_trio_data(DATABASE_PATH, self.dataset)
        print("data prepared!")

    @classmethod
    def build_dataset(cls, data_pool_path, save_path, cutoff=0):
        """
        build dataset entity from raw midi files
        parse as pretty_midi PrettyMIDI object
        save out at local holder
        :param data_pool_path: the folder path that stores all raw midi files
        :param save_path: the path of saving persisted dataset
        :param cutoff: limit of read-in midi file number
        :return: dataset(list of PrettyMIDI object) for GameMusicVAE
        """
        # load midi files as PrettyMIDI
        midi_file_list = []
        file_name_list = os.listdir(data_pool_path)
        print("\nloading midi files...")
        cutoff = cutoff if cutoff > 0 else len(file_name_list)

        # batch loading
        loaded_midi_count = 0
        file_num = min(len(file_name_list), cutoff if cutoff > 0 else len(file_name_list))
        for batch in range(int(file_num // LOADING_BATCH_SIZE)):
            midi_file_list = []
            print("loading batch #{}".format(batch))
            start = batch * LOADING_BATCH_SIZE
            end = (batch + 1) * LOADING_BATCH_SIZE if (batch + 1) * LOADING_BATCH_SIZE <= file_num else file_num
            for file_name in file_name_list[start:end]:
                if not os.path.isdir(file_name) and file_name.endswith(".mid") and loaded_midi_count < cutoff:
                    try:
                        midi_file = pretty_midi.PrettyMIDI(data_pool_path + "/" + file_name)
                        midi_file_list.append(midi_file)
                        loaded_midi_count += 1
                        if loaded_midi_count % 25 == 0:
                            print("loaded {} files".format(loaded_midi_count))
                    except BaseException as e:
                        print(e)
            print("load batch #{} success! (#{})".format(batch, len(midi_file_list)))

            # persisting dataset
            print("saving dataset of batch #{}...".format(batch))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(int(math.ceil(LOADING_BATCH_SIZE / DATASET_PART_LENGTH))):
                try:
                    print("saving dataset_part_{}_{}.txt...".format(batch, i))
                    file_path_name = save_path + "/dataset_part_{}_{}.txt".format(batch, i)
                    f = open(file_path_name, 'wb')
                    start = i * DATASET_PART_LENGTH
                    end = (i + 1) * DATASET_PART_LENGTH if (i + 1) * DATASET_PART_LENGTH <= LOADING_BATCH_SIZE else LOADING_BATCH_SIZE
                    pickle.dump(midi_file_list[start:end], f)
                    f.close()
                    print("saved dataset_part_{}_{}.txt at {}!".format(batch, i, save_path))
                except BaseException as e:
                    print(e)
        print("saved all dataset (#{}) at {}!".format(loaded_midi_count, save_path))

        # return
        if len(file_name_list) < LOADING_BATCH_SIZE:
            return midi_file_list
        else:
            return []

    @classmethod
    def generate_trio_data(cls, dataset_path):
        """
        parse dataset to screen and split layers
        generate a wrapped dataset for training networks
        save trainable_dataset to same path with dataset
            quote:
                For the trio data, we used a 16-bar sliding window (with a stride of 1 bar)
                to extract all unique sequences containing an instrument with a program number
                in the piano, chromatic percussion, organ, or guitar interval, [0, 31], one
                in the bass interval, [32, 39], and one that is a drum (channel 10), with at most
                a single bar of consecutive rests in any instrument. If there were multiple
                instruments in any of the three categories, we took the cross product to consider
                all possible combinations. This resulted in 9.4 million examples.
        :param dataset_path: path of loading dataset and saving trio_dataset
        :return: prepared alive trio dataset
        """
        file_name_list = os.listdir(dataset_path)
        loop_count = 0
        trio_midi_count = 0
        for file_name in file_name_list:
            # restore dataset
            dataset = []
            if file_name.startswith(DATASET_NAME):
                print("loading {} into running memory...".format(file_name))
                try:
                    f = open(dataset_path + "/" + file_name, 'rb')
                    dataset = pickle.load(f)
                    f.close()
                    print("loaded {} (#{}) into running memory!".format(file_name, len(dataset)))
                except BaseException as e:
                    print(e)

            # screen and split data for trio dataset
            print("screening & splitting dataset...")
            trio_dataset = []
            for midi in dataset:
                melodies = [instr.program for instr in midi.instruments
                            if instr.program in range(0, 31 + 1) and not instr.is_drum]  # need validate the numbers
                bases = [instr.program for instr in midi.instruments
                         if instr.program in range(32, 39 + 1) and not instr.is_drum]
                drums = [instr.program for instr in midi.instruments if instr.is_drum]
                if len(drums) > 0 and len(bases) > 0 and len(melodies) > 0:
                    # cross product
                    trio_structs = [[melody, base, drum] for melody in melodies for base in bases for drum in drums]
                    # window-split
                    for trio_struct in trio_structs:
                        try:
                            trio_dataset += DataPreparation.window_split_trio(midi, trio_struct)
                        except BaseException as e:
                            print(e)
            trio_midi_count += len(trio_dataset)
            print("created {} trio midis!".format(len(trio_dataset)))

            # persisting trainable_dataset
            print("saving trio dataset from {}...".format(file_name))
            saved_midi_count = 0
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            try:
                file_path_name = dataset_path + "/trio_dataset_part_{}.txt".format(loop_count)
                f = open(file_path_name, 'wb')
                pickle.dump(trio_dataset, f)
                f.close()
                print("saved trio_dataset_part_{}.txt at {}!".format(loop_count, dataset_path))
            except BaseException as e:
                print(e)

            loop_count += 1
        print("loaded all dataset parts (#{}) into running memory!".format(trio_midi_count))

        # return
        return trio_dataset

    @classmethod
    def window_split_trio(cls, midi, trio_struct):
        """
        split the midi layers specified by trio_struct with a TRIO_MIDI_BAR_NUMBER long window
        return a list of window sized midi files that screened by notes' number
        :param midi: PrettyMIDI object to be split
        :param trio_struct: python list of length 3: [melody_instr_num, bass_instr_num, drum_instr_num]
        :return: list of trio midi files split from midi controled by trio_struct
        """
        # data prepare
        length = midi.get_end_time()
        melody_instr = DataPreparation.get_instrument_by_program(midi, trio_struct[0])
        bass_instr   = DataPreparation.get_instrument_by_program(midi, trio_struct[1])
        drum_instr   = DataPreparation.get_instrument_by_program(midi, trio_struct[2])

        # build window list
        change_time_list, tempo_list = midi.get_tempo_changes()
        change_time_list = change_time_list.tolist()
        tempo_list = tempo_list.tolist()

        tse_list = midi.time_signature_changes
        full_note_tempo_list = (np.array([TRIO_MIDI_BAR_NUMBER * 60 * 4 / tempo for tempo in tempo_list])).tolist()
        # bar_tempo_list = (np.array(full_note_tempo_list) * np.array([tse.numerator/tse.denominator for tse in tse_list])).tolist()
        time_line = change_time_list
        time_line.append(length)
        time_segment_list = [[time_line[i], time_line[i + 1]] for i in range(len(time_line)) if (i + 1) < len(time_line)]
        window_list = []
        for i in range(len(time_segment_list)):
            slice = full_note_tempo_list[i]
            end = time_segment_list[i][-1]
            tempo = time_segment_list[i][-1] - time_segment_list[i][0]
            window_list += [[start * slice, (start + 1) * slice] for start in range(int(tempo // slice))
                            if (start + 1) * slice < end]

        # windowing
        trio_midi_list = []
        for window in window_list: # time consuming method!!
            tempo = window[-1] - window[0]
            melody_note_list = DataPreparation.find_notes_in_window(window, melody_instr.notes)
            bass_note_list   = DataPreparation.find_notes_in_window(window, bass_instr.notes)
            drum_note_list   = DataPreparation.find_notes_in_window(window, drum_instr.notes)
            if len(melody_note_list) >= MIN_MELODY_NOTES_IN_SEGMENT \
                    and len(bass_note_list) >= MIN_BASS_NOTES_IN_SEGMENT \
                    and len(drum_note_list) >= MIN_DRUM_NOTES_IN_SEGMENT \
                    and len(melody_note_list) + len(bass_note_list) + len(drum_note_list) >= MIN_NOTES_IN_SEGMENT\
                    and DataPreparation.trio_continuous(window, melody_note_list, bass_note_list, drum_note_list):
                trio_midi = pretty_midi.PrettyMIDI()
                trio_melody = pretty_midi.Instrument(melody_instr.program, is_drum=False)
                trio_bass   = pretty_midi.Instrument(bass_instr.program, is_drum=False)
                trio_drum   = pretty_midi.Instrument(drum_instr.program, is_drum=True)
                trio_melody.notes = melody_note_list
                trio_bass.notes   = bass_note_list
                trio_drum.notes   = drum_note_list
                trio_midi.instruments = [trio_melody, trio_bass, trio_drum]
                trio_midi_list.append(trio_midi)
        # return
        return trio_midi_list

    @classmethod
    def build_note_sequence_dataset(cls):
        # convert_dir_to_note_sequences --input_dir=.\midi --output_file=./note_sequence/notesequences.tfrecord --recursive
        print("building note-sequence dataset with magenta...")
        info = os.popen('convert_dir_to_note_sequences --input_dir=.\midi --output_file=./note_sequence/notesequences.tfrecord --recursive')
        print("system info: {}".format(info))

    @staticmethod
    def trio_continuous(window, melody, bass, drum):
        """
        check if the three instrument is continuous playing sound in a whole in a given time window
        :param window: target time slice
        :param melody: melody instrument's note list
        :param bass: bass instrument's note list
        :param drum: drum instrument's note list
        :return: true/flase
        """
        start = window[0]
        end = window[-1]
        length = end - start
        note_tempo = length / TRIO_MIDI_BAR_NUMBER
        window_list = [[start * note_tempo, (start + SILENCE_STRIDE) * note_tempo]
                       for start in np.arange(0, TRIO_MIDI_BAR_NUMBER, SILENCE_STRIDE)
                       if (start + SILENCE_STRIDE) * note_tempo <= length]
        for window in window_list:
            if len(DataPreparation.find_notes_in_window(window, melody)) > 0 \
                    or len(DataPreparation.find_notes_in_window(window, bass)) > 0\
                    or len(DataPreparation.find_notes_in_window(window, drum)) > 0:
                return True
        return False

    @staticmethod
    def dataset2midi_folder(dataset_path, dataset_name, midi_folder_path):
        """
        read (trio) dataset file and transfer PretrtyMIDI objects to .mid file and save in midi_folder_path
        :param dataset_path:
        :param dataset_name:
        :param midi_folder_path:
        :param alive_dataset:
        :return:
        """

        dataset = []
        file_name_list = os.listdir(dataset_path)
        midi_count = 0
        print("\nsaving midi object to .mid files...")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        for file_name in file_name_list:
            # restore dataset
            if file_name.startswith(dataset_name):
                try:
                    print("\nloading {} to memory...".format(file_name))
                    f = open(dataset_path + "/" + file_name, 'rb')
                    dataset = pickle.load(f)
                    f.close()
                    print("loaded {} into running memory!".format(file_name))
                except BaseException as e:
                    print(e)

                # generate & save midi files
                print("saving {} to midi files...".format(file_name))
                count = 0;
                for midi in dataset:
                    try:
                        count += 1
                        midi.write(midi_folder_path + "/" + str(count + midi_count) + ".mid")
                        if (count + midi_count) % 100 == 0:
                            print("saved {} midi objects".format(count + midi_count))
                    except BaseException as e:
                        print(e)
                midi_count += count
                print("saved all midi files (#{}) in {}!".format(count, file_name))
        print("saved all midi objects (#{}) to {}!".format(midi_count, dataset_path))



    @staticmethod
    def get_instrument_by_program(midi, program):
        """
        return the instrument object specified by given program number in a PrettyMIDI object
        :param midi:
        :param program:
        :return:
        """
        for instr in midi.instruments:
            if instr.program == program:
                return instr
        return None

    @staticmethod
    def find_notes_in_window(window, instr_note_list):
        """
        return a list of node from given instr_note_list whose life time in window
        :param window:
        :param instr_note_list:
        :return:
        """
        note_list = []
        start = window[0]
        end   = window[-1]
        for note in instr_note_list:
            if start <= note.start < end and note.end <= end:
                note_list.append(note)
        return note_list


class DataAnalysis(object):
    @staticmethod
    def dataset_analysis(dataset_path, dataset_name, cutoff=0, shuffle=True):
        """
        return a beat tempo histogram of a dataset
        :param dataset_path:
        :param dataset_name: specified the dataset file prefix
        :param cutoff: stop loading dataset when loaded more PrettyMIDI objects than this number
        :param shuffle: shuffle the dataset if True
        :return:
        """
        file_name_list = os.listdir(dataset_path)
        count = 0
        if shuffle:
            random.shuffle(file_name_list)

        dataset = []
        for file_name in file_name_list:
            # restore dataset
            if file_name.startswith(dataset_name)  and count <= cutoff:
                print("loading {} into running memory...".format(file_name))
                try:
                    f = open(dataset_path + "/" + file_name, 'rb')
                    dataset = pickle.load(f)
                    f.close()
                    count += len(dataset)
                    print("loaded {} (#{}) into memory!".format(file_name, len(dataset)))
                except BaseException as e:
                    print(e)
        if shuffle:
            random.shuffle(dataset)
        print("loaded all (#{}) midi files into memory!".format(count))

