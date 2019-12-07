import pretty_midi
import os
import pickle
import numpy as np
import random


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


def train_beat_tempo_vector(dataset_path, dataset_name, thresh_hold=None):
    """

    :param dataset_path:
    :param dataset_name:
    :param thresh_hold:
    :return:
    """
    # load dataset in memory
    dataset = []
    file_name_list = os.listdir(dataset_path)
    print("\nsaving midi object to .mid files...")
    for file_name in file_name_list:
        # restore dataset
        if file_name.startswith(dataset_name):
            try:
                print("\nloading {} to memory...".format(file_name))
                f = open(dataset_path + "/" + file_name, 'rb')
                dataset += pickle.load(f)
                f.close()
                print("loaded {} into running memory!".format(file_name))
            except BaseException as e:
                print(e)
    print("dataset (#{}) loaded into memory!".format(len(dataset)))

    # get beat length
    if not thresh_hold:
        beat_length_dict = {}
        beat_length = []
        segment_count = 0
        for midi in dataset:
            end_time = midi.get_end_time()
            tse_list = midi.time_signature_changes
            change_time_list, tempo_list = midi.get_tempo_changes()
            segment_count += len(change_time_list)
            cur_beat_length = [(60 * 4) / tempo_list[i] / tse_list[i].denominator for i in range(len(change_time_list))] * \
                              [(change_time_list + [end_time])[i + 1] - (change_time_list +[end_time])[i] for i in range(len(change_time_list))]
            beat_length_dict[midi] = np.average(cur_beat_length)
            beat_length.append(np.average(cur_beat_length))
        print("datatset average beat length is {}.".format(np.average(beat_length)))

    # split dataset by beat length
    sorted_beat_length_list = sorted(beat_length_dict.items(), key=lambda item: item[1])



    # collect hidden codes

