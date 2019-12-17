import argparse
import os

from lib.data import *


def main():
    # check directory
    upper_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    source_dir = os.path.join(upper_dir, 'source')
    data_dir = os.path.join(source_dir, 'data')
    database_dir = os.path.join(source_dir, 'database')
    trio_midi_dir = os.path.join(source_dir, 'trio_midi')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)
    if not os.path.exists(trio_midi_dir):
        os.makedirs(trio_midi_dir)

    # handel command parameters
    parser = argparse.ArgumentParser(description='Deal with all data processing.')
    parser.add_argument('-p', '--pipeline', help='pipeline: select data collect process')
    # parser.add_argument('-tm', '--trio2mid', action='store_true', default=False, help='transfer trio dataset to midi')
    # parser.add_argument('-', '--', help='')
    args = parser.parse_args()

    if args.pipeline == 'collect':
        from lib.crawler import run
        run.spider_man_vgmusic(False, save_path=DATA_POOL_PATH)
    elif args.pipeline == 'build_dataset':
        DataPreparation.build_dataset(DATA_POOL_PATH, DATABASE_PATH, cutoff=0)
    elif args.pipeline == 'build_trio':
        DataPreparation.generate_trio_data(DATABASE_PATH)
    elif args.pipeline == 'transfer_midi':
        DataPreparation.dataset2midi_folder(DATABASE_PATH, TRIO_DATASET_NAME, TRIO_MIDI_DATASET_PATH)
    elif args.pipeline == 'transfer_sequence_note':
        DataPreparation.build_note_sequence_dataset()


if __name__ == '__main__':
    main()
