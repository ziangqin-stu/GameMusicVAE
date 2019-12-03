"""
test & debug playground
"""
import lib
from lib.data import *
from lib.config import *

# dataset = DataPreparation.build_dataset(DATA_POOL_PATH, DATABASE_PATH, cutoff=0)
# trio_dataset = DataPreparation.generate_trio_data(DATABASE_PATH, dataset)
# DataPreparation.dataset2midi_folder(DATABASE_PATH, TRIO_DATASET_NAME, "./trio_midi", trio_dataset)


# DataPreparation.generate_trio_data(DATABASE_PATH)
# DataPreparation.dataset2midi_folder(DATABASE_PATH, TRIO_DATASET_NAME, "./trio_midi")


# dataset = DataPreparation.build_dataset(DATA_POOL_PATH, DATABASE_PATH, cutoff=150)
# trio_dataset = DataPreparation.generate_trio_data(DATABASE_PATH)
# DataPreparation.dataset2midi_folder(DATABASE_PATH, TRIO_DATASET_NAME, "./trio_midi", trio_dataset)


# DataPreparation.build_dataset(DATA_POOL_PATH, DATABASE_PATH, cutoff=0)
# trio_dataset = lib.data.DataPreparation.generate_trio_data(DATABASE_PATH)
lib.data.DataPreparation.dataset2midi_folder(DATABASE_PATH, TRIO_DATASET_NAME, "./source/trio_midi")
# lib.data.DataPreparation.build_note_sequence_dataset()