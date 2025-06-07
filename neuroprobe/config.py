##########################
# 1. Paths and directories settings
##########################

# It is recommended that all the paths are absolute!
# Feel free to change this setting to your own directory.
ROOT_DIR = "/om2/user/zaho/braintreebank/braintreebank" # Root directory for the braintreebank data. Recommended to use the absolute path.
SAVE_SUBJECT_TRIAL_DF_DIR = "/om2/user/zaho/neuroprobe/neuroprobe/braintreebank_features_time_alignment"
#SAVE_SUBJECT_TRIAL_DF_DIR = "/om2/user/zaho/neuroprobe/btbench_subject_metadata"
PITCH_VOLUME_FEATURES_DIR = "/om2/user/zaho/neuroprobe/neuroprobe/pitch_volume_features" # This comes together with the neuroprobe package


# Disable file locking for HDF5 files. This is helpful for parallel processing.
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 


##########################
# 2. Neuroprobe settings
# Do not change the settings below if you want the neuroprobe results to be compatible with others' evaluations!
##########################

SAMPLING_RATE = 2048 # Sampling rate for the BrainTreebank data (do not change this)
# Define a global random seed for reproducibility
NEUROPROBE_GLOBAL_RANDOM_SEED = 42  

# No need to change the settings below because you can change these values dynamically when defining the splits using the functions from neuroprobe_train_test_splits.py and the dataset from neuroprobe_datasets.py
START_NEURAL_DATA_BEFORE_WORD_ONSET = 0.5 # in seconds. NOTE: for the 1-second evaluation on the leaderboard, this is overridden to 0.
END_NEURAL_DATA_AFTER_WORD_ONSET = 2 # in seconds. NOTE: for the 1-second evaluation on the leaderboard, this is overridden to 1.
NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2 # how many seconds to wait between the last word off-set and the start of a "non-verbal" chunk
NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP = 0.8 # proportion of overlap between consecutive nonverbal chunks (0 means no overlap)
# some sanity check code as well as disabling file locking for HDF5 files
assert NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP >= 0 and NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP < 1, "NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"

# Standardizing pretraining and evaluation subjects and trials
DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID = 2, 4
# Define the maximum number of samples to use for the lite dataset
NEUROPROBE_LITE_MAX_SAMPLES = 3500
NEUROPROBE_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]

# Only used for the "Full" Neuroprobe dataset
NEUROPROBE_FULL_SUBJECT_TRIALS = [
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 0), (3, 1), (3, 2), 
    (4, 0), (4, 1), (4, 2), 
    (5, 0), 
    (6, 0), (6, 1), (6, 4),
    (7, 0), (7, 1), 
    (8, 0), 
    (9, 0), 
    (10, 0), (10, 1)
]
NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT = {
    1: [0, 1],
    2: [4, 6],
    3: [2, 1],
    4: [2, 1],
    5: [0],
    6: [0, 2],
    7: [1, 0],
    8: [0], 
    9: [0],
    10: [1, 0]
}

BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING = {
    "btbank1_0": "fantastic-mr-fox",
    "btbank1_1": "the-martian",
    "btbank1_2": "thor-ragnarok",
    "btbank2_0": "venom",
    "btbank2_1": "spider-man-3-homecoming",
    "btbank2_2": "guardians-of-the-galaxy",
    "btbank2_3": "guardians-of-the-galaxy-2",
    "btbank2_4": "avengers-infinity-war",
    "btbank2_5": "black-panther",
    "btbank2_6": "aquaman",
    "btbank3_0": "cars-2",
    "btbank3_1": "lotr-1",
    "btbank3_2": "lotr-2",
    "btbank4_0": "shrek-the-third",
    "btbank4_1": "megamind",
    "btbank4_2": "incredibles",
    "btbank5_0": "fantastic-mr-fox",
    "btbank6_0": "megamind",
    "btbank6_1": "toy-story",
    "btbank6_4": "coraline",
    "btbank7_0": "cars-2",
    "btbank7_1": "megamind",
    "btbank8_0": "sesame-street-episode-3990",
    "btbank9_0": "ant-man",
    "btbank10_0": "cars-2",
    "btbank10_1": "spider-man-far-from-home"
}

NEUROPROBE_LITE_ELECTRODES = {"btbank1": ["F3aOFa14", "F3aOFa3", "F3aOFa2", "F3aOFa8", "F3aOFa7", "F3aOFa12", "F3aOFa15", "F3aOFa9", "F3aOFa16", "F3aOFa13", "F3aOFa4", "F3aOFa10", "F3bIaOFb2", "F3bIaOFb1", "F3bIaOFb16", "F3bIaOFb4", "F3bIaOFb13", "F3bIaOFb9", "F3bIaOFb10", "F3bIaOFb15", "F3bIaOFb12", "F3bIaOFb6", "F3bIaOFb7", "F3bIaOFb5", "F3bIaOFb8", "F3bIaOFb11", "F3bIaOFb14", "F3cId10", "F3cId5", "F3cId1", "F3cId7", "F3cId2", "F3cId4", "F3cId3", "F3cId6", "F3cId8", "T1aIb4", "T1aIb7", "T1aIb3", "T1aIb1", "T1aIb5", "T1aIb2", "T1aIb8", "T2aA6", "T2aA12", "T2aA10", "T2aA5", "T2aA1", "T2aA11", "T2aA4", "T2aA9", "T2aA8", "T2aA2", "T2aA7", "T2bHa11", "T2bHa7", "T2bHa12", "T2bHa8", "T2bHa5", "T2bHa14", "T2bHa1", "T2bHa9", "T2bHa3", "T2bHa10", "T2bHa13", "T1bIc4", "T1bIc7", "T1bIc1", "T1bIc8", "T1bIc3", "T1bIc2", "T1bIc5", "F3dIe3", "F3dIe6", "F3dIe10", "F3dIe4", "F3dIe9", "F3dIe14", "F3dIe1", "F3dIe2", "F3dIe7", "F3dIe5", "T3aHb9", "T3aHb12", "T3aHb10", "T3aHb6", "T1cIf16", "T1cIf12", "T1cIf13", "T1cIf10", "T1cIf4", "T1cIf6", "T1cIf1", "T1cIf15", "T1cIf7", "T1cIf14", "T1cIf3", "T1cIf8", "T1cIf2", "T1cIf11", "T2c5", "T2c6", "T2c4", "T2c8", "T2c7", "T3bOT9", "T3bOT3", "T3bOT5", "T3bOT2", "T3bOT8", "T3bOT12", "T3bOT10", "T3bOT4", "T3bOT6", "T2d5", "T2d4", "T2d3", "T2d1", "T2d6", "T2d2"], "btbank2": ["LT3a2", "LT3a1", "LT3a10", "LT3a8", "LT3a6", "LT3a7", "LT3a4", "LT3a3", "LT3a9", "LT2aA8", "LT2aA14", "LT2aA11", "LT2aA9", "LT2aA10", "LT2aA6", "LT2aA12", "LT2aA1", "LT2aA7", "LT2aA4", "LT2aA5", "LT2aA2", "LT3bHa5", "LT3bHa8", "LT3bHa10", "LT3bHa14", "LT3bHa7", "LT3bHa4", "LT3bHa6", "LT3bHa9", "LT3bHa12", "LT1bIb9", "LT1bIb2", "LT1bIb5", "LT1bIb7", "LT1bIb6", "LT1bIb8", "LT1bIb10", "LT1bIb1", "LT1bIb4", "LT2b2", "LT2b5", "LT2b1", "LT2b3", "LT2b6", "LT3cHb11", "LT3cHb4", "LT3cHb12", "LT3cHb6", "LT3cHb9", "LT3cHb5", "LT3cHb10", "LT3cHb8", "LT1c6", "LT1c2", "LT1c7", "LT1c5", "LT1c8", "LT1c4", "LT3d3", "LT3d6", "LT3d4", "LT3d5", "LT3d2", "LT3d7", "LT3d8", "RT1aIa4", "RT1aIa5", "RT1aIa6", "RT1aIa2", "RT1aIa3", "RT1aIa7", "RT1aIa8", "RT3aHa11", "RT3aHa12", "RT3aHa14", "RT3aHa5", "RT3aHa13", "RT3aHa10", "RT3aHa4", "RT3aHa6", "RT3aHa8", "RT3aHa7", "RT2aA10", "RT2aA4", "RT2aA1", "RT2aA8", "RT2aA6", "RT2aA7", "RT2aA2", "RT2aA5", "RT2aA3", "RT2b3", "RT2b1", "RT2b2", "RT2b4", "RT2b8", "RT1bIb4", "RT1bIb5", "RT1bIb2", "RT1bIb7", "RT1bIb8", "RT1bIb1", "RT1bIb6", "RT3bHb11", "RT3bHb5", "RT3bHb4", "RT3bHb10", "RT3bHb7", "RT3bHb12", "RT3bHb6", "RT2c2", "RT2c1", "RT1c1", "RT1c2", "RT1c10", "RT1c4", "RT1c3", "RT1c8", "RT1c7", "RT1c5"], "btbank3": ["P2b14", "P2b6", "O1bId13", "F3aOF6", "T1b6", "P2a8", "F3aOF3", "O1bId14", "O1aIb13", "F2Ia5", "F2Ia13", "O1bId4", "P2a2", "O1aIb1", "F3aOF2", "F2Ia11", "F2Ia4", "F3d6", "O1aIb4", "F3c9", "O1bId3", "O1bId11", "F3c5", "O1aIb5", "P2a1", "T1cIe11", "P2b15", "O1bId2", "T1cIe8", "F2Ia9", "F3c10", "F2Ia1", "P2b8", "F3c4", "T1cIe12", "F2Ia7", "T1aIc1", "T1cIe9", "O1aIb2", "F3c6", "P2a3", "O1aIb7", "O1aIb11", "F3d10", "F3d8", "P2b16", "F3d2", "F3b8", "F3aOF4", "F2Ia6", "T1cIe6", "O1aIb14", "T1aIc2", "P2b10", "F3aOF7", "O1aIb16", "F3c8", "P2b3", "F2Ia3", "T1b3", "P2b13", "P2a10", "F3c3", "T1aIc3", "T1cIe10", "T1aIc6", "O1aIb9", "P2b9", "O1bId9", "P2a5", "F3c1", "F3c2", "F2Ia10", "F3d3", "O1bId15", "F3b1", "F2Ia8", "T1b2", "P2b5", "F2Ia12", "F3aOF5", "T1b5", "O1aIb6", "T1aIc4", "P2b12", "P2b11", "F3d7", "F3b2", "P2a4", "P2a9", "O1bId1", "O1aIb3", "O1bId5", "T1b4", "O1aIb8", "F3aOF1", "F3aOF8", "P2b1", "P2a6", "O1bId8", "O1bId16", "F2Ia2", "T1cIe7", "P2b7", "P2a7", "F3d4", "F3d9", "T1cIe2", "O1aIb10", "T1cIe5", "F3b7", "F3d5", "T1aIc5", "P2b4", "F3d1", "O1bId12", "F3b3", "O1bId10", "F2Ia14", "T1cIe1"], "btbank4": ["LT2aA1", "LT2aA7", "LT2aA12", "LT2aA9", "LT2aA4", "LT2aA3", "LT2aA8", "LT2aA10", "LT3Ha12", "LT3Ha6", "LT3Ha11", "LT3Ha13", "LT3Ha2", "LT3Ha5", "LT3Ha8", "LT3Ha3", "LT1aIb7", "LT1aIb6", "LT1aIb10", "LT1aIb9", "LT1aIb3", "LT1aIb5", "LF3aOFa5", "LF3aOFa4", "LF3aOFa7", "LF3aOFa11", "LF3aOFa14", "LF3aOFa13", "LF3aOFa3", "LF3aOFa16", "LF3aOFa12", "LF3aOFa9", "LF3bIa1", "LF3bIa5", "LF3bIa11", "LF3bIa4", "LF3bIa7", "LF3bIa8", "LF3bIa10", "LF3bIa6", "LT2bHb11", "LT2bHb6", "LT2bHb8", "LT2bHb10", "LT2bHb9", "LT2bHb12", "LT2bHb2", "LT2bHb5", "LT1cCd16", "LT1cCd13", "LT1cCd15", "LT1cCd12", "LT1cCd3", "LT1cCd11", "LT1cCd2", "LT1cCd5", "LT1cCd6", "LT1cCd1", "LT1bId4", "LT1bId6", "LT1bId3", "LT1bId2", "LT1bId5", "LF3cIc3", "LF3cIc1", "LF3cIc9", "LF3cIc2", "LF3cIc4", "LF3cIc5", "LF1aCaOF3", "LF1aCaOF8", "LF1aCaOF7", "LF1aCaOF11", "LF1aCaOF10", "LF1aCaOF4", "LF1aCaOF14", "LF1aCaOF15", "LF1aCaOF12", "LF1aCaOF2", "RF1aCaOF5", "RF1aCaOF11", "RF1aCaOF12", "RF1aCaOF1", "RF1aCaOF2", "RF1aCaOF7", "RF1aCaOF4", "RF1aCaOF3", "RF1aCaOF8", "RF1aCaOF13", "LF1bCb9", "LF1bCb4", "LF1bCb2", "LF1bCb10", "LF1bCb8", "LF1bCb7", "RF1bCb9", "RF1bCb10", "RF1bCb6", "RF1bCb7", "RF1bCb4", "RF1bCb5", "LF1cCc12", "LF1cCc9", "LF1cCc7", "LF1cCc11", "LF1cCc8", "LF1cCc2", "LF1cCc6", "LF1cCc4", "RF1cCc8", "RF1cCc5", "RF1cCc4", "RF1cCc11", "RF1cCc12", "RF1cCc6", "RF1cCc2", "RF1cCc3", "RF1bCb1", "LT1aIb1", "LF3cIc6"], "btbank7": ["LT2H8", "LT2H5", "LT2H6", "LT2H14", "LT2H9", "LT2H2", "LT2H3", "LT2H10", "LT1A4", "LT1A7", "LT1A12", "LT1A3", "LT1A14", "LT1A2", "LT1A13", "LF3aOFa14", "LF3aOFa6", "LF3aOFa9", "LF3aOFa8", "LF3aOFa7", "LF3aOFa1", "LF3aOFa4", "LF3aOFa12", "LF3cIc7", "LF3cIc1", "LF3cIc10", "LF3cIc4", "LF3cIc9", "LF3bOFb12", "LF3bOFb5", "LF3bOFb13", "LF3bOFb7", "LF3bOFb14", "LF3bOFb9", "LF3bOFb11", "LF3bOFb4", "LF2b8", "LF2b4", "LF2b5", "LF2b7", "LF2b14", "LF2b1", "LF2b11", "LF2e6", "LF2e8", "LF2e7", "LF2e4", "LF2c3", "LF2c7", "LF2c4", "LF2c2", "LF2d1", "LF2d4", "LF2d5", "LF2d6", "LF2aIa1", "LF2aIa2", "LF2aIa11", "LF2aIa7", "LF2aIa3", "LF2aIa8", "LF2aIa12", "LF2aIa16", "LF1aCaOF9", "LF1aCaOF7", "LF1aCaOF6", "LF1aCaOF4", "LF1aCaOF8", "LF1aCaOF14", "LF1aCaOF11", "LF1aCaOF15", "LF1bIb9", "LF1bIb13", "LF1bIb14", "LF1bIb7", "LF1bIb5", "LF1bIb2", "LF1bIb8", "LF1bIb1", "LF1cCb12", "LF1cCb9", "LF1cCb1", "LF1cCb6", "LF1cCb4", "LF1cCb11", "RF3aOFa3", "RF3aOFa13", "RF3aOFa11", "RF3aOFa1", "RF3aOFa15", "RF3aOFa4", "RF3aOFa14", "RF3aOFa12", "RF3bOFb5", "RF3bOFb4", "RF3bOFb3", "RF3bOFb8", "RF3bOFb11", "RF3bOFb2", "RF3bOFb10", "RF3bOFb15", "RF2I8", "RF2I12", "RF2I5", "RF2I6", "RF2I3", "RF2I2", "RF1aCaOF6", "RF1aCaOF5", "RF1aCaOF2", "RF1aCaOF10", "RF1aCaOF1", "RF1aCaOF15", "RF1aCaOF13", "RF1aCaOF7", "RF1bCb7", "RF1bCb4", "RF1bCb1", "RF1bCb5", "RF3aOFa7"], "btbank10": ["P2cCc4", "F2aOFb5", "T1cIf8", "F2e8", "F2bCa6", "F2aOFb11", "F3bIb7", "T2aA12", "F10Fa3", "T2bH12", "P2bIg5", "P2cCc8", "F3bIb6", "F3bIb8", "F3cIc7", "P2cCc1", "T1bId7", "T2bH13", "T1aIa3", "F2bCa14", "T2aA1", "T2bH4", "F3cIc6", "F10Fa1", "T2aA11", "F2bCa13", "F2e5", "F2c8", "F2dCb8", "T1cIf6", "F2aOFb14", "P2bIg6", "P2cCc14", "T2aA9", "T1aIa5", "T2aA4", "F3aOFc2", "F3cIc9", "P2bIg7", "P2bIg4", "F2e4", "T1bId4", "F3aOFc11", "P2aIe4", "F3bIb4", "F2bCa2", "F2aOFb15", "F3cIc8", "P2cCc10", "T2bH1", "T2aA2", "T1bId8", "T1bId10", "T1aIa2", "F2aOFb2", "T1cIf12", "T1aIa7", "T1cIf11", "P2cCc5", "T2bH6", "F2e6", "T2bH15", "F3aOFc13", "F2aOFb12", "P2aIe8", "P2bIg8", "F3aOFc12", "T1cIf7", "T2bH5", "F2dCb3", "F2c7", "P2aIe6", "P2aIe7", "P2bIg11", "T1aIa8", "F2dCb4", "F2bCa10", "F2dCb9", "F10Fa16", "P2cCc11", "F10Fa15", "F3aOFc1", "F3aOFc14", "F2c6", "T1aIa1", "P2aIe12", "F10Fa12", "T1cIf2", "F2bCa5", "P2aIe10", "F2bCa3", "T2bH11", "T1bId5", "F2dCb10", "F2aOFb16", "F2dCb11", "F2e3", "F3cIc1", "F2dCb12", "T2bH2", "F3aOFc8", "F10Fa8", "F10Fa11", "F2dCb14", "P2cCc6", "F2c5", "P2cCc7", "F2aOFb6", "F3bIb2", "F3bIb10", "F2aOFb4", "P2aIe3", "F2c3", "T2aA8", "F3aOFc16", "F2bCa12", "F3aOFc15", "F2bCa4", "F3cIc5", "T1bId3"]}