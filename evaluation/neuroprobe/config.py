##########################
# 1. Paths and directories settings
##########################

# It is recommended that all the paths are absolute!
# Feel free to change this setting to your own directory.
ROOT_DIR = "/om2/user/zaho/braintreebank/braintreebank" # Root directory for the braintreebank data. Recommended to use the absolute path.
SAVE_SUBJECT_TRIAL_DF_DIR = "/om2/user/zaho/bfm/evaluation/neuroprobe/braintreebank_features_time_alignment"
#SAVE_SUBJECT_TRIAL_DF_DIR = "/om2/user/zaho/neuroprobe/btbench_subject_metadata"
PITCH_VOLUME_FEATURES_DIR = "/om2/user/zaho/bfm/evaluation/neuroprobe/pitch_volume_features" # This comes together with the neuroprobe package


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

NEUROPROBE_LITE_ELECTRODES = {"btbank1": ["F3bIaOFb1", "F3bIaOFb2", "F3bIaOFb3", "F3bIaOFb4", "F3bIaOFb5", "F3bIaOFb6", "F3bIaOFb7", "F3bIaOFb8", "F3bIaOFb9", "F3bIaOFb10", "F3bIaOFb11", "F3bIaOFb12", "F3bIaOFb13", "F3bIaOFb14", "F3bIaOFb15", "F3bIaOFb16", "T2aA1", "T2aA2", "T2aA3", "T2aA4", "T2aA5", "T2aA6", "T2aA7", "T2aA8", "T2aA9", "T2aA10", "T2aA11", "T2aA12", "F3dIe1", "F3dIe2", "F3dIe3", "F3dIe4", "F3dIe5", "F3dIe6", "F3dIe7", "F3dIe8", "F3dIe9", "F3dIe10", "F3aOFa7", "F3aOFa8", "F3aOFa9", "F3aOFa10", "F3aOFa11", "F3aOFa12", "F3aOFa13", "F3aOFa14", "F3aOFa15", "F3aOFa16", "F3cId1", "F3cId2", "F3cId3", "F3cId4", "F3cId5", "F3cId6", "F3cId7", "F3cId8", "F3cId9", "F3cId10", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5", "T1cIf6", "T1cIf7", "T1cIf8", "T2bHa7", "T2bHa8", "T2bHa9", "T2bHa10", "T2bHa11", "T2bHa12", "T2bHa13", "T2bHa14", "T1aIb1", "T1aIb2", "T1aIb3", "T1aIb4", "T1aIb5", "T1aIb6", "T1aIb7", "T1aIb8", "T1bIc1", "T1bIc2", "T1bIc3", "T1bIc4", "T1bIc5", "T1bIc6", "T1bIc7", "T1bIc8", "T1cIf10", "T1cIf11", "T1cIf12", "T1cIf13", "T1cIf14", "T1cIf15", "T1cIf16", "T3bOT1", "T3bOT2", "T3bOT3", "T3bOT4", "T3bOT5", "T3bOT6", "T2d1", "T2d2", "T2d3", "T2d4", "T2d5", "T2d6", "T2c4", "T2c5", "T2c6", "T2c7", "T2c8", "T2bHa3", "T2bHa4", "T2bHa5", "T3bOT8", "T3bOT9", "T3bOT10"], "btbank2": ["LT2aA4", "LT2aA5", "LT2aA6", "LT2aA7", "LT2aA8", "LT2aA9", "LT2aA10", "LT2aA11", "LT2aA12", "LT2aA13", "LT2aA14", "RT3aHa4", "RT3aHa5", "RT3aHa6", "RT3aHa7", "RT3aHa8", "RT3aHa9", "RT3aHa10", "RT3aHa11", "RT3aHa12", "RT3aHa13", "RT3aHa14", "LT3a1", "LT3a2", "LT3a3", "LT3a4", "LT3a5", "LT3a6", "LT3a7", "LT3a8", "LT3a9", "LT3a10", "LT1bIb1", "LT1bIb2", "LT1bIb3", "LT1bIb4", "LT1bIb5", "LT1bIb6", "LT1bIb7", "LT1bIb8", "LT1bIb9", "LT1bIb10", "RT2aA1", "RT2aA2", "RT2aA3", "RT2aA4", "RT2aA5", "RT2aA6", "RT2aA7", "RT2aA8", "RT2aA9", "RT2aA10", "LT3cHb4", "LT3cHb5", "LT3cHb6", "LT3cHb7", "LT3cHb8", "LT3cHb9", "LT3cHb10", "LT3cHb11", "LT3cHb12", "LT3bHa4", "LT3bHa5", "LT3bHa6", "LT3bHa7", "LT3bHa8", "LT3bHa9", "LT3bHa10", "LT3bHa11", "LT3bHa12", "RT1bIb1", "RT1bIb2", "RT1bIb3", "RT1bIb4", "RT1bIb5", "RT1bIb6", "RT1bIb7", "RT1bIb8", "LT3d1", "LT3d2", "LT3d3", "LT3d4", "LT3d5", "LT3d6", "LT3d7", "LT3d8", "RT1aIa1", "RT1aIa2", "RT1aIa3", "RT1aIa4", "RT1aIa5", "RT1aIa6", "RT1aIa7", "RT1aIa8", "LT1c2", "LT1c3", "LT1c4", "LT1c5", "LT1c6", "LT1c7", "LT1c8", "LT2b1", "LT2b2", "LT2b3", "LT2b4", "LT2b5", "LT2b6", "RT3bHb4", "RT3bHb5", "RT3bHb6", "RT3bHb7", "RT3bHb8", "RT1c1", "RT1c2", "RT1c3", "RT1c4", "RT1c5", "RT3bHb10", "RT3bHb11", "RT3bHb12"], "btbank3": ["O1aIb1", "O1aIb2", "O1aIb3", "O1aIb4", "O1aIb5", "O1aIb6", "O1aIb7", "O1aIb8", "O1aIb9", "O1aIb10", "O1aIb11", "O1aIb12", "O1aIb13", "O1aIb14", "O1aIb15", "O1aIb16", "F2Ia1", "F2Ia2", "F2Ia3", "F2Ia4", "F2Ia5", "F2Ia6", "F2Ia7", "F2Ia8", "F2Ia9", "F2Ia10", "F2Ia11", "F2Ia12", "F2Ia13", "F2Ia14", "P2b3", "P2b4", "P2b5", "P2b6", "P2b7", "P2b8", "P2b9", "P2b10", "P2b11", "P2b12", "P2b13", "P2b14", "P2b15", "P2b16", "P2a1", "P2a2", "P2a3", "P2a4", "P2a5", "P2a6", "P2a7", "P2a8", "P2a9", "P2a10", "F3d1", "F3d2", "F3d3", "F3d4", "F3d5", "F3d6", "F3d7", "F3d8", "F3d9", "F3d10", "O1bId8", "O1bId9", "O1bId10", "O1bId11", "O1bId12", "O1bId13", "O1bId14", "O1bId15", "O1bId16", "F3aOF1", "F3aOF2", "F3aOF3", "F3aOF4", "F3aOF5", "F3aOF6", "F3aOF7", "F3aOF8", "T1cIe5", "T1cIe6", "T1cIe7", "T1cIe8", "T1cIe9", "T1cIe10", "T1cIe11", "T1cIe12", "T1aIc1", "T1aIc2", "T1aIc3", "T1aIc4", "T1aIc5", "T1aIc6", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6", "F3c1", "F3c2", "F3c3", "F3c4", "F3c5", "F3c6", "O1bId1", "O1bId2", "O1bId3", "O1bId4", "O1bId5", "F3b1", "F3b2", "F3b3", "F3b4", "F3c8", "F3c9", "F3c10", "P2b1"], "btbank4": ["LF3aOFa1", "LF3aOFa2", "LF3aOFa3", "LF3aOFa4", "LF3aOFa5", "LF3aOFa6", "LF3aOFa7", "LF3aOFa8", "LF3aOFa9", "LF3aOFa10", "LF3aOFa11", "LF3aOFa12", "LF3aOFa13", "LF3aOFa14", "LF3aOFa15", "LF3aOFa16", "RF1aCaOF1", "RF1aCaOF2", "RF1aCaOF3", "RF1aCaOF4", "RF1aCaOF5", "RF1aCaOF6", "RF1aCaOF7", "RF1aCaOF8", "RF1aCaOF9", "RF1aCaOF10", "RF1aCaOF11", "RF1aCaOF12", "RF1aCaOF13", "RF1aCaOF14", "RF1aCaOF15", "RF1aCaOF16", "LT1cCd1", "LT1cCd2", "LT1cCd3", "LT1cCd4", "LT1cCd5", "LT1cCd6", "LT1cCd7", "LT1cCd8", "LT1cCd9", "LT1cCd10", "LT1cCd11", "LT1cCd12", "LT1cCd13", "LT1cCd14", "LT1cCd15", "LT1cCd16", "LF1aCaOF1", "LF1aCaOF2", "LF1aCaOF3", "LF1aCaOF4", "LF1aCaOF5", "LF1aCaOF6", "LF1aCaOF7", "LF1aCaOF8", "LF1aCaOF9", "LF1aCaOF10", "LF1aCaOF11", "LF1aCaOF12", "LF1aCaOF13", "LF1aCaOF14", "LF1aCaOF15", "LF1aCaOF16", "LF3bIa1", "LF3bIa2", "LF3bIa3", "LF3bIa4", "LF3bIa5", "LF3bIa6", "LF3bIa7", "LF3bIa8", "LF3bIa9", "LF3bIa10", "LF3bIa11", "LF3bIa12", "LF1cCc1", "LF1cCc2", "LF1cCc3", "LF1cCc4", "LF1cCc5", "LF1cCc6", "LF1cCc7", "LF1cCc8", "LF1cCc9", "LF1cCc10", "LF1cCc11", "LF1cCc12", "LT2bHb1", "LT2bHb2", "LT2bHb3", "LT2bHb4", "LT2bHb5", "LT2bHb6", "LT2bHb7", "LT2bHb8", "LT2bHb9", "LT2bHb10", "LT2bHb11", "LT2bHb12", "LT2aA1", "LT2aA2", "LT2aA3", "LT2aA4", "LT2aA5", "LT2aA6", "LT2aA7", "LT2aA8", "LT2aA9", "LT2aA10", "LT2aA11", "LT2aA12", "LT1bId1", "LT1bId2", "LT1bId3", "LT1bId4", "LT1bId5", "LT1bId6", "LT1bId7", "LT1bId10"], "btbank5": ["T1bId1", "T1bId2", "T1bId3", "T1bId4", "T1bId5", "T1bId6", "T1bId7", "T1bId8", "T1bId9", "T1bId10", "T1bId11", "T1bId12", "T1bId13", "T1bId14", "T1bId15", "T1bId16", "P2Ie2", "P2Ie3", "P2Ie4", "P2Ie5", "P2Ie6", "P2Ie7", "P2Ie8", "P2Ie9", "P2Ie10", "P2Ie11", "P2Ie12", "P2Ie13", "P2Ie14", "P2Ie15", "P2Ie16", "T1aA1", "T1aA2", "T1aA3", "T1aA4", "T1aA5", "T1aA6", "T1aA7", "T1aA8", "T1aA9", "T1aA10", "T1aA11", "T1aA12", "T1aA13", "T1aA14", "F1bCb1", "F1bCb2", "F1bCb3", "F1bCb4", "F1bCb5", "F1bCb6", "F1bCb7", "F1bCb8", "F1bCb9", "F1bCb10", "F1bCb11", "F1bCb12", "F1bCb13", "F1bCb14", "F1aCa1", "F1aCa2", "F1aCa3", "F1aCa4", "F1aCa5", "F1aCa6", "F1aCa7", "F1aCa8", "F1aCa9", "F1aCa10", "F1aCa11", "F1aCa12", "P1Cc1", "P1Cc2", "P1Cc3", "P1Cc4", "P1Cc5", "P1Cc6", "P1Cc7", "P1Cc8", "P1Cc9", "P1Cc10", "P1Cc11", "F2aIb1", "F2aIb2", "F2aIb3", "F2aIb4", "F2aIb5", "F2aIb6", "F2aIb7", "F2aIb8", "F2aIb9", "F2aIb10", "F3IaOF7", "F3IaOF8", "F3IaOF9", "F3IaOF10", "F3IaOF11", "F3IaOF12", "F3IaOF13", "F3IaOF14", "F3IaOF15", "F3IaOF16", "T2aHa4", "T2aHa5", "T2aHa6", "T2aHa7", "T2aHa8", "T2aHa9", "T2aHa10", "T2aHa11", "T2aHa12", "T2bHb5", "T2bHb6", "T2bHb7", "T2bHb8", "T2bHb9", "T2bHb10", "T2bHb11", "T2bHb12", "T2aHa1"], "btbank6": ["F2eIc1", "F2eIc2", "F2eIc3", "F2eIc4", "F2eIc5", "F2eIc6", "F2eIc7", "F2eIc8", "F2eIc9", "F2eIc10", "F2eIc11", "F2eIc12", "F2eIc13", "F2eIc14", "F2eIc15", "F2eIc16", "F3bOFb1", "F3bOFb2", "F3bOFb3", "F3bOFb4", "F3bOFb5", "F3bOFb6", "F3bOFb7", "F3bOFb8", "F3bOFb9", "F3bOFb10", "F3bOFb11", "F3bOFb12", "F3bOFb13", "F3bOFb14", "F3bOFb15", "F3bOFb16", "F2c1", "F2c2", "F2c3", "F2c4", "F2c5", "F2c6", "F2c7", "F2c8", "F2c9", "F2c10", "F2c11", "F2c12", "F2c13", "F2c14", "F2c15", "F2c16", "F2bIa1", "F2bIa2", "F2bIa3", "F2bIa4", "F2bIa5", "F2bIa6", "F2bIa7", "F2bIa8", "F2bIa9", "F2bIa10", "F2bIa11", "F2bIa12", "F2bIa13", "F2bIa14", "F2bIa15", "F2bIa16", "F3cOFc1", "F3cOFc2", "F3cOFc3", "F3cOFc4", "F3cOFc5", "F3cOFc6", "F3cOFc7", "F3cOFc8", "F3cOFc9", "F3cOFc10", "F3cOFc11", "F3cOFc12", "F3cOFc13", "F3cOFc14", "F3cOFc15", "F3cOFc16", "F2dCb1", "F2dCb2", "F2dCb3", "F2dCb4", "F2dCb5", "F2dCb6", "F2dCb7", "F2dCb8", "F2dCb9", "F2dCb10", "F2dCb11", "F2dCb12", "F2dCb13", "F2dCb14", "F3aOFa1", "F3aOFa2", "F3aOFa3", "F3aOFa4", "F3aOFa5", "F3aOFa6", "F3aOFa7", "F3aOFa8", "F3aOFa9", "F3aOFa10", "F3aOFa11", "F3aOFa12", "F3aOFa13", "F3aOFa14", "F3dIb1", "F3dIb2", "F3dIb3", "F3dIb4", "F3dIb5", "F3dIb6", "F3dIb7", "F3dIb8", "F3dIb9", "F3dIb10"], "btbank7": ["LF2aIa1", "LF2aIa2", "LF2aIa3", "LF2aIa4", "LF2aIa5", "LF2aIa6", "LF2aIa7", "LF2aIa8", "LF2aIa9", "LF2aIa10", "LF2aIa11", "LF2aIa12", "LF2aIa13", "LF2aIa14", "LF2aIa15", "LF2aIa16", "LF1bIb1", "LF1bIb2", "LF1bIb3", "LF1bIb4", "LF1bIb5", "LF1bIb6", "LF1bIb7", "LF1bIb8", "LF1bIb9", "LF1bIb10", "LF1bIb11", "LF1bIb12", "LF1bIb13", "LF1bIb14", "LF1bIb15", "LF1bIb16", "LF3bOFb1", "LF3bOFb2", "LF3bOFb3", "LF3bOFb4", "LF3bOFb5", "LF3bOFb6", "LF3bOFb7", "LF3bOFb8", "LF3bOFb9", "LF3bOFb10", "LF3bOFb11", "LF3bOFb12", "LF3bOFb13", "LF3bOFb14", "LF3bOFb15", "LF3bOFb16", "RF3aOFa1", "RF3aOFa2", "RF3aOFa3", "RF3aOFa4", "RF3aOFa5", "RF3aOFa6", "RF3aOFa7", "RF3aOFa8", "RF3aOFa9", "RF3aOFa10", "RF3aOFa11", "RF3aOFa12", "RF3aOFa13", "RF3aOFa14", "RF3aOFa15", "RF3aOFa16", "LF3aOFa1", "LF3aOFa2", "LF3aOFa3", "LF3aOFa4", "LF3aOFa5", "LF3aOFa6", "LF3aOFa7", "LF3aOFa8", "LF3aOFa9", "LF3aOFa10", "LF3aOFa11", "LF3aOFa12", "LF3aOFa13", "LF3aOFa14", "LF3aOFa15", "LF3aOFa16", "RF3bOFb1", "RF3bOFb2", "RF3bOFb3", "RF3bOFb4", "RF3bOFb5", "RF3bOFb6", "RF3bOFb7", "RF3bOFb8", "RF3bOFb9", "RF3bOFb10", "RF3bOFb11", "RF3bOFb12", "RF3bOFb13", "RF3bOFb14", "RF3bOFb15", "RF3bOFb16", "LT2H1", "LT2H2", "LT2H3", "LT2H4", "LT2H5", "LT2H6", "LT2H7", "LT2H8", "LT2H9", "LT2H10", "LT2H11", "LT2H12", "LT2H13", "LT2H14", "LT2H15", "LT2H16", "LF2c1", "LF2c2", "LF2c3", "LF2c4", "LF2c5", "LF2c6", "LF2c7", "LF2c8"], "btbank8": ["F1aCaOFb1", "F1aCaOFb2", "F1aCaOFb3", "F1aCaOFb4", "F1aCaOFb5", "F1aCaOFb6", "F1aCaOFb7", "F1aCaOFb8", "F1aCaOFb9", "F1aCaOFb10", "F1aCaOFb11", "F1aCaOFb12", "F1aCaOFb13", "F1aCaOFb14", "F1aCaOFb15", "F1aCaOFb16", "F3bIaOFc1", "F3bIaOFc2", "F3bIaOFc3", "F3bIaOFc4", "F3bIaOFc5", "F3bIaOFc6", "F3bIaOFc7", "F3bIaOFc8", "F3bIaOFc9", "F3bIaOFc10", "F3bIaOFc11", "F3bIaOFc12", "F3bIaOFc13", "F3bIaOFc14", "F3bIaOFc15", "F3bIaOFc16", "F2bCb1", "F2bCb2", "F2bCb3", "F2bCb4", "F2bCb5", "F2bCb6", "F2bCb7", "F2bCb8", "F2bCb9", "F2bCb10", "F2bCb11", "F2bCb12", "F2bCb13", "F2bCb14", "F2cCc1", "F2cCc2", "F2cCc3", "F2cCc4", "F2cCc5", "F2cCc6", "F2cCc7", "F2cCc8", "F2cCc9", "F2cCc10", "F2cCc11", "F2cCc12", "F2cCc13", "F2cCc14", "F1bCd1", "F1bCd2", "F1bCd3", "F1bCd4", "F1bCd5", "F1bCd6", "F1bCd7", "F1bCd8", "F1bCd9", "F1bCd10", "F1bCd11", "F1bCd12", "F2aIb1", "F2aIb2", "F2aIb3", "F2aIb4", "F2aIb5", "F2aIb6", "F2aIb7", "F2aIb8", "F2aIb9", "F2aIb10", "T3H3", "T3H4", "T3H5", "T3H6", "T3H7", "T3H8", "T3H9", "T3H10", "T3H11", "T3H12", "F3aOFa1", "F3aOFa2", "F3aOFa3", "F3aOFa4", "F3aOFa5", "F3aOFa6", "F3aOFa7", "F3aOFa8", "F3aOFa9", "T1aId1", "T1aId2", "T1aId3", "T1aId4", "T1aId5", "T1aId6", "T1aId7", "T1aId8", "F3cIc1", "F3cIc2", "F3cIc3", "F3cIc4", "F3cIc5", "F3cIc6", "F3cIc7", "F3cIc8"], "btbank9": ["P1C1", "P1C2", "P1C3", "P1C4", "P1C5", "P1C6", "P1C7", "P1C8", "P1C9", "P1C10", "P1C11", "P1C12", "P1C13", "P1C14", "P2d1", "P2d2", "P2d3", "P2d4", "P2d5", "P2d6", "P2d7", "P2d8", "P2d9", "P2d10", "P2d11", "P2d12", "P2d13", "P2d14", "T1d1", "T1d2", "T1d3", "T1d4", "T1d5", "T1d6", "T1d7", "T1d8", "T1d9", "T1d10", "T1d11", "T1d12", "T1aI1", "T1aI2", "T1aI3", "T1aI4", "T1aI5", "T1aI6", "T1aI7", "T1aI8", "P2a1", "P2a2", "P2a3", "P2a4", "P2a5", "P2a6", "P2a7", "P2a8", "P2e1", "P2e2", "P2e3", "P2e4", "P2e5", "P2e6", "P2e7", "P2e8", "P2b1", "P2b2", "P2b3", "P2b4", "P2b5", "P2b6", "P2b7", "P2b8", "F3b1", "F3b2", "F3b3", "F3b4", "F3b5", "F3b6", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6", "T1c1", "T1c2", "T1c3", "T1c4", "T1c5", "P2c6", "P2c7", "P2c8", "P2c9", "P2c10", "F3a5", "F3a6", "F3a3", "P2c1", "P2c4"], "btbank10": ["F2aOFb1", "F2aOFb2", "F2aOFb3", "F2aOFb4", "F2aOFb5", "F2aOFb6", "F2aOFb7", "F2aOFb8", "F2aOFb9", "F2aOFb10", "F2aOFb11", "F2aOFb12", "F2aOFb13", "F2aOFb14", "F2aOFb15", "F2aOFb16", "F2bCa1", "F2bCa2", "F2bCa3", "F2bCa4", "F2bCa5", "F2bCa6", "F2bCa7", "F2bCa8", "F2bCa9", "F2bCa10", "F2bCa11", "F2bCa12", "F2bCa13", "F2bCa14", "F2bCa15", "F2bCa16", "F3aOFc1", "F3aOFc2", "F3aOFc3", "F3aOFc4", "F3aOFc5", "F3aOFc6", "F3aOFc7", "F3aOFc8", "F3aOFc9", "F3aOFc10", "F3aOFc11", "F3aOFc12", "F3aOFc13", "F3aOFc14", "F3aOFc15", "F3aOFc16", "P2cCc1", "P2cCc2", "P2cCc3", "P2cCc4", "P2cCc5", "P2cCc6", "P2cCc7", "P2cCc8", "P2cCc9", "P2cCc10", "P2cCc11", "P2cCc12", "P2cCc13", "P2cCc14", "P2cCc15", "F2dCb1", "F2dCb2", "F2dCb3", "F2dCb4", "F2dCb5", "F2dCb6", "F2dCb7", "F2dCb8", "F2dCb9", "F2dCb10", "F2dCb11", "F2dCb12", "F2dCb13", "F2dCb14", "T2bH4", "T2bH5", "T2bH6", "T2bH7", "T2bH8", "T2bH9", "T2bH10", "T2bH11", "T2bH12", "T2bH13", "T2bH14", "T2bH15", "T2bH16", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5", "T1cIf6", "T1cIf7", "T1cIf8", "T1cIf9", "T1cIf10", "T1cIf11", "T1cIf12", "P2aIe1", "P2aIe2", "P2aIe3", "P2aIe4", "P2aIe5", "P2aIe6", "P2aIe7", "P2aIe8", "P2aIe9", "P2aIe10", "P2aIe11", "P2aIe12", "F10Fa8", "F10Fa9", "F10Fa10", "F10Fa11", "F10Fa12"]}