# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

ROOT_DIR = "/om2/user/zaho/braintreebank/braintreebank" # Root directory for the braintreebank data
SAVE_SUBJECT_TRIAL_DF_DIR = "./btbench_subject_metadata"
SAMPLING_RATE = 2048 # Sampling rate

START_NEURAL_DATA_BEFORE_WORD_ONSET = 0.5 # in seconds
END_NEURAL_DATA_AFTER_WORD_ONSET = 2 # in seconds
NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2 # how many seconds to wait between the last word off-set and the start of a "non-verbal" chunk
NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP = 0.8 # proportion of overlap between consecutive nonverbal chunks (0 means no overlap)

# some sanity check code as well as disabling file locking for HDF5 files
assert NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP >= 0 and NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP < 1, "NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable file locking for HDF5 files. This is helpful for parallel processing.

# Standardizing pretraining and evaluation subjects and trials
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
all_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in all_subject_trials]
eval_subject_trials = [(1, 2), (2, 6), (3, 0), (6, 4), (7, 0), (4, 1), (10, 0)]
eval_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in eval_subject_trials]
train_subject_trials = [st for st in all_subject_trials if st not in eval_subject_trials]