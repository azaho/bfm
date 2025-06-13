# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

ROOT_DIR = "/om2/user/hmor/ajile12/000055"
SAMPLING_RATE = 512 # Sampling rate

START_NEURAL_DATA_BEFORE_REACH_ONSET = 0.5 # in seconds
END_NEURAL_DATA_AFTER_REACH_ONSET = 1.5 # in seconds
NEURAL_DATA_NONREACH_WINDOW_PADDING_TIME = 2 # how many seconds to wait between the last reach off-set and the start of a "non-reach" chunk
NEURAL_DATA_NONREACH_WINDOW_OVERLAP = 0.0 # proportion of overlap between consecutive nonreach chunks (0 means no overlap)
NEURAL_DATA_NONREACH_EPOCH_TYPES = ['TV', 'Inactive']

# some sanity check code as well as disabling file locking for HDF5 files
assert NEURAL_DATA_NONREACH_WINDOW_OVERLAP >= 0 and NEURAL_DATA_NONREACH_WINDOW_OVERLAP < 1, "NONREACH_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable file locking for HDF5 files. This is helpful for parallel processing.
