# Part of the code is adapted from https://braintreebank.dev/, file "quickstart.ipynb"
import os
import json
import pandas as pd
import numpy as np
from .btbench_config import *

# Data frames column IDs
start_col, end_col, lbl_col = 'start', 'end', 'pos'
trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'
word_time_col, word_text_col, is_onset_col, is_offset_col = 'word_time', 'text', 'is_onset', 'is_offset'
def obtain_aligned_words_df(sub_id, trial_id, verbose=True, save_to_dir=None):
    # Path to neural data h5 file
    neural_data_file = os.path.join(ROOT_DIR, f'sub_{sub_id}_trial{trial_id:03}.h5')
    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    # Path format to trial metadata json file
    metadata_file = os.path.join(ROOT_DIR, f'subject_metadata/sub_{sub_id}_trial{trial_id:03}_metadata.json')
    with open(metadata_file, 'r') as f:
        meta_dict = json.load(f)
        title = meta_dict['title']
        movie_id = meta_dict['filename']
    # # Path to transcript csv file
    transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{movie_id}/features.csv')
    # Path format to electrode labels file -- mapping each ID to an subject specific label
    electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{sub_id}/electrode_labels.json')

    if verbose: print(f"Computing words dataframe for subject {sub_id} trial {trial_id}")
    trigs_df = pd.read_csv(trigger_times_file)
    words_df = pd.read_csv(transcript_file_format.format(movie_id)).set_index('Unnamed: 0')
    words_df = words_df.drop(['word_diff', 'onset_diff'], axis=1) # remove those columns because they are unnecessary and cause excessive filtering with NaN values
    words_df = words_df.dropna().reset_index(drop=True)

    # Vectorized sample index estimation
    def add_estimated_sample_index_vectorized(w_df, t_df):
        last_t = t_df[trig_time_col].iloc[-1]
        last_t_idx = t_df[trig_idx_col].idxmax()
        w_df = w_df[w_df[start_col] < last_t].copy()

        # Vectorized nearest trigger finding
        start_indices = np.searchsorted(t_df[trig_time_col].values, w_df[start_col].values)
        end_indices = np.searchsorted(t_df[trig_time_col].values, w_df[end_col].values)
        end_indices = np.minimum(end_indices, last_t_idx) # handle the edge case where movie cuts off right at the word
        start_indices = np.maximum(start_indices, 0) # handle the edge case where movie starts right at the word
        
        # Vectorized sample index calculation
        w_df[est_idx_col] = np.round(
            t_df.loc[start_indices, trig_idx_col].values + 
            (w_df[start_col].values - t_df.loc[start_indices, trig_time_col].values) * SAMPLING_RATE
        )
        w_df[est_end_idx_col] = np.round(
            t_df.loc[end_indices, trig_idx_col].values + 
            (w_df[end_col].values - t_df.loc[end_indices, trig_time_col].values) * SAMPLING_RATE
        )
        return w_df

    words_df = add_estimated_sample_index_vectorized(words_df, trigs_df)  # align all words to data samples
    words_df = words_df.dropna().reset_index(drop=True)  # no need to keep words with no start time

    # Remove words that would create invalid windows (too close to the start or end of the trial)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]
    valid_words = (words_df[est_idx_col] >= int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)) & \
                 (words_df[est_end_idx_col] <= int(total_samples - END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE))
    words_df = words_df[valid_words].reset_index(drop=True)

    if verbose: print(f"Kept {len(words_df)} words after removing invalid windows")
    # Save the processed words dataframe
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        words_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_words_df.csv', index=False)
    return words_df


def obtain_nonverbal_df(sub_id, trial_id, words_df, verbose=True, save_to_dir=None):
    window_length = int((START_NEURAL_DATA_BEFORE_WORD_ONSET + END_NEURAL_DATA_AFTER_WORD_ONSET) * SAMPLING_RATE)

    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    trigs_df = pd.read_csv(trigger_times_file)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]

    nonverbal_windows = []
    # Iterate through consecutive word pairs
    for i in range(len(words_df)-1):
        current_word = words_df.iloc[i]
        next_word = words_df.iloc[i+1]
        
        # Calculate gap between current word offset and next word onset
        current_word_offset = int(current_word[est_end_idx_col] + int((NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME+START_NEURAL_DATA_BEFORE_WORD_ONSET) * SAMPLING_RATE))
        next_word_onset = int(next_word[est_idx_col] - int((NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME+END_NEURAL_DATA_AFTER_WORD_ONSET) * SAMPLING_RATE))
        gap_samples = next_word_onset - current_word_offset

        window_start_sample = current_word_offset
        window_end_sample = window_start_sample + window_length

        # If gap is large enough for a window
        while (gap_samples >= window_length) and (window_end_sample <= total_samples):
            # Add new row to nonverbal_df with window start/end times and sample indices
            nonverbal_windows.append({
                start_col: window_start_sample / SAMPLING_RATE,
                end_col: window_end_sample / SAMPLING_RATE, 
                est_idx_col: window_start_sample,
                est_end_idx_col: window_end_sample
            })
            window_start_sample += int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
            window_end_sample += int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
            gap_samples -= int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
    nonverbal_df = pd.DataFrame(nonverbal_windows)        
    print(f"Kept {len(nonverbal_df)} nonverbal windows")
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        nonverbal_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_nonverbal_df.csv', index=False)
    return nonverbal_df


if __name__ == "__main__":
    all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
    for sub_id, trial_id in all_subject_trials:
        words_df = obtain_aligned_words_df(sub_id, trial_id, save_to_dir=SAVE_SUBJECT_TRIAL_DF_DIR)
        nonverbal_df = obtain_nonverbal_df(sub_id, trial_id, words_df, save_to_dir=SAVE_SUBJECT_TRIAL_DF_DIR)