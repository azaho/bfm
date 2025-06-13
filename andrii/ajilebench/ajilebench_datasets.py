import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from .ajilebench_config import *
from subject_ajile12 import AjileSubject

# TODO: fix bug
# Problem left: some values are NaN. To reproduce, run single electrode on subject 1 trial 3 with reach_onset eval_name
# I suspect that it has something to do with the range value of the time offset being outside of the expected one (if its <0)
# and maybe some values are just NaN there
# I think two solutions are either just remove the indices with nan values and send a warning (more principled but requires going through the whole data on loading the class)
# or just substitute the nan values with 0s (faster but less principled)
# ASSUMPTION BROKEN: WARNING: Found NaN values in X_test for electrode 18GRID, trial 4, time bin [1.0, 1.125]

def _get_nonreach_start_times(subject, trial_id, 
                              neural_data_nonreach_epoch_types=NEURAL_DATA_NONREACH_EPOCH_TYPES, 
                              neural_data_nonreach_window_padding_time=NEURAL_DATA_NONREACH_WINDOW_PADDING_TIME,
                              neural_data_nonreach_window_overlap=NEURAL_DATA_NONREACH_WINDOW_OVERLAP,
                              neural_data_nonreach_window_size=START_NEURAL_DATA_BEFORE_REACH_ONSET+END_NEURAL_DATA_AFTER_REACH_ONSET):
    reach_events = subject.get_reach_events(trial_id)
    epochs = subject.get_epochs(trial_id)

    start_times = []
    end_times = []

    epoch_start_times = epochs['start_time']
    epoch_end_times = epochs['stop_time']
    epoch_labels = epochs['labels']

    reach_start_times = np.array(reach_events['start_time'])-neural_data_nonreach_window_padding_time
    reach_end_times = np.array(reach_events['stop_time'])+neural_data_nonreach_window_padding_time

    for epoch_i in range(len(epoch_labels)):
        epoch_label = epoch_labels[epoch_i]
        if str(epoch_label) not in neural_data_nonreach_epoch_types: continue # Only consider allowed epoch types

        start_time = epoch_start_times[epoch_i]
        end_time = epoch_end_times[epoch_i]
        current_time = start_time
        while current_time + neural_data_nonreach_window_size < end_time:
            window_start = current_time
            window_end = current_time + neural_data_nonreach_window_size

            intersects_reach = np.any(
                ((window_start < reach_start_times) & (window_end > reach_start_times)) | \
                ((window_start < reach_end_times) & (window_end > reach_end_times)) | \
                ((window_start > reach_start_times) & (window_end < reach_end_times))
            )
            if intersects_reach:
                current_time += neural_data_nonreach_window_padding_time
                continue

            start_times.append(window_start)
            end_times.append(window_end)
            current_time += neural_data_nonreach_window_size * (1 - neural_data_nonreach_window_overlap)
    return np.array(start_times), np.array(end_times)


all_tasks = ["reach_onset"] # TODO: add more tasks, based on qualitative features of the reaches
class AjileSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, 
                 start_neural_data_before_reach_onset=int(START_NEURAL_DATA_BEFORE_REACH_ONSET * SAMPLING_RATE),
                 end_neural_data_after_reach_onset=int(END_NEURAL_DATA_AFTER_REACH_ONSET * SAMPLING_RATE),
                 replace_nan_with=0):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
                Options for eval_name: 
                    reach_onset

            output_indices (bool): 
                if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
                if False, the dataset will output the neural data directly
            
            start_neural_data_before_reach_onset (int): the number of samples to start the neural data before each reach onset
            end_neural_data_after_reach_onset (int): the number of samples to end the neural data after each reach onset
        """
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.output_indices = output_indices
        self.start_neural_data_before_reach_onset = start_neural_data_before_reach_onset
        self.end_neural_data_after_reach_onset = end_neural_data_after_reach_onset    
        self.replace_nan_with = replace_nan_with

        nonreach_start_times, _ = _get_nonreach_start_times(subject, trial_id)
        self.negative_indices = (nonreach_start_times * subject.get_sampling_rate()).astype(np.int32)
        reach_start_times = np.array(subject.get_reach_events(trial_id)['start_time'])
        self.positive_indices = (reach_start_times * subject.get_sampling_rate()).astype(np.int32) - self.start_neural_data_before_reach_onset

        # Note that sometimes the times end up being outside of the range of the neural data
        # In this case, we just remove the index
        self.positive_indices = self.positive_indices[self.positive_indices < subject.get_electrode_data_length(trial_id)]
        self.negative_indices = self.negative_indices[self.negative_indices < subject.get_electrode_data_length(trial_id)]
        
        # balancing the classes
        min_len = min(len(self.positive_indices), len(self.negative_indices))
        self.positive_indices = np.sort(np.random.choice(self.positive_indices, size=min_len, replace=False))
        self.negative_indices = np.sort(np.random.choice(self.negative_indices, size=min_len, replace=False))
        self.n_samples = len(self.positive_indices) + len(self.negative_indices)

    def _get_neural_data(self, window_from, window_to):
        if not self.output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            input = input.to(dtype=self.dtype)
            input[torch.isnan(input)] = self.replace_nan_with
            return input
        else:
            return window_from, window_to # just return the window indices

    def __getitem__(self, idx):
        if idx % 2 == 0: # even indices are positive samples
            neural_start_idx = self.positive_indices[idx//2]
            neural_end_idx = neural_start_idx + self.end_neural_data_after_reach_onset + self.start_neural_data_before_reach_onset
            input = self._get_neural_data(neural_start_idx, neural_end_idx)
            return input, 1
        else: # odd indices are negative samples
            neural_start_idx = self.negative_indices[idx//2]
            neural_end_idx = neural_start_idx + self.end_neural_data_after_reach_onset + self.start_neural_data_before_reach_onset
            input = self._get_neural_data(neural_start_idx, neural_end_idx)
            return input, 0
    def __len__(self):
        return self.n_samples
    
if __name__ == "__main__":
    subject = AjileSubject(subject_id=1)
    trial_id = 3
    dataset = AjileSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name="reach_onset")
    print(len(dataset))
    print(dataset[0], dataset[0][0].shape)
