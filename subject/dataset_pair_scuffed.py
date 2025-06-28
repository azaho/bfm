import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from subject.braintreebank import BrainTreebankSubject
from subject.mgh2024 import MGH2024Subject
from subject.dataset import SubjectTrialDataset # for testing purposes
from training_setup.training_config import log
from multiprocessing import Pool
import torch.multiprocessing as mp
import random
import numpy as np
import pandas as pd
import os
from evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE

class SubjectTrialPairDataset(Dataset):
    def __init__(self, subject, trial_id, window_size, dtype=torch.float32, output_metadata=False, output_electrode_labels=False,
                 subject_b=None, trial_id_b=None, movie_times=None, trigger_times_dir=None, sampling_rate=None):
        """
        Args:
            subject (BrainTreebankSubject or MGHSubject): Subject object
            trial_id (int): Trial ID
            dtype (torch.dtype): Data type to load the data in (float32, bfloat16)
            window_size (int): Number of time samples per data item
            subject_b, trial_id_b, movie_times, trigger_times_dir, sampling_rate: for paired/contrastive mode
        """
        self.subject = subject
        self.trial_id = trial_id
        self.window_size = window_size
        self.dtype = dtype
        self.output_metadata = output_metadata
        self.output_electrode_labels = output_electrode_labels
        self.subject_b = subject_b
        self.trial_id_b = trial_id_b
        self.movie_times = movie_times
        self.trigger_times_dir = trigger_times_dir
        self.sampling_rate = sampling_rate

        # Load neural data for subject A
        subject.load_neural_data(trial_id)
        self.n_windows = self.subject.electrode_data_length[trial_id] // self.window_size

        # Use the alignment helper
        paired_args = [subject_b, trial_id_b, movie_times, trigger_times_dir, sampling_rate]

        # some argument checks to prevent errors
        if not all(arg is not None for arg in paired_args):
            raise ValueError("All paired mode arguments (subject_b, trial_id_b, movie_times, trigger_times_dir, sampling_rate) must be provided for paired mode.")
        if self.subject_b is None:
            raise ValueError("subject_b must not be None.")
        
        self.subject_b.load_neural_data(trial_id_b)
        
        # Calculate aligned indices directly
        sub_id_a = int(self.subject.subject_identifier.replace("btbank", ""))
        sub_id_b = int(self.subject_b.subject_identifier.replace("btbank", ""))
        self.indices = self.obtain_neural_data_index(sub_id_a, trial_id, movie_times)
        self.indices_b = self.obtain_neural_data_index(sub_id_b, trial_id_b, movie_times)
        self.n_windows = len(self.indices)

    def obtain_neural_data_index(self, sub_id, trial_id, movie_times):
        # Path to trigger times csv file
        trigger_times_file = os.path.join(self.trigger_times_dir, f'sub_{sub_id}_trial{int(trial_id):03}_timings.csv')
        trigs_df = pd.read_csv(trigger_times_file)
        trig_time_col, trig_idx_col = 'movie_time', 'index'
        # Vectorized nearest trigger finding
        start_indices = np.searchsorted(trigs_df[trig_time_col].values, movie_times)

        # Clamp indices to valid range (0 to len-1)
        start_indices = np.maximum(start_indices, 0)
        # was throwing an error when the movie time was greater than the last trigger time,
        # so added another clamping step
        start_indices = np.minimum(start_indices, len(trigs_df) - 1)

        # Vectorized sample index calculation
        return np.round(
            trigs_df.loc[start_indices, trig_idx_col].values +
            (movie_times - trigs_df.loc[start_indices, trig_time_col].values) * self.sampling_rate
        ).astype(int)

    def get_aligned_indices(self):
        return self.indices, self.indices_b

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        #if self.subject_b is None:
        #    raise ValueError("subject_b cannot be None")
        idx_a = self.indices[idx]
        
        window = self.subject.get_all_electrode_data(self.trial_id, idx_a, idx_a + self.window_size).to(dtype=self.dtype)

        if self.subject_b is not None:
            idx_b = self.indices_b[idx]
            window_b = self.subject_b.get_all_electrode_data(self.trial_id_b, idx_b, idx_b + self.window_size).to(dtype=self.dtype)
        else:
            window_b = None
        
        # Create nested structure that matches original format
        output = {
            'data': window,
            'data_b': window_b,
        }
        
        if self.output_metadata:
            output['metadata'] = {
                'subject_identifier': self.subject.subject_identifier,
                'trial_id': self.trial_id,
                'sampling_rate': self.subject.get_sampling_rate(self.trial_id),
            }
            output['metadata_b'] = {
                'subject_identifier': self.subject_b.subject_identifier,
                'trial_id': self.trial_id_b,
                'sampling_rate': self.subject_b.get_sampling_rate(self.trial_id_b),
            } if self.subject_b is not None else None
            output['subject_trial'] = (self.subject.subject_identifier, self.trial_id)
            output['subject_trial_b'] = (self.subject_b.subject_identifier, self.trial_id_b) if self.subject_b is not None else None
            
        if self.output_electrode_labels:
            output['electrode_labels'] = self.subject.electrode_labels
            output['electrode_labels_b'] = self.subject_b.electrode_labels if self.subject_b is not None else None
        return output

class PreprocessCollatorPair:
    def __init__(self, preprocess_functions=[]):
        self.preprocess_functions = preprocess_functions

    def __call__(self, batch):
        # batch is now a list of dictionaries with nested structure

        # Process each item in batch
        output = {
            'data': torch.stack([item['data'] for item in batch]),
            'data_b': torch.stack([item['data_b'] for item in batch])
        }
        
        # Handle electrode labels
        if 'electrode_labels' in batch[0]:
            # made a list of a list
            output['electrode_labels'] = [batch[0]['electrode_labels']]
        if 'electrode_labels_b' in batch[0]:
            # made a list of a list
            output['electrode_labels_b'] = [batch[0]['electrode_labels_b']]
        
        # Handle metadata
        if 'metadata' in batch[0]:
            output['metadata'] = batch[0]['metadata']
        if 'metadata_b' in batch[0]:
            output['metadata_b'] = batch[0]['metadata_b']

        # If any preprocess functions are provided, apply them to the batch
        for preprocess_function in self.preprocess_functions:
            output = preprocess_function(output)
            
        # Copy through any other fields that don't need processing
        for key in batch[0].keys():
            if key not in output and key != 'data' and key != 'data_b':
                if isinstance(batch[0][key], dict):
                    # Handle nested dictionaries
                    output[key] = {}
                    for subkey in batch[0][key].keys():
                        output[key][subkey] = [item[key][subkey] for item in batch]
                        if isinstance(batch[0][key][subkey], torch.Tensor):
                            output[key][subkey] = torch.stack(output[key][subkey])
                else:
                    # Handle simple fields
                    output[key] = [item[key] for item in batch]
                    if isinstance(batch[0][key], torch.Tensor):
                        output[key] = torch.stack(output[key])
        
        return output

# based on random subject/trial pairs where they're watching the same movie, make a batch
# TODO: ask about logic in this class
class SubjectBatchPairSampler(torch.utils.data.Sampler):
        def __init__(self, dataset_sizes, batch_size, shuffle=True, drop_last=True):
            self.dataset_sizes = dataset_sizes
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.drop_last = drop_last
        def __iter__(self):
            # Create batches for each subject
            all_batches = []
            start_idx = 0
            
            # size is the number of windows in the subject
            for size in self.dataset_sizes:
                # Create indices for this subject
                subject_indices = list(range(start_idx, start_idx + size))

                if self.shuffle:
                    random.shuffle(subject_indices)
                
                # Create batches for all subjects
                # the batches are of the indices of the windows in the subject
                subject_batches = [subject_indices[i:i + self.batch_size] 
                                for i in range(0, len(subject_indices), self.batch_size)
                                if not self.drop_last or i + self.batch_size <= len(subject_indices)]
                all_batches.extend(subject_batches)
                start_idx += size
            
            # Shuffle the order of batches if needed
            if self.shuffle:
                random.shuffle(all_batches)
                
            return iter(all_batches)
        
        def __len__(self):
            if self.drop_last:
                return sum(size // self.batch_size for size in self.dataset_sizes)
            return sum((size + self.batch_size - 1) // self.batch_size 
                    for size in self.dataset_sizes)

def load_subjects(train_subject_trials, eval_subject_trials, dtype, cache=True, allow_corrupted=False):
    all_subject_identifiers = [subject_identifier for subject_identifier, trial_id in train_subject_trials]
    all_subject_identifiers += [subject_identifier for subject_identifier, trial_id in eval_subject_trials]
    all_subject_identifiers = list(set(all_subject_identifiers))
    all_subjects = {}

    for subject_identifier in all_subject_identifiers:
        log(f"loading subject {subject_identifier}...", indent=1, priority=1)
        if "btbank" in subject_identifier:
            subject_id = int(subject_identifier.replace("btbank", ""))
            all_subjects[subject_identifier] = BrainTreebankSubject(subject_id, dtype=dtype, cache=cache, allow_corrupted=allow_corrupted)
        elif "mgh" in subject_identifier:
            subject_id = int(subject_identifier.replace("mgh", ""))
            all_subjects[subject_identifier] = MGH2024Subject(subject_id, dtype=dtype, cache=cache, allow_corrupted=allow_corrupted)
        else:
            raise ValueError(f"Unknown subject identifier: {subject_identifier}")

    return all_subjects

if __name__ == "__main__":
    # Create two subjects for testing
    subject_a = BrainTreebankSubject(3, cache=False)
    subject_b = BrainTreebankSubject(4, cache=False)

    sampling_rate = SAMPLING_RATE  # Use the proper sampling rate
    window_size = 100
    
    # For testing, we need to provide all the paired arguments
    # Use the proper trigger times directory from the braintreebank data
    # Calculate how many windows fit in 100 seconds
    total_time_seconds = 100
    window_time_seconds = window_size / sampling_rate  # 100/2048 = 0.05 seconds per window
    n_windows = int(total_time_seconds / window_time_seconds)  # Should be ~2048 windows
    movie_times = np.linspace(0, total_time_seconds, n_windows)  # Create consecutive windows
    trigger_times_dir = os.path.join(ROOT_DIR, "subject_timings")  # Use the proper directory
    
    dataset = SubjectTrialPairDataset(
        # window size is 100, number of samples in each window
        # this is the number of windows in the dataset, whereas 100 is the number
        # of samples in each window
        subject_a, 0, window_size, torch.float32, subject_b=subject_b, trial_id_b=0,
        movie_times=movie_times, trigger_times_dir=trigger_times_dir, sampling_rate=sampling_rate
    )
    print("Length of dataset:", len(dataset))
    print("Shape of dataset[0]['data']:", dataset[0]['data'].shape)
    print("Shape of dataset[0]['data_b']:", dataset[0]['data_b'].shape)
    print("Subject trial A:", dataset[0]['subject_trial'])
    print("Subject trial B:", dataset[0]['subject_trial_b'])

    subject = BrainTreebankSubject(3, cache=False)
    dataset = SubjectTrialDataset(subject, 0, 100, torch.float32)
    print("Length of dataset:", len(dataset))
    print("Shape of dataset[0]:", dataset[0]['data'].shape)

    subject = BrainTreebankSubject(4, cache=False)
    dataset = SubjectTrialDataset(subject, 0, 100, torch.float32)
    print("Length of dataset:", len(dataset))
    print("Shape of dataset[0]:", dataset[0]['data'].shape)