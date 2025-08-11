import torch
from torch.utils.data import Dataset
from subject.podcast import PodcastSubject
from subject.dataset import SubjectTrialDataset
from training_setup.training_config import log
import random
import numpy as np

class PodcastTrialPairDataset(Dataset):
    """
    Paired dataset for Podcast ECoG data.
    
    This dataset creates pairs of neural data from different subjects 
    listening to the same podcast content at the same time points.
    Since all subjects listen to the same 30-minute podcast, we can
    align their neural responses temporally.
    """
    
    def __init__(self, subject_a, subject_b, window_size, dtype=torch.float32, 
                 output_metadata=False, output_electrode_labels=False,
                 alignment_strategy='temporal'):
        """
        Args:
            subject_a (PodcastSubject): First subject
            subject_b (PodcastSubject): Second subject  
            window_size (int): Number of time samples per data item
            dtype (torch.dtype): Data type to load the data in
            output_metadata (bool): Whether to output metadata
            output_electrode_labels (bool): Whether to output electrode labels
            alignment_strategy (str): How to align subjects ('temporal' for same time points)
        """
        self.subject_a = subject_a
        self.subject_b = subject_b
        self.window_size = window_size
        self.dtype = dtype
        self.output_metadata = output_metadata
        self.output_electrode_labels = output_electrode_labels
        self.alignment_strategy = alignment_strategy
        
        # Both subjects listen to the same podcast, so they have the same session_id
        self.session_id = 0
        
        # Load neural data for both subjects
        self.subject_a.load_neural_data(self.session_id)
        self.subject_b.load_neural_data(self.session_id)
        
        # Calculate number of windows based on the shorter data length
        data_length_a = self.subject_a.electrode_data_length[self.session_id]
        data_length_b = self.subject_b.electrode_data_length[self.session_id]
        self.data_length = min(data_length_a, data_length_b)
        
        # Calculate number of windows
        self.n_windows = self.data_length // self.window_size
        
        # Set electrode labels
        self.electrode_labels_a = list(self.subject_a.electrode_labels)
        self.electrode_labels_b = list(self.subject_b.electrode_labels)
        
        # Set verbose flag for logging
        self.verbose = True
        
        if self.verbose:
            log(f"Created podcast pair dataset: {subject_a.subject_identifier} + {subject_b.subject_identifier}")
            log(f"  Data length: {self.data_length} samples")
            log(f"  Number of windows: {self.n_windows}")
            log(f"  Window size: {self.window_size} samples")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # Calculate start and end indices for this window
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        
        # Get data for both subjects at the same time point
        window_a = self.subject_a.get_all_electrode_data(
            self.session_id, start_idx, end_idx
        ).to(dtype=self.dtype)
        
        window_b = self.subject_b.get_all_electrode_data(
            self.session_id, start_idx, end_idx
        ).to(dtype=self.dtype)
        
        # Create output structure
        output = {
            'data': window_a,
            'data_b': window_b,
        }
        
        if self.output_metadata:
            output['metadata'] = {
                'subject_identifier': self.subject_a.subject_identifier,
                'trial_id': self.session_id,
                'sampling_rate': self.subject_a.get_sampling_rate(self.session_id),
                'time_point': start_idx / self.subject_a.get_sampling_rate(self.session_id),  # Time in seconds
            }
            output['metadata_b'] = {
                'subject_identifier': self.subject_b.subject_identifier,
                'trial_id': self.session_id,
                'sampling_rate': self.subject_b.get_sampling_rate(self.session_id),
                'time_point': start_idx / self.subject_b.get_sampling_rate(self.session_id),  # Time in seconds
            }
            output['subject_trial'] = (self.subject_a.subject_identifier, self.session_id)
            output['subject_trial_b'] = (self.subject_b.subject_identifier, self.session_id)
            
        if self.output_electrode_labels:
            output['electrode_labels'] = self.electrode_labels_a
            output['electrode_labels_b'] = self.electrode_labels_b
            
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
            output['electrode_labels'] = [batch[0]['electrode_labels']]
        if 'electrode_labels_b' in batch[0]:
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

class PodcastBatchPairSampler(torch.utils.data.Sampler):
    """
    Batch sampler for podcast paired data.
    Similar to SubjectBatchPairSampler but adapted for podcast data.
    """
    def __init__(self, dataset_sizes, batch_size, shuffle=True, drop_last=True):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        # Create batches for each subject pair
        all_batches = []
        start_idx = 0
        
        for size in self.dataset_sizes:
            # Create indices for this subject pair
            subject_indices = list(range(start_idx, start_idx + size))
            
            if self.shuffle:
                random.shuffle(subject_indices)
            
            # Create batches
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

def load_podcast_subjects(train_subject_pairs, eval_subject_pairs, dtype, cache=True, allow_corrupted=False):
    """
    Load podcast subjects for training and evaluation.
    
    Args:
        train_subject_pairs: List of (subject_a_id, subject_b_id) tuples for training
        eval_subject_pairs: List of (subject_a_id, subject_b_id) tuples for evaluation
        dtype: Data type for neural data
        cache: Whether to cache data
        allow_corrupted: Whether to allow corrupted electrodes
    
    Returns:
        dict: Dictionary mapping subject identifiers to PodcastSubject objects
    """
    from subject.podcast import PodcastSubject
    
    # Get all unique subject IDs
    all_subject_ids = set()
    for subject_a_id, subject_b_id in train_subject_pairs:
        all_subject_ids.add(subject_a_id)
        all_subject_ids.add(subject_b_id)
    for subject_a_id, subject_b_id in eval_subject_pairs:
        all_subject_ids.add(subject_a_id)
        all_subject_ids.add(subject_b_id)
    
    all_subjects = {}
    
    for subject_id in all_subject_ids:
        subject_identifier = f'podcast{subject_id:02d}'
        log(f"Loading podcast subject {subject_identifier}...", indent=1, priority=1)
        all_subjects[subject_identifier] = PodcastSubject(
            subject_id, dtype=dtype, cache=cache, allow_corrupted=allow_corrupted
        )
    
    return all_subjects

if __name__ == "__main__":
    # Test the paired podcast dataset
    try:
        from subject.podcast import PodcastSubject
        
        # Create two podcast subjects
        subject_a = PodcastSubject(1, cache=False)
        subject_b = PodcastSubject(2, cache=False)
        
        # Create paired dataset
        window_size = 1536  # 3 seconds at 512 Hz
        dataset = PodcastTrialPairDataset(
            subject_a, subject_b, window_size, 
            dtype=torch.float32, output_metadata=True, output_electrode_labels=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Sample data shape A: {dataset[0]['data'].shape}")
        print(f"Sample data shape B: {dataset[0]['data_b'].shape}")
        print(f"Subject A: {dataset[0]['subject_trial']}")
        print(f"Subject B: {dataset[0]['subject_trial_b']}")
        print(f"Time point: {dataset[0]['metadata']['time_point']:.2f}s")
        
    except Exception as e:
        print(f"Error testing podcast pair dataset: {e}") 