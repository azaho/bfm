import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from subject.braintreebank import BrainTreebankSubject
from subject.mgh2024 import MGH2024Subject
from training_setup.training_config import log
from multiprocessing import Pool
import torch.multiprocessing as mp
import random

class SubjectTrialPairDataset(Dataset):
    def __init__(self, subject, trial_id, window_size, dtype=torch.float32, output_metadata=False, output_electrode_labels=False):
        """
        Args:
            subject (BrainTreebankSubject or MGHSubject): Subject object
            trial_id (int): Trial ID
            dtype (torch.dtype): Data type to load the data in (float32, bfloat16)
            window_size (int): Number of time samples per data item
        """
        self.subject = subject
        self.trial_id = trial_id
        self.window_size = window_size
        self.dtype = dtype
        self.output_metadata = output_metadata
        self.output_electrode_labels = output_electrode_labels

        subject.load_neural_data(trial_id)
        self.n_windows = self.subject.electrode_data_length[trial_id] // self.window_size
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        window = self.subject.get_all_electrode_data(self.trial_id, start_idx, end_idx).to(dtype=self.dtype)

        output = {'data': window}
        if self.output_metadata: 
            output['subject_trial'] = (self.subject.subject_identifier, self.trial_id)
        if self.output_electrode_labels:
            output['electrode_labels'] = self.subject.electrode_labels # Also output the electrode label
        
        # options:
        # 1. have array w/in each metadata item
        # 2. store metadata_a and metadata_b as separate fields
        output['metadata'] = {
            'subject_identifier': self.subject.subject_identifier,
            'trial_id': self.trial_id,
            'sampling_rate': self.subject.get_sampling_rate(self.trial_id),
        }
        
        return output

class PreprocessCollatorPair:
    def __init__(self, preprocess_functions=[]):
        self.preprocess_functions = preprocess_functions

    def __call__(self, batch):
        # batch is now a list of dictionaries

        # Process each item in batch
        output = {
            'data': torch.stack([item['data'] for item in batch]),
        }
        # TODO: fix based on the index (can't just be 0)
        if 'electrode_labels' in batch[0]:
            output['electrode_labels'] = [item['electrode_labels'] for item in batch]
        if 'metadata' in batch[0]:
            output['metadata'] = batch[0]['metadata']

        # If any preprocess functions are provided, apply them to the batch
        for preprocess_function in self.preprocess_functions:
            output = preprocess_function(output)
            
        # Copy through any other fields that don't need processing
        # TODO: 0-based indexing here as well
        for key in batch[0].keys():
            if key not in output and key != 'data':
                output[key] = [item[key] for item in batch]
                if isinstance(batch[0][key], torch.Tensor): # If the field is a tensor, stack it
                    output[key] = torch.stack(output[key])
        
        return output

# based on random subject/trial, make a batch
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
            
            for size in self.dataset_sizes:
                # Create indices for this subject
                subject_indices = list(range(start_idx, start_idx + size))
                if self.shuffle:
                    random.shuffle(subject_indices)
                
                # Create batches for all subjects
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
    subject = BrainTreebankSubject(3, cache=False)
    dataset = SubjectTrialPairDataset(subject, 0, 100, torch.float32)
    print("Length of dataset:", len(dataset))
    print("Shape of dataset[0]:", dataset[0]['data'].shape)
