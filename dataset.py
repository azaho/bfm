import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from subject_braintreebank import BrainTreebankSubject
from subject_mgh import MGHSubject
from train_utils import log
from multiprocessing import Pool
import torch.multiprocessing as mp
import random

# Standardizing pretraining and evaluation subjects and trials
all_btbank_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
all_btbank_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in all_btbank_subject_trials]
eval_subject_trials = [(1, 2), (2, 6), (3, 0), (6, 4), (7, 0), (4, 1), (10, 0)]
eval_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in eval_subject_trials]
train_subject_trials = [st for st in all_btbank_subject_trials if st not in eval_subject_trials]

# Filter: only 1024Hz, only >20min recording session
all_mgh_subject_trials = [('mgh1', 0), ('mgh1', 1), ('mgh1', 2), ('mgh1', 3), ('mgh2', 0), ('mgh2', 1), ('mgh3', 0), ('mgh3', 1), ('mgh3', 2), ('mgh4', 0), ('mgh5', 0), ('mgh6', 0), ('mgh6', 1), ('mgh7', 0), ('mgh7', 1), ('mgh7', 2), ('mgh8', 0), ('mgh9', 0), ('mgh9', 1), ('mgh9', 2), ('mgh10', 0), ('mgh10', 1), ('mgh10', 2), ('mgh10', 3), ('mgh10', 4), ('mgh11', 0), ('mgh11', 1), ('mgh12', 0), ('mgh12', 1), ('mgh13', 0), ('mgh14', 0), ('mgh14', 1), ('mgh15', 0), ('mgh16', 0), ('mgh16', 1), ('mgh16', 2), ('mgh16', 3), ('mgh16', 4), ('mgh16', 5), ('mgh16', 6), ('mgh17', 0), ('mgh17', 1), ('mgh18', 0), ('mgh18', 1), ('mgh19', 0), ('mgh19', 1), ('mgh20', 0), ('mgh21', 0), ('mgh22', 0), ('mgh23', 0), ('mgh23', 1), ('mgh24', 0), ('mgh25', 0), ('mgh25', 1), ('mgh26', 0), ('mgh27', 0), ('mgh27', 1), ('mgh28', 0), ('mgh28', 1), ('mgh29', 0), ('mgh29', 1), ('mgh30', 0), ('mgh30', 1), ('mgh31', 0), ('mgh32', 0), ('mgh32', 1), ('mgh33', 0), ('mgh34', 0), ('mgh34', 1), ('mgh34', 2), ('mgh34', 3), ('mgh34', 4), ('mgh34', 5), ('mgh34', 6), ('mgh35', 0), ('mgh36', 0), ('mgh36', 1), ('mgh36', 2), ('mgh36', 3), ('mgh39', 0), ('mgh40', 0), ('mgh40', 1), ('mgh40', 2), ('mgh40', 3), ('mgh40', 4), ('mgh40', 5), ('mgh40', 6), ('mgh41', 0), ('mgh42', 0), ('mgh43', 0), ('mgh43', 1), ('mgh43', 2), ('mgh44', 0), ('mgh45', 0), ('mgh46', 0), ('mgh47', 0), ('mgh48', 0), ('mgh49', 0), ('mgh50', 0), ('mgh50', 1), ('mgh51', 0), ('mgh52', 0), ('mgh53', 0), ('mgh54', 0), ('mgh54', 1), ('mgh55', 0), ('mgh56', 0), ('mgh57', 0), ('mgh57', 1), ('mgh59', 0), ('mgh60', 0), ('mgh60', 1), ('mgh61', 0), ('mgh62', 0), ('mgh62', 1), ('mgh62', 2)]

class SubjectTrialDataset(Dataset):
    def __init__(self, subject, trial_id, window_size, dtype=torch.float32, output_subject_trial_id=False):
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
        self.output_subject_trial_id = output_subject_trial_id

        subject.load_neural_data(trial_id)
        self.n_windows = self.subject.electrode_data_length[trial_id] // self.window_size
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        window = self.subject.get_all_electrode_data(self.trial_id, start_idx, end_idx).to(dtype=self.dtype)
        if self.output_subject_trial_id: 
            return window, (self.subject.subject_identifier, self.trial_id)
        else: return window

def load_dataloaders(train_subject_trials, eval_subject_trials, p_test, sample_timebin_size, max_n_timebins, dtype, batch_size, num_workers_dataloaders=12, prefetch_factor=2, cache=True, allow_corrupted=False, test_num_workers_fraction=0.15):
    # Step 1: Load all subjects
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
            all_subjects[subject_identifier] = MGHSubject(subject_id, dtype=dtype, cache=cache, allow_corrupted=allow_corrupted)
        else:
            raise ValueError(f"Unknown subject identifier: {subject_identifier}")

    # Step 2: Load all datasets 
    datasets = []
    for subject_identifier, trial_id in train_subject_trials:
        log(f"loading dataset for {subject_identifier}_{trial_id}...", indent=1, priority=1)
        datasets.append(
            SubjectTrialDataset(
                all_subjects[subject_identifier], 
                trial_id, 
                int(sample_timebin_size * all_subjects[subject_identifier].get_sampling_rate(trial_id) * max_n_timebins), 
                dtype=dtype, 
                output_subject_trial_id=True
            )
        )
        log(f"finished loading dataset for {subject_identifier}_{trial_id}", indent=1, priority=1)

    # Step 3: Split into train and test
    train_datasets = []
    test_datasets = []
    for dataset in datasets:
        train_size = int(len(dataset) * (1 - p_test))
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    # Step 4: Create dataloaders with custom sampler
    class SubjectBatchSampler(torch.utils.data.Sampler):
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
    num_workers_dataloader_test = max(int(num_workers_dataloaders * test_num_workers_fraction), 1)
    num_workers_dataloader_train = num_workers_dataloaders - num_workers_dataloader_test
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=SubjectBatchSampler(
            [len(ds) for ds in train_datasets],
            batch_size=batch_size,
            shuffle=True
        ),
        num_workers=num_workers_dataloader_train,
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=True,  # Keep worker processes alive between iterations
        prefetch_factor=prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=SubjectBatchSampler(
            [len(ds) for ds in test_datasets],
            batch_size=batch_size,
            shuffle=False
        ),
        num_workers=num_workers_dataloader_test,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    return all_subjects, train_dataloader, test_dataloader


if __name__ == "__main__":
    subject = BrainTreebankSubject(3, cache=False)
    dataset = SubjectTrialDataset(subject, 0, 100, torch.float32)
    print(len(dataset))
    print(dataset[0].shape)
