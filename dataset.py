import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from subject.braintreebank import BrainTreebankSubject
from subject.mgh2024 import MGH2024Subject
from utils.training_config import log
from multiprocessing import Pool
import torch.multiprocessing as mp
import random

class SubjectTrialDataset(Dataset):
    def __init__(self, subject, trial_id, window_size, dtype=torch.float32, output_embeddings_map=None, output_subject_trial_id=False, output_electrode_labels=False):
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
        self.output_embeddings_map = output_embeddings_map
        self.output_subject_trial_id = output_subject_trial_id
        self.output_electrode_labels = output_electrode_labels

        self.electrode_keys = [(subject.subject_identifier, electrode_label) for electrode_label in self.subject.electrode_labels]
        if output_embeddings_map is not None:                
            self.electrode_indices = torch.tensor([self.output_embeddings_map[key] for key in self.electrode_keys])

        subject.load_neural_data(trial_id)
        self.n_windows = self.subject.electrode_data_length[trial_id] // self.window_size
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        window = self.subject.get_all_electrode_data(self.trial_id, start_idx, end_idx).to(dtype=self.dtype)

        output = {'data': window}
        if self.output_subject_trial_id: 
            output['subject_trial'] = (self.subject.subject_identifier, self.trial_id)
        if self.output_electrode_labels:
            output['electrode_labels'] = self.subject.electrode_labels # Also output the electrode label
        if self.output_embeddings_map: 
            output['electrode_index'] = self.electrode_indices

        output['metadata'] = {
            'subject_identifier': self.subject.subject_identifier,
            'trial_id': self.trial_id,
            'sampling_rate': self.subject.get_sampling_rate(self.trial_id),
        }
        
        return output

class RandomElectrodeCollator:
    def __init__(self, max_n_electrodes=None):
        self.max_n_electrodes = max_n_electrodes

    def __call__(self, batch):
        # batch is now a list of dictionaries
        # Extract signals and any additional data
        data = [item['data'] for item in batch]
        
        # Find minimum number of electrodes in batch
        min_electrodes = min(item['data'].shape[0] for item in batch)
        max_electrodes = max(item['data'].shape[0] for item in batch)
        n_electrodes = min(min_electrodes, self.max_n_electrodes) if self.max_n_electrodes else min_electrodes

        selected_idx = torch.randperm(min_electrodes)[:n_electrodes]

        # Process each item in batch
        output = {}
        
        # Process signals
        processed_data = []
        processed_electrode_indices = []
        processed_electrode_labels = []
        for item in batch:
            data = item['data']
            if min_electrodes != max_electrodes: # Only if all electrodes are not the same length (assuming they come from different places) # XXX Surely there must be a better way to do this, for example look at variance of electrode_indices
                selected_idx =  torch.randperm(data.shape[0])[:n_electrodes]
            if 'electrode_index' in item:
                processed_electrode_indices.append(item['electrode_index'][selected_idx])
            if 'electrode_labels' in item:
                processed_electrode_labels.append([item['electrode_labels'][i] for i in selected_idx])
            processed_data.append(data[selected_idx])
        output['data'] = torch.stack(processed_data)
        if len(processed_electrode_indices) > 0:
            output['electrode_index'] = torch.stack(processed_electrode_indices)
        if len(processed_electrode_labels) > 0:
            output['electrode_labels'] = processed_electrode_labels
        
        if 'metadata' in batch[0]:
            output['metadata'] = batch[0]['metadata'] # assume all metadata is the same for all items in the batch
        
        # Copy through any other fields that don't need processing
        for key in batch[0].keys():
            if key not in output and key != 'data':
                output[key] = [item[key] for item in batch]
        
        return output

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

def load_dataloaders(all_subjects, train_subject_trials, p_test, context_length, dtype, 
                     batch_size, max_n_electrodes=None, output_embeddings_map=None, 
                     num_workers_dataloaders=12, prefetch_factor=2, test_num_workers_fraction=0.15):
    # Step 2: Load all datasets 
    datasets = []
    for subject_identifier, trial_id in train_subject_trials:
        log(f"loading dataset for {subject_identifier}_{trial_id}...", indent=1, priority=1)
        datasets.append(
            SubjectTrialDataset(
                all_subjects[subject_identifier], 
                trial_id, 
                int(context_length * all_subjects[subject_identifier].get_sampling_rate(trial_id)), 
                dtype=dtype, 
                output_embeddings_map=output_embeddings_map,
                output_subject_trial_id=True,
                output_electrode_labels=True
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
        collate_fn=RandomElectrodeCollator(max_n_electrodes)
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
        collate_fn=RandomElectrodeCollator(max_n_electrodes)
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    subject = BrainTreebankSubject(3, cache=False)
    dataset = SubjectTrialDataset(subject, 0, 100, torch.float32)
    print("Length of dataset:", len(dataset))
    print("Shape of dataset[0]:", dataset[0]['data'].shape)
