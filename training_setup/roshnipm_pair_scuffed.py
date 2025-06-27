from model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from training_setup.training_config import log
import torch
from subject.dataset_pair import SubjectTrialPairDataset, PreprocessCollatorPair, SubjectBatchPairSampler
from torch.utils.data import DataLoader, ConcatDataset
import os
import json
from training_setup.training_config import convert_dtypes
import numpy as np
from evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE, BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING
import pandas as pd
from training_setup.training_setup import TrainingSetup

# for main function
from subject.dataset_pair import load_subjects
from evaluation.neuroprobe.config import NEUROPROBE_FULL_SUBJECT_TRIALS

# TODO: Checking for whether to run eval or not based on presence of 'metadata_a' vs. 'metadata'
# these lines in pretrain.py need to be handled:
# model_preprocess_functions=training_setup.get_preprocess_functions(pretraining=False),
# model_evaluation_function=training_setup.generate_frozen_features,


# Note: the functions in this file that have a "NotImplementedError" indicate that
#   this file is just an interface for a training setup. You will create a new file 
#   for your training setup, make a class inherit from TrainingSetup and implement the functions in that file.

class roshnipm_pair_scuffed(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        self.config = config
        self.all_subjects = all_subjects
        self.verbose = verbose

        self.model_components = {}

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model, "electrode_embeddings": electrode_embeddings}, where model and electrode_embeddings are PyTorch modules (those classes must inherit from model.BFModule)
        """
        raise NotImplementedError("This function is not (yet) implemented for this training setup.")
    
    def calculate_pretrain_loss(self, batch, output_accuracy=True):
        """
        Calculate the pretraining loss for a batch of data.

        Args:
            batch (dict): Dictionary containing:
                data (dict): Contains 'a' and 'b' keys, each with shape (batch_size, n_electrodes, n_timesamples)
                electrode_labels (dict): Contains 'a' and 'b' keys, each with electrode labels
                metadata (dict): Contains 'a' and 'b' keys, each with subject identifier, trial id, sampling rate, etc.
            output_accuracy (bool): Whether to output accuracy metrics

        Returns:
            dict: Dictionary of losses where keys are loss names and values are loss values.
                 The final loss is the mean of all losses. Accuracies are exempt and used only for logging.
        """
        raise NotImplementedError("This function is not (yet) implemented for this training setup.")

    def generate_frozen_features(self, batch):
        """
        Generate frozen features (meaning, the weights of the model are frozen) for a batch of data. This function is used for the model evaluation on the benchmarks.

        Args:
            batch (dict): Dictionary containing:
                data (dict): Contains 'a' and 'b' keys, each with shape (batch_size, n_electrodes, n_timebins)
                electrode_labels (dict): Contains 'a' and 'b' keys, each with electrode labels
                metadata (dict): Contains 'a' and 'b' keys, each with subject identifier, trial id, sampling rate, etc.

        Returns:
            torch.Tensor: Features with shape (batch_size, feature_vector_length) where feature_vector_length can be arbitrary
        """
        raise NotImplementedError("This function is not (yet) implemented for this training setup.")

    def _preprocess_laplacian_rereference(self, batch):
        batch_a = {'data': batch['data']}
        laplacian_rereference_batch(batch_a, remove_non_laplacian=False, inplace=True)
        batch['data'] = batch_a['data']

        if 'data_b' in batch:
            batch_b = {'data': batch['data_b']}
            laplacian_rereference_batch(batch_b, remove_non_laplacian=False, inplace=True)
            batch['data_b'] = batch_b['data']
        
        return batch
    
    def _preprocess_normalize_voltage(self, batch):
        batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
        batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)
        
        if 'data_b' in batch:
            batch['data_b'] = batch['data_b'] - torch.mean(batch['data_b'], dim=[0, 2], keepdim=True)
            batch['data_b'] = batch['data_b'] / (torch.std(batch['data_b'], dim=[0, 2], keepdim=True) + 1)
        
        return batch
    
    # TODO: this part should be changed because now we're comparing brains, not electrodes -- ask Andrii
    # For paired datasets, we might want to ensure both subjects have the same electrode subset
    # or handle electrode alignment differently since we're comparing brain responses
    def _preprocess_subset_electrodes(self, batch, output_selected_idx=False):
        # Find minimum number of electrodes in each batch for both 'a' and 'b'
        batch_size_a = batch['data'].shape[0]
        batch_size_b = batch['data_b'].shape[0]
        n_electrodes_a = batch['data'].shape[1]
        n_electrodes_b = batch['data_b'].shape[1]
        
        subset_n_electrodes_a = min(n_electrodes_a, self.config['training']['max_n_electrodes']) if self.config['training']['max_n_electrodes']>0 else n_electrodes_a
        subset_n_electrodes_b = min(n_electrodes_b, self.config['training']['max_n_electrodes']) if self.config['training']['max_n_electrodes']>0 else n_electrodes_b

        # Randomly subselect / permute electrodes for both 'a' and 'b'
        selected_idx_a = torch.randperm(n_electrodes_a)[:subset_n_electrodes_a]
        selected_idx_b = torch.randperm(n_electrodes_b)[:subset_n_electrodes_b]
        
        batch['data'] = batch['data'][:, selected_idx_a]
        batch['data_b'] = batch['data_b'][:, selected_idx_b]
        
        if 'electrode_labels' in batch:
            batch['electrode_labels'] = [batch['electrode_labels'][i] for i in selected_idx_a]
        if 'electrode_labels_b' in batch:
            batch['electrode_labels_b'] = [batch['electrode_labels_b'][i] for i in selected_idx_b]

        # returns a tuple
        if output_selected_idx:
            return batch, (selected_idx_a, selected_idx_b)
        else:
            return batch

    def load_dataloaders(self):
        """Load dataloaders for training and test sets."""
        config = self.config

        # Step 1: Create paired subject/trial combinations
        # Group subjects by movie to create pairs
        movie_to_subject_trials = {}
        for subject_identifier, trial_id in config['training']['train_subject_trials']:
            movie_key = f"{subject_identifier}_{trial_id}"
            if movie_key in BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING:
                movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[movie_key]
                # if movie_name not already in movie_to_subject_trials, create an empty list
                if movie_name not in movie_to_subject_trials:
                    movie_to_subject_trials[movie_name] = []
                # add the subject_identifier and trial_id to the list
                movie_to_subject_trials[movie_name].append((subject_identifier, trial_id))
        
        # Create pairs of subjects watching the same movie
        paired_datasets = []
        for movie_name, subject_trials in movie_to_subject_trials.items():
            # if there are at least 2 subjects watching the same movie, create pairs
            if len(subject_trials) >= 2:
                # create pairs from all combinations
                for i in range(len(subject_trials)):
                    for j in range(i + 1, len(subject_trials)):
                        subject_a_id, trial_a_id = subject_trials[i]
                        subject_b_id, trial_b_id = subject_trials[j]
                        
                        if self.verbose: 
                            log(f"Creating paired dataset: {subject_a_id}_{trial_a_id} + {subject_b_id}_{trial_b_id} (movie: {movie_name})", indent=1, priority=1)
                        
                        # Calculate window size in samples
                        window_size = int(config['model']['context_length'] * SAMPLING_RATE)
                        
                        # Calculate actual movie duration from trigger times file
                        # Use the shorter of the two trials to ensure both have data
                        subject_a = self.all_subjects[subject_a_id]
                        subject_b = self.all_subjects[subject_b_id]
                        
                        # Load neural data for both subjects first
                        subject_a.load_neural_data(trial_a_id)
                        subject_b.load_neural_data(trial_b_id)
                        
                        # Get movie duration from trigger times
                        trigger_times_file_a = os.path.join(ROOT_DIR, "subject_timings", f'sub_{int(subject_a_id.replace("btbank", ""))}_trial{int(trial_a_id):03}_timings.csv')
                        trigger_times_file_b = os.path.join(ROOT_DIR, "subject_timings", f'sub_{int(subject_b_id.replace("btbank", ""))}_trial{int(trial_b_id):03}_timings.csv')
                        
                        trigs_df_a = pd.read_csv(trigger_times_file_a)
                        trigs_df_b = pd.read_csv(trigger_times_file_b)
                        
                        # Get the end time from the last row (should be the 'end' type row)
                        movie_duration_a = trigs_df_a[trigs_df_a['type'] == 'end']['movie_time'].iloc[-1] if 'end' in trigs_df_a['type'].values else trigs_df_a['movie_time'].max()
                        movie_duration_b = trigs_df_b[trigs_df_b['type'] == 'end']['movie_time'].iloc[-1] if 'end' in trigs_df_b['type'].values else trigs_df_b['movie_time'].max()
                        
                        # Debug: Print movie durations
                        if self.verbose:
                            log(f"  Movie durations: {subject_a_id}_{trial_a_id}={movie_duration_a:.2f}s, {subject_b_id}_{trial_b_id}={movie_duration_b:.2f}s", indent=2, priority=1)
                        
                        # Use the shorter duration to ensure both subjects have data
                        total_time_seconds = min(movie_duration_a, movie_duration_b)
                        
                        window_time_seconds = window_size / SAMPLING_RATE
                        n_windows = int(total_time_seconds / window_time_seconds)
                        
                        # Create movie times for consecutive windows
                        movie_times = np.linspace(0, total_time_seconds, n_windows)
                        
                        # Create the paired dataset
                        dataset = SubjectTrialPairDataset(
                            subject_a, trial_a_id, window_size,
                            dtype=config['training']['data_dtype'],
                            output_metadata=True,
                            output_electrode_labels=True,
                            subject_b=subject_b, 
                            trial_id_b=trial_b_id,
                            movie_times=movie_times,
                            trigger_times_dir=os.path.join(ROOT_DIR, "subject_timings"),
                            sampling_rate=SAMPLING_RATE
                        )
                        
                        paired_datasets.append(dataset)
                        if self.verbose: 
                            log(f"Finished creating paired dataset: {len(dataset)} windows", indent=1, priority=1)

        if not paired_datasets:
            raise ValueError("No valid paired datasets found. Make sure subjects in train_subject_trials watch the same movies.")

        # Step 2: Split into train and test
        train_datasets = []
        test_datasets = []
        for dataset in paired_datasets:
            train_size = int(len(dataset) * (1 - config['training']['p_test']))
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

        # Step 3: Create dataloaders with custom sampler
        num_workers_dataloader_test = max(int(config['cluster']['num_workers_dataloaders'] * 0.15), 1)
        num_workers_dataloader_train = config['cluster']['num_workers_dataloaders'] - num_workers_dataloader_test
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=SubjectBatchPairSampler(
                [len(ds) for ds in train_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=True
            ),
            num_workers=num_workers_dataloader_train,
            pin_memory=True,  # pin memory for faster GPU transfer
            persistent_workers=True,  # keep worker processes alive between iterations
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollatorPair(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=SubjectBatchPairSampler(
                [len(ds) for ds in test_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=False
            ),
            num_workers=num_workers_dataloader_test,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollatorPair(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

def __main__():
    """Test the paired training setup."""
    
    test_config = {
        'training': {
            # initially tested with lite, then full
            'train_subject_trials': [(f"btbank{subject_id}", trial_id) for subject_id, trial_id in NEUROPROBE_FULL_SUBJECT_TRIALS],  # Convert integer subject IDs to string format that load_subjects expects
            'p_test': 0.2,
            # TODO: ask about batch size and movie duration
            # if smallest movie duration is actually 8.3 s, then maximum batch size is 83
            'batch_size': 4,
            'data_dtype': torch.float32,
            'max_n_electrodes': 50  # limit electrodes for testing
        },
        'model': {
            'context_length': 0.1,  # 100ms windows, results in n_timebins = 204
            'dtype': torch.float32
        },
        'cluster': {
            'num_workers_dataloaders': 2,
            'prefetch_factor': 2,
            'dir_name': 'test_paired_setup'
        }
    }
    
    print(f"Test config: {len(test_config['training']['train_subject_trials'])} subject trials")
    
    # load subjects
    all_subjects = load_subjects(
        test_config['training']['train_subject_trials'], 
        [], # no eval trials for testing
        test_config['training']['data_dtype'],
        cache=False,
        allow_corrupted=False
    )
    print(f"Loaded {len(all_subjects)} subjects")
    
    # creating training setup
    training_setup = TrainingSetupNewDataloader(all_subjects, test_config, verbose=True)
    print("Training setup created")
    
    # loading dataloaders
    training_setup.load_dataloaders()
    print("Dataloaders created")
    
    # testing a few batches
    batch_count = 0
    for batch in training_setup.train_dataloader:
        print(f"Batch {batch_count + 1}:")
        print(f"  Data shapes: A={batch['data']['a'].shape}, B={batch['data']['b'].shape}")
        print(f"  Subject trials A: {batch['subject_trial']['a']}")  
        print(f"  Subject trials B: {batch['subject_trial']['b']}")
        
        if 'metadata' in batch:
            print(f"  Metadata A: {batch['metadata']['a']['subject_identifier']}_{batch['metadata']['a']['trial_id']}")
            print(f"  Metadata B: {batch['metadata']['b']['subject_identifier']}_{batch['metadata']['b']['trial_id']}")
        
        if 'electrode_labels' in batch:
            print(f"  Electrode labels A: {len(batch['electrode_labels']['a'])} electrodes")
            print(f"  Electrode labels B: {len(batch['electrode_labels']['b'])} electrodes")
        
        batch_count += 1
        if batch_count >= 3:  # only test first 3 batches
            break
    
    print(f"Tested {batch_count} batches")
    
    # Test test dataloader
    test_batch = next(iter(training_setup.test_dataloader))
    print(f"Test batch data shapes: A={test_batch['data']['a'].shape}, B={test_batch['data']['b'].shape}")

if __name__ == "__main__":
    __main__()