'''
Defines the TrainingSetup class, which is an interface for training setups.

Functions that raise NotImplementedError are meant to be overridden.
To create a custom training setup, define a new class in a separate file that 
inherits from `TrainingSetup` and implements the required methods.
'''
import os
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset

from model.electrode_embedding import (
    ElectrodeEmbedding_Learned,
    ElectrodeEmbedding_NoisyCoordinate,
    ElectrodeEmbedding_Learned_CoordinateInit,
    ElectrodeEmbedding_Zero,
)
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch

from subject.dataset import SubjectTrialDataset, PreprocessCollator, SubjectBatchSampler

from training_setup.training_config import (
    log,
    convert_dtypes,
)

RUN_DIR = 'runs/data'
class TrainingSetup:
    '''
    Interface for training setups.
    
    Must implement the following methods:
        - initialize_model: Initializes the model components and sets self.model_components.
        - calculate_pretrain_loss: Calculates the pretraining loss for a batch of data.
        - generate_frozen_features: Generates frozen features for a batch of data.
    '''
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
                data (torch.Tensor): Shape (batch_size, n_electrodes, n_timesamples)
                electrode_index (torch.Tensor): Shape (batch_size, n_electrodes)
                metadata (dict): Contains subject identifier, trial id, sampling rate, etc.
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
                data (torch.Tensor): Shape (batch_size, n_electrodes, n_timesamples)
                electrode_labels (list): List of length 1 (since it's the same across the batch), each element is a list of electrode labels
                metadata (dict): Contains subject identifier, trial id, sampling rate, etc.

        Returns:
            features (torch.Tensor): Shape (batch_size, n_electrodes or n_electrodes+1, n_timebins, *) where * can be arbitrary
                                   if n_electrodes+1, then the first dimension is the cls token
        """
        raise NotImplementedError("This function is not (yet) implemented for this training setup.")

    def _preprocess_laplacian_rereference(self, batch):
        laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)
        return batch
    def _preprocess_normalize_voltage(self, batch):
        batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
        batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)
        return batch
    
    def _preprocess_subset_electrodes(self, batch, output_selected_idx=False):
        # Find minimum number of electrodes in batch
        batch_size = batch['data'].shape[0]
        n_electrodes = batch['data'].shape[1]
        subset_n_electrodes = min(n_electrodes, self.config['training']['max_n_electrodes']) if self.config['training']['max_n_electrodes']>0 else n_electrodes

        # Randomly subselect / permute electrodes
        selected_idx = torch.randperm(n_electrodes)[:subset_n_electrodes]
        batch['data'] = batch['data'][:, selected_idx]
        if 'electrode_labels' in batch:
            batch['electrode_labels'] = [[batch['electrode_labels'][0][i] for i in selected_idx]] * batch_size

        if output_selected_idx:
            return batch, selected_idx
        else:
            return batch

    def get_preprocess_functions(self, pretraining=False):
        """
        Get the preprocess functions for the training setup.
        Default: only subset electrodes to the maximum number during pretraining (during eval, pass all electrodes).

        This must be an array of functions, each takes just batch as input and returns the (modified) batch. Modifying it in place is fine (but still return the batch).
        """
        return [self._preprocess_subset_electrodes] if pretraining else []

    def generate_state_dicts(self):
        """
            This function generates the state dicts for the model components. It is used for saving and retrieving the model.
        """
        return {"state_dicts": 
            {
                model_component_name: model_component.state_dict() 
                for model_component_name, model_component in self.model_components.items()
            }
        }

    def save_model(self, epoch, eval_results={}, save_in_dir=RUN_DIR, training_statistics_store=None):
        path = f"{save_in_dir}/{self.config['cluster']['dir_name']}"
        model_path = f"{path}/model_epoch_{epoch}.pth"
        os.makedirs(path, exist_ok=True)
        torch.save({
            'eval_results': eval_results,
            'epoch': epoch,
            'state_dicts': self.generate_state_dicts()['state_dicts'],
            'config': convert_dtypes(self.config),
        }, model_path)
        if training_statistics_store is not None:
            with open(f"{path}/training_statistics.json", 'w') as f:
                json.dump(training_statistics_store, f)

    def load_model(self, epoch, load_from_dir=RUN_DIR):
        model_path = f"{load_from_dir}/{self.config['cluster']['dir_name']}/model_epoch_{epoch}.pth"
        state_dicts = torch.load(model_path, map_location=self.config['device'])
        for model_component_name, model_component in self.model_components.items():
            if "state_dicts" in state_dicts:
                model_component.load_state_dict(state_dicts["state_dicts"][model_component_name])
            else:
                model_component.load_state_dict(state_dicts[model_component_name+"_state_dict"]) # XXX: this is only here for backwards compatibility, can remove soon
            model_component.to(self.config['device'], dtype=self.config['model']['dtype'])
    
    def model_parameters(self, verbose=False):
        """
            This function returns all of the parameters in the model, and stores the number of parameters in the config.
            It must output a list of all parameters in the model.
        """
        if 'n_params' not in self.config['model']: self.config['model']['n_params'] = {}
        all_params = []
        for model_component_name, model_component in self.model_components.items():
            all_params.extend(list(model_component.parameters()))
            if verbose: log(f"{model_component_name} parameters: {sum(p.numel() for p in model_component.parameters()):,}", priority=0)
            self.config['model']['n_params'][model_component_name] = sum(p.numel() for p in model_component.parameters())
        total_n_params = sum(p.numel() for p in all_params)
        log(f"Total model parameters: {total_n_params:,}", priority=0)
        self.config['model']['n_params']['total'] = total_n_params
        return all_params
    
    def train_mode(self):
        for model_component in self.model_components.values():
            model_component.train()
    
    def eval_mode(self):
        for model_component in self.model_components.values():
            model_component.eval()

    def calculate_pretrain_test_loss(self):
        """
        Calculate the pretraining test loss. This function uses the calculate_pretrain_loss function.

        Returns:
            dict: Dictionary of losses where keys are loss names and values are loss values.
                 The final loss is the mean of all losses. Accuracies are exempt and used only for logging.
        """
        losses = {}
        n_batches = 0
        for batch in self.test_dataloader:
            loss = self.calculate_pretrain_loss(batch)
            
            for key, value in loss.items():
                if key not in losses: losses[key] = 0
                losses[key] += value
            n_batches += 1
        
        # Handle case where test dataloader is empty
        if n_batches == 0:
            return {}
        
        return {k: v / n_batches for k, v in losses.items()}

    def load_dataloaders(self):
        """
            This function loads the dataloaders for the training and test sets.

            It must set the self.train_dataloader and self.test_dataloader attributes to the dataloaders (they are used in the pretraining code in pretrain.py)
        """
        config = self.config

        # Step 1: Load datasets
        datasets = []
        for subject_identifier, trial_id in config['training']['train_subject_trials']:
            if self.verbose: log(f"loading dataset for {subject_identifier}_{trial_id}...", indent=1, priority=1)
            datasets.append(
                SubjectTrialDataset(
                    self.all_subjects[subject_identifier], 
                    trial_id, 
                    int(config['model']['context_length'] * self.all_subjects[subject_identifier].get_sampling_rate(trial_id)), 
                    dtype=config['training']['data_dtype'], 
                    output_metadata=True,
                    output_electrode_labels=True
                )
            )
            if self.verbose: log(f"finished loading dataset for {subject_identifier}_{trial_id}", indent=1, priority=1)

        # Step 2: Split into train and test
        train_datasets = []
        test_datasets = []
        for dataset in datasets:
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
            batch_sampler=SubjectBatchSampler(
                # length of each dataset = number of windows in the subject
                [len(ds) for ds in train_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=True
            ),
            num_workers=num_workers_dataloader_train,
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True,  # Keep worker processes alive between iterations
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollator(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=SubjectBatchSampler(
                # length of each dataset = number of windows in the subject
                [len(ds) for ds in test_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=False
            ),
            num_workers=num_workers_dataloader_test,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollator(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader