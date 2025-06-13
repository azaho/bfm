import time
import psutil
import torch
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class ParamConfig:
    value: Any
    type: type
    help: str
    include_in_dirname: bool = False
    dirname_format: Optional[callable] = None
    default: Any = None

CONFIG_SCHEMA = {
    'model': {
        'name': ParamConfig('OM', str, 'Model name'), # Right now "Original Model"
        
        'context_length': ParamConfig(3, float, 'Context length in seconds'),


        'max_n_electrodes': ParamConfig(128, int, 'Maximum number of electrodes to use', dirname_format=lambda x: f'nes{x}'),

        'electrode_embedding': {
            'type': ParamConfig('learned', str, 'Type of electrode embedding', dirname_format=lambda x: f'ee{x}'), # Allowed values: 'learned', 'zero', 'coordinate_init', 'noisy_coordinate'
            'dim': ParamConfig(None, int, 'Dimension of electrode embeddings', dirname_format=lambda x: f'ed{x}'),

            'coordinate_noise_std': ParamConfig(0.0, float, 'Coordinate noise std for electrode embedding'), # Only relevant for the 'noisy_coordinate' type
        },

        'signal_preprocessing': {
            'laplacian_rereference': ParamConfig(True, bool, 'Whether to Laplacian rereference the signal before feeding it to the model'),
            'normalize_voltage': ParamConfig(True, bool, 'Whether to normalize the voltage of the signal (batch norm) before feeding it to the model'),

            'spectrogram': ParamConfig(True, bool, 'Whether to use spectrogram'), # Whether to use spectrogram of the signal or take raw voltage as input
            'spectrogram_max_frequency': ParamConfig(200, int, 'Maximum frequency for spectrogram'),

            'time_bin_size': ParamConfig(0.125, float, 'Time bin size in seconds'), # Only relevant for spectrogram = 0, when we are binning raw voltage
        },

        'transformer': {
            'd_model': ParamConfig(192, int, 'Dimensionality of the latent space of the model', dirname_format=lambda x: f'dm{x}'),
            'n_heads': ParamConfig(12, int, 'Number of attention heads', dirname_format=lambda x: f'nh{x}'),
            'n_layers_electrode': ParamConfig(5, int, 'Number of transformer layers for electrode path', dirname_format=lambda x: f'nl{x}'),
            'n_layers_time': ParamConfig(5, int, 'Number of transformer layers for time path'),
        },

        'dtype': ParamConfig(torch.float32, torch.dtype, 'Model data type'), # Data type for the model weights and activations 
        'use_mixed_precision': ParamConfig(True, bool, 'Whether to use mixed precision training'), # Helps to conserve GPU memory while not really hurting performance
        'amp_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'AMP data type'), # Data type for the mixed precision training
    },

    'cluster': {
        'wandb_project': ParamConfig("", str, 'wandb.com project name'), # If empty, no wandb is used
        'save_model_every_n_epochs': ParamConfig(5, int, 'Save the model weights and training statistics every n epochs'),
        'eval_model_every_n_epochs': ParamConfig(5, int, 'Evaluate the model every n epochs'),

        'timestamp': ParamConfig(time.strftime("%Y%m%d_%H%M%S"), str, 'Timestamp'), # the time when the model was trained
        'cache_subjects': ParamConfig(True, bool, 'Whether to cache subjects'), # Whether to cache the subject datasets in RAM, or to load them from the disk as the training proceeds.

        'num_workers_dataloaders': ParamConfig(4, int, 'Number of processes for dataloaders'), # note that you will need to request enough CPU cores to cover all these workers
        'num_workers_eval': ParamConfig(4, int, 'Number of processes for evaluation'),        
        'prefetch_factor': ParamConfig(2, int, 'Prefetch factor'), # for the dataloader workers

        'quick_eval': ParamConfig(True, bool, 'Whether to do quick evaluation'), # Whether to do quick evaluation on a subset of the data
        'eval_aggregation_method': ParamConfig('concat', str, 'Evaluation aggregation method'),
    },

    'training': {
        'train_subject_trials': ParamConfig("btbank3_1", str, 'Train subject trials'), # a string like btbank3_1,btbank3_2,...
        'eval_subject_trials': ParamConfig("btbank3_0", str, 'Eval subject trials'), # a string like btbank3_0,btbank3_1,...
        'data_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'Data type for tensors'),
        
        'n_epochs': ParamConfig(100, int, 'Number of epochs to train'),
        'p_test': ParamConfig(0.1, float, 'Proportion of data to use as a test split (different from the eval)'),

        'optimizer': ParamConfig('Muon', str, 'Optimizer type'),
        'batch_size': ParamConfig(100, int, 'Batch size for training'),
        
        'learning_rate': ParamConfig(0.003, float, 'Learning rate'),
        'lr_schedule': ParamConfig('linear', str, 'Learning rate schedule (none, linear, cosine)'),
        'weight_decay': ParamConfig(0.0001, float, 'Weight decay for optimizer'),
        'dropout': ParamConfig(0.1, float, 'Dropout rate', include_in_dirname=True, dirname_format=lambda x: f'dr{x}'),
        
        'p_electrodes_per_stream': ParamConfig(0.5, float, 'Proportion of electrodes per stream'),
        'future_bin_idx': ParamConfig(1, int, 'Future bin index'),
        'warmup_steps': ParamConfig(100, int, 'Warmup steps'),
        
        'normalize_features': ParamConfig(True, bool, 'Whether to normalize features'),
        'use_temperature_param': ParamConfig(True, bool, 'Whether to use temperature parameter'),
        'max_temperature_param': ParamConfig(1000.0, float, 'Maximum temperature parameter value'),
        
        'random_string': ParamConfig("X", str, 'Random string for seed generation', include_in_dirname=True, dirname_format=lambda x: f'r{x}'),
    },
}

def get_default_config(random_string, wandb_project):
    # Convert schema to actual config
    def convert_schema_to_config(schema):
        config = {}
        for key, value in schema.items():
            if isinstance(value, ParamConfig):
                config[key] = value.value
            elif isinstance(value, dict):
                config[key] = convert_schema_to_config(value)
        return config

    config = convert_schema_to_config(CONFIG_SCHEMA)
    config['cluster']['wandb_project'] = wandb_project
    config['training']['random_string'] = random_string

    return config

def parse_config_from_args(config):
    parser = argparse.ArgumentParser()
    
    def add_args_from_schema(schema, prefix=''):
        for key, value in schema.items():
            if isinstance(value, ParamConfig):
                arg_name = f'--{prefix}{key}' if prefix else f'--{key}'
                parser.add_argument(arg_name, type=value.type, default=None, help=value.help)
            elif isinstance(value, dict):
                new_prefix = f'{prefix}{key}.' if prefix else f'{key}.'
                add_args_from_schema(value, new_prefix)

    add_args_from_schema(CONFIG_SCHEMA)
    args = parser.parse_args()

    def update_config_from_args(config, schema, args, prefix=''):
        for key, value in schema.items():
            if isinstance(value, ParamConfig):
                arg_name = f'{prefix}{key}' if prefix else key
                if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                    config[key] = getattr(args, arg_name)
            elif isinstance(value, dict):
                new_prefix = f'{prefix}{key}.' if prefix else f'{key}.'
                update_config_from_args(config[key], value, args, new_prefix)

    update_config_from_args(config, CONFIG_SCHEMA, args)

def parse_subject_trials_from_config(config):
    train_subject_trials = config['training']['train_subject_trials'] # a string like btbank1_1,btbank1_2,...
    eval_subject_trials = config['training']['eval_subject_trials']
    train_subject_trials = [[subject_identifier.split('_')[0], int(subject_identifier.split('_')[1])] for subject_identifier in train_subject_trials.split(',')]
    eval_subject_trials = [[subject_identifier.split('_')[0], int(subject_identifier.split('_')[1])] for subject_identifier in eval_subject_trials.split(',')]
    config['training']['train_subject_trials'] = train_subject_trials
    config['training']['eval_subject_trials'] = eval_subject_trials

def update_dir_name(config):
    dir_name = config['model']['name']
    def add_to_dirname(config, schema, prefix=''):
        nonlocal dir_name
        for key, value in schema.items():
            if isinstance(value, ParamConfig):
                if value.include_in_dirname:
                    config_value = config[key]
                    if value.dirname_format:
                        dir_name += '_' + value.dirname_format(config_value)
                    else:
                        dir_name += f"_{key}{config_value}"
            elif isinstance(value, dict):
                new_prefix = f'{prefix}{key}.' if prefix else f'{key}.'
                add_to_dirname(config[key], value, new_prefix)

    add_to_dirname(config, CONFIG_SCHEMA)
    
    config['cluster']['dir_name'] = dir_name
    return dir_name

max_log_priority = 1
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:.1f}G ram {ram_usage:.1f}G] {' '*4*indent}{message}")

def update_random_seed(config):
    random_seed = hash(config['training']['random_string']) % (2**32)
    config['training']['random_seed'] = random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    return random_seed

# Convert torch dtypes to strings before saving
def convert_dtypes(config):
    if isinstance(config, dict):
        return {k: convert_dtypes(v) for k, v in config.items()}
    elif isinstance(config, torch.dtype):
        return str(config)
    return config

# Convert string dtypes back to torch dtypes
def unconvert_dtypes(config):
    if isinstance(config, dict):
        return {k: unconvert_dtypes(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith('torch.'):
        return getattr(torch, config.split('.')[-1])
    return config