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
    required: bool = False

CONFIG_SCHEMA = {
    'model': {
        'name': ParamConfig('OM', str, 'Model name'), # Right now "Original Model"
        
        'context_length': ParamConfig(3, float, 'Context length in seconds'),

        'electrode_embedding': {
            'type': ParamConfig('learned', str, 'Type of electrode embedding', dirname_format=lambda x: f'ee{x}'), # Allowed values: 'learned', 'zero', 'coordinate_init', 'noisy_coordinate'
            'dim': ParamConfig(None, int, 'Dimension of electrode embeddings', dirname_format=lambda x: f'ed{x}'),

            'coordinate_noise_std': ParamConfig(0.0, float, 'Coordinate noise std for electrode embedding'), # Only relevant for the 'noisy_coordinate' type
        },

        'signal_preprocessing': {
            'laplacian_rereference': ParamConfig(True, bool, 'Whether to Laplacian rereference the signal before feeding it to the model'),
            'normalize_voltage': ParamConfig(True, bool, 'Whether to normalize the voltage of the signal (batch norm) before feeding it to the model'),

            'spectrogram': ParamConfig(True, bool, 'Whether to use spectrogram'), # Whether to use spectrogram of the signal or take raw voltage as input
            'spectrogram_parameters': {
                'max_frequency': ParamConfig(150, int, 'Maximum frequency for spectrogram'),
                'tperseg': ParamConfig(0.25, float, 'Time of each spectrogram segment in seconds'),
                'poverlap': ParamConfig(0.75, float, 'Proportion of overlap between segments for spectrogram'),
                'window': ParamConfig('hann', str, 'Window function for spectrogram'), # Allowed values: 'hann', 'boxcar'
            },

            'time_bin_size': ParamConfig(0.125, float, 'Time bin size in seconds'), # Only relevant for spectrogram = 0, when we are binning raw voltage
        },

        'transformer': {
            'd_model': ParamConfig(192, int, 'Dimensionality of the latent space of the model', dirname_format=lambda x: f'dm{x}'),
            'n_heads': ParamConfig(12, int, 'Number of attention heads', dirname_format=lambda x: f'nh{x}'),
            'n_layers': ParamConfig(5, int, 'Number of transformer layers'),
        },

        'dtype': ParamConfig(torch.float32, torch.dtype, 'Model data type'), # Data type for the model weights and activations 
        'use_mixed_precision': ParamConfig(True, bool, 'Whether to use mixed precision training'), # Helps to conserve GPU memory while not really hurting performance
        'amp_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'AMP data type'), # Data type for the mixed precision training
    },

    'cluster': {
        'wandb_project': ParamConfig("", str, 'wandb.com project name'), # If empty, no wandb is used
        'save_model_every_n_epochs': ParamConfig(1, int, 'Save the model weights and training statistics every n epochs'),
        'eval_model_every_n_epochs': ParamConfig(1, int, 'Evaluate the model every n epochs'),
        'eval_at_beginning': ParamConfig(True, bool, 'Whether to evaluate the model at the beginning of the training'),

        'cache_subjects': ParamConfig(True, bool, 'Whether to cache subjects'), # Whether to cache the subject datasets in RAM, or to load them from the disk as the training proceeds.

        'num_workers_dataloaders': ParamConfig(4, int, 'Number of processes for dataloaders'), # note that you will need to request enough CPU cores to cover all these workers
        'num_workers_eval': ParamConfig(4, int, 'Number of processes for evaluation'),        
        'prefetch_factor': ParamConfig(2, int, 'Prefetch factor'), # for the dataloader workers

        'quick_eval': ParamConfig(False, bool, 'Whether to do quicker evaluation by only evaluating on one fold of the data'), # Whether to do quick evaluation on a subset of the data
        'eval_aggregation_method': ParamConfig('concat', str, 'Evaluation aggregation method'),
    },

    'training': {
        'setup_name': ParamConfig("andrii0", str, 'Setup name', required=True),

        'train_subject_trials': ParamConfig("btbank3_1", str, 'Train subject trials'), # a string like btbank3_1,btbank3_2,...
        'eval_subject_trials': ParamConfig("", str, 'Eval subject trials'), # a string like btbank3_0,btbank3_1,...
        'data_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'Data type for tensors'),
        
        'eval_tasks': ParamConfig("onset,gpt2_surprisal", str, 'Eval tasks from Neuroprobe'), # a list of neuroprobe tasks to evaluate on during pretraining
        
        'n_epochs': ParamConfig(100, int, 'Number of epochs to train'),
        'p_test': ParamConfig(0.1, float, 'Proportion of data to use as a test split (different from the eval)'),

        'optimizer': ParamConfig('Muon', str, 'Optimizer type'),
        'batch_size': ParamConfig(100, int, 'Batch size for training'),
        
        'learning_rate': ParamConfig(0.003, float, 'Learning rate', include_in_dirname=True, dirname_format=lambda x: f'lr{x}'),
        'lr_schedule': ParamConfig('linear', str, 'Learning rate schedule (none, linear, cosine)'),
        'weight_decay': ParamConfig(0.0001, float, 'Weight decay for optimizer', include_in_dirname=True, dirname_format=lambda x: f'wd{x}'),
        'dropout': ParamConfig(0.1, float, 'Dropout rate', include_in_dirname=True, dirname_format=lambda x: f'dr{x}'),
        
        'max_n_electrodes': ParamConfig(128, int, 'Maximum number of electrodes to use during pretraining'),

        'p_electrodes_per_stream': ParamConfig(0.5, float, 'Proportion of electrodes per stream'),
        'future_bin_idx': ParamConfig(1, int, 'Future bin index'),
        'warmup_steps': ParamConfig(100, int, 'Warmup steps'),
        
        'normalize_features': ParamConfig(True, bool, 'Whether to normalize features'),
        'use_temperature_param': ParamConfig(True, bool, 'Whether to use temperature parameter'),
        'max_temperature_param': ParamConfig(1000.0, float, 'Maximum temperature parameter value'),
        
        'random_string': ParamConfig("X", str, 'Random string for seed generation', include_in_dirname=True, dirname_format=lambda x: f'r{x}'),
        'timestamp': ParamConfig(time.strftime("%Y%m%d_%H%M%S"), str, 'Timestamp', include_in_dirname=True, dirname_format=lambda x: f't{x}'), # the time when the model was trained
    },

    # This is a dictionary that can be used to store any additional parameters that are not part of the schema. 
    # For example, if you want to store a param like config['other']['X'] and it have a value '12345' (only strings are supported)
    # You can pass it as an argument to the script like this: --other.X 12345
    'other': {} 
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
    
    def bool_type(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            if x.lower() in ('true', 't', 'yes', 'y', '1'):
                return True
            if x.lower() in ('false', 'f', 'no', 'n', '0'):
                return False
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {x}')
    
    def add_args_from_schema(schema, prefix=''):
        for key, value in schema.items():
            if isinstance(value, ParamConfig):
                arg_name = f'--{prefix}{key}' if prefix else f'--{key}'
                # Use bool_type for boolean parameters
                arg_type = bool_type if value.type == bool else value.type
                parser.add_argument(arg_name, type=arg_type, default=None, help=value.help, required=value.required)
            elif isinstance(value, dict):
                new_prefix = f'{prefix}{key}.' if prefix else f'{key}.'
                add_args_from_schema(value, new_prefix)

    # Add a catch-all argument for any unknown arguments
    parser.add_argument('--other', nargs='*', action='store', help='Additional arguments to be stored in the other dictionary')

    add_args_from_schema(CONFIG_SCHEMA)
    args, unknown = parser.parse_known_args()

    # Initialize other dictionary if it doesn't exist
    if 'other' not in config:
        config['other'] = {}

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

    # Handle unknown arguments
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--other.'):
            key = unknown[i][8:]  # Remove '--other.' prefix
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                config['other'][key] = unknown[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1

def parse_subject_trials_from_config(config):
    train_subject_trials = config['training']['train_subject_trials'] # a string like btbank1_1,btbank1_2,...
    eval_subject_trials = config['training']['eval_subject_trials']
    train_subject_trials = [[subject_identifier.split('_')[0], int(subject_identifier.split('_')[1])] for subject_identifier in train_subject_trials.split(',')] if len(train_subject_trials) > 0 else []
    eval_subject_trials = [[subject_identifier.split('_')[0], int(subject_identifier.split('_')[1])] for subject_identifier in eval_subject_trials.split(',')] if len(eval_subject_trials) > 0 else []
    config['training']['train_subject_trials'] = train_subject_trials
    config['training']['eval_subject_trials'] = eval_subject_trials

def update_dir_name(config):
    dir_name = config['training']['setup_name']
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


# Testing the config parsing
if __name__ == "__main__":
    ### LOADING CONFIGS ###

    config = get_default_config(random_string="TEMP", wandb_project="") # Outputs a dictionary, see utils/training_config.py for how it looks like
    parse_config_from_args(config) # Parses the command line arguments and updates the config dictionary
    parse_subject_trials_from_config(config) # Parses the subject trials from the config dictionary

    dir_name = update_dir_name(config) # This is used to name the directory where the model is saved, based on the training parameters (the config)
    update_random_seed(config) # This is used to set the random seed for the model (for reproducibility)

    print("CONFIG:")
    print(config)