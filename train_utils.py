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
    dirname_format: Optional[str] = None
    default: Any = None

CONFIG_SCHEMA = {
    'training': {
        'n_epochs': ParamConfig(100, int, 'Number of epochs to train'),
        'p_test': ParamConfig(0.1, float, 'Proportion of data to use for testing'),
        'optimizer': ParamConfig('Muon', str, 'Optimizer type'),
        'batch_size': ParamConfig(100, int, 'Batch size for training'),
        'learning_rate': ParamConfig(0.003, float, 'Learning rate'),
        'weight_decay': ParamConfig(0.0001, float, 'Weight decay for optimizer'),
        'p_electrodes_per_stream': ParamConfig(0.5, float, 'Proportion of electrodes per stream'),
        'future_bin_idx': ParamConfig(1, int, 'Future bin index'),
        'lr_schedule': ParamConfig('linear', str, 'Learning rate schedule (none, linear, cosine)'),
        'warmup_steps': ParamConfig(100, int, 'Warmup steps'),
        'train_subject_trials': ParamConfig("btbank3_1", str, 'Train subject trials'), # a string like btbank3_1,btbank3_2,...
        'eval_subject_trials': ParamConfig("btbank3_0", str, 'Eval subject trials'), # a string like btbank3_0,btbank3_1,...
        'normalize_features': ParamConfig(True, bool, 'Whether to normalize features'),
        'use_temperature_param': ParamConfig(True, bool, 'Whether to use temperature parameter'),
        'max_temperature_param': ParamConfig(1000.0, float, 'Maximum temperature parameter value'),
        'data_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'Data type for tensors'),
        'random_string': ParamConfig("X", str, 'Random string for seed generation', include_in_dirname=True, dirname_format='r{}'),
    },

    'model': {
        'name': ParamConfig('M', str, 'Model name'),
        'sample_timebin_size': ParamConfig(0.125, float, 'Sample timebin size in seconds', dirname_format='stbs{}'),
        'context_length': ParamConfig(3, float, 'Context length in seconds'),
        'max_frequency': ParamConfig(200, int, 'Maximum frequency bin'),
        'max_n_electrodes': ParamConfig(128, int, 'Maximum number of electrodes to use', dirname_format='nes{}'),
        'electrode_embedding': {
            'type': ParamConfig('learned', str, 'Type of electrode embedding', dirname_format='ee{}'),
            'coordinate_noise_std': ParamConfig(0.0, float, 'Coordinate noise std for electrode embedding'),
            'embedding_dim': ParamConfig(None, int, 'Dimension of electrode embeddings', dirname_format='ed{}'),
            'spectrogram': ParamConfig(True, bool, 'Whether to use spectrogram'),
            'spectrogram_power': ParamConfig(True, bool, 'Whether to use spectrogram power'),
        },
        'dtype': ParamConfig(torch.float32, torch.dtype, 'Model data type'),
        'transformer': {
            'd_model': ParamConfig(192, int, 'Dimension of transformer model', dirname_format='dm{}'),
            'd_model_bin': ParamConfig(192, int, 'Dimension of transformer model', dirname_format='dmb{}'),
            'n_heads': ParamConfig(12, int, 'Number of attention heads', dirname_format='nh{}'),
            'n_layers_electrode': ParamConfig(5, int, 'Number of transformer layers for electrode path', dirname_format='nl{}'),
            'n_layers_time': ParamConfig(5, int, 'Number of transformer layers for time path'),
            'dropout': ParamConfig(0.1, float, 'Dropout rate', include_in_dirname=True, dirname_format='dr{}'),
        },
        'laplacian_rereference': ParamConfig(True, bool, 'Whether to use Laplacian rereference'),
        'use_mixed_precision': ParamConfig(True, bool, 'Whether to use mixed precision'),
        'amp_dtype': ParamConfig(torch.bfloat16, torch.dtype, 'AMP data type'),
    },

    'cluster': {
        'save_model_every_n_epochs': ParamConfig(5, int, 'Save model every n epochs'),
        'eval_model_every_n_epochs': ParamConfig(5, int, 'Eval model every n epochs'),
        'wandb_project': ParamConfig("", str, 'Wandb project name'),
        'timestamp': ParamConfig(time.strftime("%Y%m%d_%H%M%S"), str, 'Timestamp'),
        'cache_subjects': ParamConfig(True, bool, 'Whether to cache subjects'),
        'num_workers_dataloaders': ParamConfig(4, int, 'Number of workers for dataloaders'),
        'num_workers_eval': ParamConfig(4, int, 'Number of workers for evaluation'),
        'prefetch_factor': ParamConfig(2, int, 'Prefetch factor'),
        'quick_eval': ParamConfig(True, bool, 'Whether to do quick evaluation'),
        'eval_aggregation_method': ParamConfig('concat', str, 'Evaluation aggregation method'),
    }
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
                        dir_name += '_' + value.dirname_format.format(config_value)
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

# Get shared memory info
def get_shared_memory_info():
    # On Linux, you can get /dev/shm info
    try:
        shm_usage = psutil.disk_usage('/dev/shm')
        total_gb = shm_usage.total / (1024**3)
        used_gb = shm_usage.used / (1024**3)
        return total_gb, used_gb
    except:
        return None, None