import time
import psutil
import torch
import numpy as np
import argparse

def get_default_configs(random_string, wandb_project):
    training_config = {
        'n_epochs': 100,
        'p_test': 0.1,

        'optimizer': 'Muon',
        'batch_size': 100,
        'learning_rate': 0.003,
        'weight_decay': 0.0001,
        'p_electrodes_per_stream': 0.5,
        'future_bin_idx': 1,
        'lr_schedule': "linear", # none, linear, cosine
        'warmup_steps': 100,
        
        'train_subject_trials': [('btbank3', 1)],
        'eval_subject_trials': [('btbank3', 0)],

        'normalize_features': True,
        'use_temperature_param': True,
        'max_temperature_param': 1000.0,
        
        'data_dtype': torch.bfloat16,

        'random_string': random_string,
    }
    model_config = {
        'name': 'M',

        'sample_timebin_size': 0.125, # in seconds
        'context_length': 2, # in seconds
        'max_frequency': 200,
        'max_n_electrodes': 80,

        'electrode_embedding': {
            'type': 'learned',
            'coordinate_noise_std': 0.0,
            'embedding_dim': None,
            'spectrogram': True,
            'spectrogram_power': True,
        },

        'dtype': torch.float32,

        'transformer': {
            'd_model': 192,
            'd_model_bin': 192,
            'n_heads': 12,
            'n_layers_electrode': 5,
            'n_layers_time': 5,
            'dropout': 0.1,
        },

        'laplacian_rereference': True,

        'use_mixed_precision': True,
        'amp_dtype': torch.bfloat16,
    }
    cluster_config = {
        'save_model_every_n_epochs': 5,
        'eval_model_every_n_epochs': 5,

        'wandb_project': wandb_project,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),

        'cache_subjects': True,

        'num_workers_dataloaders': 4,
        'num_workers_eval': 4,
        'prefetch_factor': 2,

        'quick_eval': True,

        'eval_aggregation_method': 'concat',
    }
    return training_config, model_config, cluster_config


def parse_configs_from_args(training_config, model_config, cluster_config):
    parser = argparse.ArgumentParser()
    
    # Transformer model arguments
    parser.add_argument('--d_model', type=int, default=None, help='Dimension of transformer model')
    parser.add_argument('--d_model_bin', type=int, default=None, help='Dimension of transformer model')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--n_layers_electrode', type=int, default=None, help='Number of transformer layers for electrode path')
    parser.add_argument('--n_layers_time', type=int, default=None, help='Number of transformer layers for time path')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Dimension of electrode embeddings')
    parser.add_argument('--max_frequency', type=int, default=None, help='Maximum frequency bin')
    parser.add_argument('--sample_timebin_size', type=float, default=None, help='Sample timebin size in seconds')
    parser.add_argument('--context_length', type=float, default=None, help='Context length in seconds')
    parser.add_argument('--max_n_electrodes', type=int, default=None, help='Maximum number of electrodes to use')
    parser.add_argument('--normalize_features', type=int, default=None, help='Whether to normalize features')
    parser.add_argument('--use_temperature_param', type=int, default=None, help='Whether to use temperature parameter') 
    parser.add_argument('--max_temperature_param', type=float, default=None, help='Maximum temperature parameter value')
    parser.add_argument('--electrode_embedding_type', type=str, default=None, help='Type of electrode embedding')
    parser.add_argument('--electrode_embedding_coordinate_noise_std', type=float, default=None, help='Coordinate noise std for electrode embedding')
    parser.add_argument('--cache_subjects', type=int, default=None, help='Whether to cache subjects')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--num_workers_dataloaders', type=int, default=None, help='Number of workers for dataloaders')
    parser.add_argument('--random_string', type=str, default=None, help='Random string for seed generation')
    parser.add_argument('--save_model_every_n_epochs', type=int, default=None, help='Save model every n epochs')
    parser.add_argument('--n_epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--train_subject_trials', type=str, default=None, help='Train subject trials')
    parser.add_argument('--eval_subject_trials', type=str, default=None, help='Eval subject trials')
    parser.add_argument('--future_bin_idx', type=int, default=None, help='Future bin index')
    parser.add_argument('--lr_schedule', type=str, default=None, help='Learning rate schedule (none, linear, cosine)')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default=None, help='Optimizer type')
    parser.add_argument('--p_electrodes_per_stream', type=float, default=None, help='Proportion of electrodes per stream')
    args = parser.parse_args()
    
    # Update configs with command line args if provided
    if args.d_model is not None:
        model_config['transformer']['d_model'] = args.d_model
    if args.d_model_bin is not None:
        model_config['transformer']['d_model_bin'] = args.d_model_bin
    if args.n_heads is not None:
        model_config['transformer']['n_heads'] = args.n_heads
    if args.n_layers_electrode is not None:
        model_config['transformer']['n_layers_electrode'] = args.n_layers_electrode
    if args.n_layers_time is not None:
        model_config['transformer']['n_layers_time'] = args.n_layers_time
    if args.dropout is not None:
        model_config['transformer']['dropout'] = args.dropout
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        training_config['weight_decay'] = args.weight_decay
    if args.optimizer is not None:
        training_config['optimizer'] = args.optimizer
    if args.p_electrodes_per_stream is not None:
        training_config['p_electrodes_per_stream'] = args.p_electrodes_per_stream
    if args.cache_subjects is not None:
        cluster_config['cache_subjects'] = bool(args.cache_subjects)
    if args.random_string is not None:
        training_config['random_string'] = args.random_string
    if args.electrode_embedding_type is not None:
        model_config['electrode_embedding']['type'] = args.electrode_embedding_type
    if args.electrode_embedding_coordinate_noise_std is not None:
        model_config['electrode_embedding']['coordinate_noise_std'] = args.electrode_embedding_coordinate_noise_std
    if args.wandb_project is not None:
        cluster_config['wandb_project'] = args.wandb_project
    if args.embedding_dim is not None:
        model_config['electrode_embedding']['embedding_dim'] = args.embedding_dim
    if args.max_frequency is not None:
        model_config['max_frequency'] = args.max_frequency if args.max_frequency != -1 else None
    if args.train_subject_trials is not None:
        train_subject_trials = []
        for subject_trial in args.train_subject_trials.replace(" ", "").split(","):
            subject_identifier, trial_id = subject_trial.split("_")[0], subject_trial.split("_")[1]
            trial_id = int(trial_id)
            train_subject_trials.append((subject_identifier, trial_id))
        training_config['train_subject_trials'] = train_subject_trials
    if args.eval_subject_trials is not None:
        eval_subject_trials = []
        for subject_trial in args.eval_subject_trials.replace(" ", "").split(","):
            if "_" not in subject_trial:
                continue # empty string
            subject_identifier, trial_id = subject_trial.split("_")[0], subject_trial.split("_")[1]
            trial_id = int(trial_id)
            eval_subject_trials.append((subject_identifier, trial_id))
        training_config['eval_subject_trials'] = eval_subject_trials
    if args.num_workers_dataloaders is not None:
        cluster_config['num_workers_dataloaders'] = args.num_workers_dataloaders
    if args.save_model_every_n_epochs is not None:
        cluster_config['save_model_every_n_epochs'] = args.save_model_every_n_epochs
    if args.sample_timebin_size is not None:
        model_config['sample_timebin_size'] = args.sample_timebin_size
    if args.context_length is not None:
        model_config['context_length'] = args.context_length
    if args.n_epochs is not None:
        training_config['n_epochs'] = args.n_epochs
    if args.future_bin_idx is not None:
        training_config['future_bin_idx'] = args.future_bin_idx
    if args.lr_schedule is not None:
        training_config['lr_schedule'] = args.lr_schedule
    if args.max_n_electrodes is not None:
        model_config['max_n_electrodes'] = args.max_n_electrodes
    if args.normalize_features is not None:
        training_config['normalize_features'] = bool(args.normalize_features)
    if args.use_temperature_param is not None:
        training_config['use_temperature_param'] = bool(args.use_temperature_param)
    if args.warmup_steps is not None:
        training_config['warmup_steps'] = args.warmup_steps

max_log_priority = 1
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:.1f}G ram {ram_usage:.1f}G] {' '*4*indent}{message}")


def update_random_seed(training_config):
    random_seed = hash(training_config['random_string']) % (2**32)
    training_config['random_seed'] = random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    return random_seed


def update_dir_name(model_config, training_config, cluster_config):
    dir_name = model_config['name']
    dir_name += f"_nst{len(training_config['train_subject_trials'])}"

    dir_name += f"_dm{model_config['transformer']['d_model']}"
    dir_name += f"_dmb{model_config['transformer']['d_model_bin']}"
    dir_name += f"_nh{model_config['transformer']['n_heads']}"
    dir_name += f"_nl{model_config['transformer']['n_layers_electrode']}" + f"_{model_config['transformer']['n_layers_time']}"
    if model_config['max_n_electrodes'] != 80:
        dir_name += f"_nes{model_config['max_n_electrodes']}"
    if training_config['normalize_features']:
        dir_name += f"_nf"
    if not training_config['use_temperature_param']:
        dir_name += f"_nUTP"

    if model_config['sample_timebin_size'] != 0.125:
        dir_name += f"_stbs{model_config['sample_timebin_size']}"

    if model_config['electrode_embedding']['type'] == 'coordinate_init':
        dir_name += f"_eeCI"
    elif model_config['electrode_embedding']['type'] == 'noisy_coordinate':
        dir_name += f"_eeNC_ecns{model_config['electrode_embedding']['coordinate_noise_std']}"
    elif model_config['electrode_embedding']['type'] == 'learned':
        dir_name += f"_eeL"
    elif model_config['electrode_embedding']['type'] == 'zero':
        dir_name += f"_eeZ"
    else:
        dir_name += f"_ee{model_config['electrode_embedding']['type'].upper()}"

    if 'p_electrodes_per_stream' in training_config and training_config['p_electrodes_per_stream'] != 0.5:
        dir_name += f"_pps{training_config['p_electrodes_per_stream']}"
    if model_config['electrode_embedding']['embedding_dim'] is not None:
        dir_name += f"_ed{model_config['electrode_embedding']['embedding_dim']}"
    if training_config['batch_size'] != 100:
        dir_name += f"_bs{training_config['batch_size']}"
    if training_config['weight_decay'] != 0.0:
        dir_name += f"_wd{training_config['weight_decay']}"
    if training_config['optimizer'] != 'Muon':
        dir_name += f"_opt{training_config['optimizer']}"
    dir_name += f"_r{training_config['random_string']}"
    cluster_config['dir_name'] = dir_name
    return dir_name

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