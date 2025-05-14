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
        'weight_decay': 0.0001, #0.0
        'p_electrodes_per_stream': 0.5,
        'symmetric_loss': True,
        'future_bin_idx': 1,
        'projection_type': None, # None, 'random_batch'
        'p_unmasked': 1.0,
        'lr_schedule': "linear", # none, linear, cosine
        'warmup_steps': 100,
        
        # MINI-BFM on braintreebank
        'train_subject_trials': [('btbank3', 0)],#, ('btbank3', 1)], #[("btbank1", 0), ("btbank1", 1), ("btbank2", 4), ("btbank2", 5), ("btbank3", 1), ("btbank3", 2), ("btbank7", 1), ("btbank10", 1)],
        'eval_subject_trials': [('btbank3', 1)], #[("btbank1", 2), ("btbank2", 6), ("btbank3", 0), ("btbank7", 0), ("btbank10", 0)],

        'n_electrodes_subset': 50,

        'normalize_features': True,
        'use_temperature_param': True,
        'max_temperature_param': 1000.0, # Clipping the temperature parameter at this value during training

        'p_show_a_embedding': 0.0,
        'p_show_b_embedding': 1.0,

        'p_unmasked_electrodes': 0.5, # XXX TODO, not added to the codebase yet properly
        'p_masked_timebins': 0.5, # this is implemented tho
        
        'data_dtype': torch.bfloat16,

        'random_string': random_string,
    }
    model_config = {
        'name': 'M',

        'sample_timebin_size': 0.125, # in seconds
        'max_frequency_bin': 64, # XXX Todo: make this based on frequency and not bin number
        'max_n_timebins': 8,
        'max_n_electrodes': 80,

        'init_normalization': True, # XXX rename to a more sensible name later
        'init_identity': True,

        'electrode_embedding': {
            'type': 'learned', # coordinate_init, noisy_coordinate, learned, zero
            'coordinate_noise_std': 0.0, # only relevant for noisy_coordinate type; note coordinates are normalized to be within [0,1]
            'embedding_dim': None,
            'spectrogram': False,
            'spectrogram_power': False,
        },

        'dtype': torch.float32,

        'bin_encoder': "transformer", # "linear" or "transformer"
        'separate_unembed': True,

        'transformer': {
            'd_model': 192,
            'd_model_bin': 192,
            'n_heads': 12,
            'n_layers_electrode': 5,
            'n_layers_time': 5,
            'dropout': 0.2,
            'momentum': 0.99,
            'use_cls_token': True,
        },

        'use_mixed_precision': True,
        'amp_dtype': torch.bfloat16,  # must be bfloat16 because scaler is not existing
    }
    cluster_config = {
        'save_model_every_n_epochs': 1,
        'eval_model_every_n_epochs': 2,

        'wandb_project': wandb_project,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),

        'cache_subjects': True,

        'num_workers_dataloaders': 4,
        'num_workers_eval': 4,
        'prefetch_factor': 2,

        'resume_run': False,
        'quick_eval': True,

        'eval_aggregation_method': 'concat', # 'mean', 'concat'
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
    parser.add_argument('--max_frequency_bin', type=int, default=None, help='Maximum frequency bin')
    parser.add_argument('--sample_timebin_size', type=float, default=None, help='Sample timebin size in seconds')
    parser.add_argument('--max_n_timebins', type=int, default=None, help='Maximum number of time bins')
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for EMA')
    parser.add_argument('--resume_run', type=int, default=None, help='Whether to resume run')
    parser.add_argument('--projection_type', type=str, default=None, help='Projection type')
    parser.add_argument('--p_unmasked', type=float, default=None, help='Proportion of unmasked electrodes')
    parser.add_argument('--n_electrodes_subset', type=int, default=None, help='Number of electrodes subset')
    parser.add_argument('--normalize_features', type=int, default=None, help='Whether to normalize features')
    parser.add_argument('--use_temperature_param', type=int, default=None, help='Whether to use temperature parameter') 
    parser.add_argument('--max_temperature_param', type=float, default=None, help='Maximum temperature parameter value')
    parser.add_argument('--p_show_a_embedding', type=float, default=None, help='Proportion of A embedding to show')
    parser.add_argument('--p_show_b_embedding', type=float, default=None, help='Proportion of B embedding to show')
    parser.add_argument('--separate_unembed', type=int, default=None, help='Whether to use separate unembed')
    parser.add_argument('--p_masked_timebins', type=float, default=None, help='Proportion of masked timebins')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default=None, help='Optimizer type')
    parser.add_argument('--p_electrodes_per_stream', type=float, default=None, help='Proportion of electrodes per stream')
    parser.add_argument('--symmetric_loss', type=int, default=None, help='Whether to use symmetric pretraining')
    parser.add_argument('--spectrogram', type=int, default=None, help='Whether to use spectrogram')
    parser.add_argument('--spectrogram_power', type=int, default=None, help='Whether to use spectrogram power')
    parser.add_argument('--future_bin_idx', type=int, default=None, help='Future bin index')
    parser.add_argument('--eval_aggregation_method', type=str, default=None, help='Feature aggregation method')
    parser.add_argument('--use_cls_token', type=int, default=None, help='Whether to use CLS token')
    parser.add_argument('--lr_schedule', type=str, default=None, help='Learning rate schedule (none, linear, cosine)')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
    
    # Other model config
    parser.add_argument('--init_normalization', type=int, default=None, help='Whether to use initial normalization')
    parser.add_argument('--init_identity', type=int, default=None, help='Whether to use identity initialization')
    parser.add_argument('--bin_encoder', type=str, default=None, help='Bin encoder (linear, transformer)')

    # Electrode embedding config
    parser.add_argument('--electrode_embedding_type', type=str, default=None, help='Type of electrode embedding')
    parser.add_argument('--electrode_embedding_coordinate_noise_std', type=float, default=None, help='Coordinate noise std for electrode embedding')

    parser.add_argument('--cache_subjects', type=int, default=None, help='Whether to cache subjects')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--num_workers_dataloaders', type=int, default=None, help='Number of workers for dataloaders')
    parser.add_argument('--random_string', type=str, default=None, help='Random string for seed generation')
    parser.add_argument('--save_model_every_n_epochs', type=int, default=None, help='Save model every n epochs')
    parser.add_argument('--n_epochs', type=int, default=None, help='Number of epochs to train')

    # train subject trials
    parser.add_argument('--train_subject_trials', type=str, default=None, help='Train subject trials')
    parser.add_argument('--eval_subject_trials', type=str, default=None, help='Eval subject trials')
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
    if args.init_normalization is not None:
        model_config['init_normalization'] = bool(args.init_normalization)
    if args.init_identity is not None:
        model_config['init_identity'] = bool(args.init_identity)
    if args.cache_subjects is not None:
        cluster_config['cache_subjects'] = bool(args.cache_subjects)
    if args.p_masked_timebins is not None:
        training_config['p_masked_timebins'] = args.p_masked_timebins
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
    if args.max_frequency_bin is not None:
        model_config['max_frequency_bin'] = args.max_frequency_bin if args.max_frequency_bin != -1 else None
    if args.bin_encoder is not None:
        model_config['bin_encoder'] = args.bin_encoder
    if args.separate_unembed is not None:
        model_config['separate_unembed'] = bool(args.separate_unembed)
    if args.p_show_a_embedding is not None:
        training_config['p_show_a_embedding'] = args.p_show_a_embedding
    if args.p_show_b_embedding is not None:
        training_config['p_show_b_embedding'] = args.p_show_b_embedding
    if args.max_temperature_param is not None:
        training_config['max_temperature_param'] = args.max_temperature_param
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
    if args.symmetric_loss is not None:
        training_config['symmetric_loss'] = bool(args.symmetric_loss)
    if args.save_model_every_n_epochs is not None:
        cluster_config['save_model_every_n_epochs'] = args.save_model_every_n_epochs
    if args.sample_timebin_size is not None:
        model_config['sample_timebin_size'] = args.sample_timebin_size
    if args.max_n_timebins is not None:
        model_config['max_n_timebins'] = args.max_n_timebins
    if args.spectrogram is not None:
        model_config['electrode_embedding']['spectrogram'] = bool(args.spectrogram)
    if args.spectrogram_power is not None:
        model_config['electrode_embedding']['spectrogram_power'] = bool(args.spectrogram_power)
    if args.n_epochs is not None:
        training_config['n_epochs'] = args.n_epochs
    if args.momentum is not None:
        model_config['transformer']['momentum'] = args.momentum
    if args.future_bin_idx is not None:
        training_config['future_bin_idx'] = args.future_bin_idx
    if args.eval_aggregation_method is not None:
        cluster_config['eval_aggregation_method'] = args.eval_aggregation_method
    if args.resume_run is not None:
        cluster_config['resume_run'] = bool(args.resume_run)
    if args.use_cls_token is not None:
        model_config['transformer']['use_cls_token'] = bool(args.use_cls_token)
    if args.projection_type is not None:
        training_config['projection_type'] = args.projection_type
    if args.p_unmasked is not None:
        training_config['p_unmasked'] = args.p_unmasked
    if args.lr_schedule is not None:
        training_config['lr_schedule'] = args.lr_schedule
    if args.n_electrodes_subset is not None:
        training_config['n_electrodes_subset'] = args.n_electrodes_subset
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
    #dir_name += f"_t{model_config['max_n_timebins']}"
    #dir_name += f"_st{model_config['sample_timebin_size']}"
    dir_name += f"_nst{len(training_config['train_subject_trials'])}"

    dir_name += f"_dm{model_config['transformer']['d_model']}"
    dir_name += f"_dmb{model_config['transformer']['d_model_bin']}"
    dir_name += f"_nh{model_config['transformer']['n_heads']}"
    dir_name += f"_nl{model_config['transformer']['n_layers_electrode']}" + f"_{model_config['transformer']['n_layers_time']}"
    if training_config['n_electrodes_subset'] != 60:
        dir_name += f"_nes{training_config['n_electrodes_subset']}"
    if training_config['normalize_features']:
        dir_name += f"_nf"
    if not training_config['use_temperature_param']:
        dir_name += f"_nUTP"

    if model_config['bin_encoder'] != "linear":
        dir_name += f"_be{model_config['bin_encoder'].upper()[0]}"
    # if not cluster_config['cache_subjects']:
    #     dir_name += f"_nCS"
    # if not model_config['init_normalization']:
    #     dir_name += f"_nIN"
    if not model_config['init_identity']:
        dir_name += f"_nII"
    # if not training_config['symmetric_loss']:
    #     dir_name += f"_nSL"
    # if not model_config['electrode_embedding']['spectrogram']:
    #     dir_name += f"_nSP"
    # if model_config['electrode_embedding']['spectrogram_power']:
    #     dir_name += f"_nSPP"
    if cluster_config['eval_aggregation_method'] != 'concat':
        dir_name += f"_ea{cluster_config['eval_aggregation_method'][0].upper()}"
    if training_config['p_masked_timebins'] != 0.5:
        dir_name += f"_pmt{training_config['p_masked_timebins']}"
    if training_config['max_temperature_param'] != 1000.0:
        dir_name += f"_mtp{training_config['max_temperature_param']}"
    # if training_config['projection_type'] is not None:
    #     dir_name += f"_proj{''.join([x[0] for x in training_config['projection_type'].upper().split('_')])}"
    # if training_config['p_unmasked'] != 1.0:
    #     dir_name += f"_pum{training_config['p_unmasked']}"
    if training_config['p_show_a_embedding'] != 0.0:
        dir_name += f"_psa{training_config['p_show_a_embedding']}"
    if training_config['p_show_b_embedding'] != 1.0:
        dir_name += f"_psb{training_config['p_show_b_embedding']}"
    if model_config['separate_unembed']:
        dir_name += f"_SU"

    if model_config['sample_timebin_size'] != 0.125:
        dir_name += f"_stbs{model_config['sample_timebin_size']}"
    # if model_config['max_n_timebins'] != 24:
    #     dir_name += f"_mxt{model_config['max_n_timebins']}"

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

    # if model_config['transformer']['momentum'] != 0.99:
    #     dir_name += f"_m{model_config['transformer']['momentum']}"
    # if training_config['future_bin_idx'] != 0:
    #     dir_name += f"_fb{training_config['future_bin_idx']}"
    # if model_config['transformer']['use_cls_token']:
    #     dir_name += f"_cls"

    if 'p_electrodes_per_stream' in training_config and training_config['p_electrodes_per_stream'] != 0.5:
        dir_name += f"_pps{training_config['p_electrodes_per_stream']}"
    if model_config['electrode_embedding']['embedding_dim'] is not None:
        dir_name += f"_ed{model_config['electrode_embedding']['embedding_dim']}"
    # if model_config['max_frequency_bin'] != 64:
    #     dir_name += f"_mbf{model_config['max_frequency_bin']}"
    # if model_config['transformer']['dropout'] != 0.2:
    #     dir_name += f"_dr{model_config['transformer']['dropout']}"
    if training_config['batch_size'] != 100:
        dir_name += f"_bs{training_config['batch_size']}"
    if training_config['weight_decay'] != 0.0:
        dir_name += f"_wd{training_config['weight_decay']}"
    # dir_name += f"_lr{training_config['learning_rate']}"
    if training_config['optimizer'] != 'Muon':
        dir_name += f"_opt{training_config['optimizer']}"
    # if training_config['lr_schedule'] != "None":
    #     dir_name += f"_lr{training_config['lr_schedule'][0].upper()}"
    # if training_config['warmup_steps'] != 0:
    #     dir_name += f"_ws{training_config['warmup_steps']}"
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