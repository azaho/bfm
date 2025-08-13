import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast
import gc
import argparse

from utils.muon_optimizer import Muon
from subject.dataset import load_subjects
from evaluation.neuroprobe_tasks import FrozenModelEvaluation_SS_SM
from training_setup.training_config import log, update_dir_name, update_random_seed, parse_config_from_args, get_default_config, parse_subject_trials_from_config, convert_dtypes, unconvert_dtypes
from torch.optim.lr_scheduler import ChainedScheduler

### PARSE ARGUMENTS ###

parser = argparse.ArgumentParser(description='Finetune a pretrained model')
parser.add_argument('--pretrained_model_dir', type=str, required=True, 
                   help='Directory containing the pretrained model (e.g., "andrii0_podcast_wd0.001_dr0.1_r12345")')
parser.add_argument('--pretrained_model_epoch', type=int, default=-1, 
                   help='Epoch of the pretrained model to load (default: latest/final)')
parser.add_argument('--finetune_mode', type=str, choices=['full', 'electrode_embeddings', 'alignment_components'], default='full',
                   help='Finetune mode: "full" for entire model, "electrode_embeddings" for only electrode embeddings, "alignment_components" for electrode embeddings + shared projection layer')
args, unknown = parser.parse_known_args()

# Add the unknown arguments to sys.argv so they can be parsed by the config system
import sys
sys.argv = [sys.argv[0]] + unknown

# Print usage example if help is requested
if '--help' in sys.argv or '-h' in sys.argv:
    print("\n=== USAGE EXAMPLES ===")
    print("# Finetune entire model:")
    print("python finetune.py --pretrained_model_dir andrii0_podcast_wd0.001_dr0.1_r12345 --finetune_mode full --training.setup_name andrii0_podcast --training.learning_rate 0.001 --training.n_epochs 50")
    print("\n# Finetune only electrode embeddings:")
    print("python finetune.py --pretrained_model_dir andrii0_podcast_wd0.001_dr0.1_r12345 --finetune_mode electrode_embeddings --training.setup_name andrii0_podcast --training.learning_rate 0.001 --training.n_epochs 50")
    print("\n# Finetune alignment components (electrode embeddings + shared projection):")
    print("python finetune.py --pretrained_model_dir andrii0_podcast_wd0.001_dr0.1_r12345 --finetune_mode alignment_components --training.setup_name andrii0_podcast_lambda --training.learning_rate 0.001 --training.n_epochs 50")
    print("\n# Load specific epoch:")
    print("python finetune.py --pretrained_model_dir andrii0_podcast_wd0.001_dr0.1_r12345 --pretrained_model_epoch 100 --finetune_mode full --training.setup_name andrii0_podcast --training.learning_rate 0.001 --training.n_epochs 50")
    print("========================\n")

### LOADING CONFIGS ###

config = get_default_config(random_string="TEMP", wandb_project="podcast") # Outputs a dictionary, see utils/training_config.py for how it looks like
parse_config_from_args(config) # Parses the command line arguments and updates the config dictionary
parse_subject_trials_from_config(config) # Parses the subject trials from the config dictionary

# Generate the finetune directory name
finetune_dir_name = update_dir_name(config) # This is used to name the directory where the model is saved, based on the training parameters (the config)
update_random_seed(config) # This is used to set the random seed for the model (for reproducibility)

# Create the full directory path: runs/data/[original_model_dir]/[finetune_dir_name]
original_model_dir = args.pretrained_model_dir
full_dir_name = os.path.join(original_model_dir, finetune_dir_name)

# Update the config to use the full directory path
config['cluster']['dir_name'] = full_dir_name

log(f"Original model directory: {original_model_dir}", priority=0)
log(f"Finetune directory name: {finetune_dir_name}", priority=0)
log(f"Full directory path: {full_dir_name}", priority=0)

# Weights & Biases dashboard setup
config['cluster']['wandb_name'] = config['cluster']['dir_name'] # This is used to name the wandb run
if len(config['cluster']['wandb_project'])==0: wandb = False # If wandb project is not set, we do not use wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['device'] = device
log(f"Using device: {device}", priority=0)

### LOAD PRETRAINED MODEL CONFIG ###

# Load the pretrained model checkpoint to get its config
pretrained_model_dir = args.pretrained_model_dir
pretrained_model_epoch = args.pretrained_model_epoch

# Determine the epoch to load
if pretrained_model_epoch < 0:
    # Try to find the latest epoch
    model_dir_path = os.path.join("runs/data", pretrained_model_dir)
    if os.path.exists(model_dir_path):
        model_files = [f for f in os.listdir(model_dir_path) if f.startswith("model_epoch_") and f.endswith(".pth")]
        if model_files:
            # Extract epoch numbers and find the latest
            epochs = []
            for f in model_files:
                try:
                    epoch_str = f.replace("model_epoch_", "").replace(".pth", "")
                    if epoch_str == "final":
                        epochs.append(float('inf'))
                    else:
                        epochs.append(int(epoch_str))
                except:
                    continue
            if epochs:
                latest_epoch = max(epochs)
                pretrained_model_epoch = "final" if latest_epoch == float('inf') else latest_epoch
                log(f"Found latest epoch: {pretrained_model_epoch}", priority=0)
            else:
                pretrained_model_epoch = "final"
        else:
            pretrained_model_epoch = "final"
    else:
        raise FileNotFoundError(f"Pretrained model directory not found: {model_dir_path}")

# Load the pretrained model checkpoint
checkpoint_path = os.path.join("runs/data", pretrained_model_dir, f"model_epoch_{pretrained_model_epoch}.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Pretrained model checkpoint not found: {checkpoint_path}")

log(f"Loading pretrained model from: {checkpoint_path}", priority=0)
checkpoint = torch.load(checkpoint_path, map_location=device)
pretrained_config = unconvert_dtypes(checkpoint['config'])

# Update current config with pretrained model's config (but keep current training settings)
for key, value in pretrained_config.items():
    if key not in config:
        config[key] = value
    elif key == 'model':
        # Merge model config but keep current training settings
        for model_key, model_value in value.items():
            if model_key not in config[key]:
                config[key][model_key] = model_value
    elif key == 'cluster':
        # Keep current cluster settings
        pass
    elif key == 'training':
        # Keep current training settings but ensure setup_name matches
        if 'setup_name' not in config[key]:
            config[key]['setup_name'] = value['setup_name']

log(f"Loaded pretrained model config from epoch {pretrained_model_epoch}", priority=0)

### FINETUNING SUMMARY ###

# Get finetune mode from args
finetune_mode = args.finetune_mode

log(f"=== FINETUNING SETUP ===", priority=0)
log(f"Pretrained model: {pretrained_model_dir}", priority=0)
log(f"Pretrained epoch: {pretrained_model_epoch}", priority=0)
log(f"Finetune mode: {finetune_mode}", priority=0)
log(f"Learning rate: {config['training']['learning_rate']}", priority=0)
log(f"Batch size: {config['training']['batch_size']}", priority=0)
log(f"Epochs: {config['training']['n_epochs']}", priority=0)
log(f"=========================", priority=0)

### LOAD SUBJECTS ###

log(f"Loading subjects...", priority=0)
# all_subjects is a dictionary of subjects, with the subject identifier as the key and the subject object as the value
all_subjects = load_subjects(config['training']['train_subject_trials'], 
                             config['training']['eval_subject_trials'], config['training']['data_dtype'], 
                             cache=config['cluster']['cache_subjects'], allow_corrupted=False)

### LOADING TRAINING SETUP ###

# Import the training setup class dynamically based on config
training_setup_name = config["training"]["setup_name"].lower() # if this is X, the filename should be trianing_setup/X.py and the class name should be XTrainingSetup
try:
    setup_module = __import__(f'training_setup.{training_setup_name}', fromlist=[training_setup_name])
    setup_class = getattr(setup_module, training_setup_name)
    training_setup = setup_class(all_subjects, config, verbose=True)
except (ImportError, AttributeError) as e:
    print(f"Could not load training setup '{config['training']['setup_name']}'. Are you sure the filename and the class name are the same and correspond to the parameter? Error: {str(e)}")
    exit()

# Save a copy of the training setup file for reproducibility
import shutil
setup_file = f'training_setup/{training_setup_name}.py'
training_setup_dir = os.path.join('runs/data', full_dir_name, 'training_setup')
os.makedirs(training_setup_dir, exist_ok=True)
shutil.copy2(setup_file, training_setup_dir)    

### LOAD MODEL ###

log(f"Loading model...", priority=0)
training_setup.initialize_model()

# Load pretrained weights
log(f"Loading pretrained weights from {checkpoint_path}...", priority=0)

# Temporarily set the config's dir_name to the pretrained model directory for loading
original_dir_name = config['cluster']['dir_name']
config['cluster']['dir_name'] = pretrained_model_dir

# Load the pretrained weights
training_setup.load_model(pretrained_model_epoch, load_from_dir="runs/data/")

# Restore the full directory name for saving
config['cluster']['dir_name'] = original_dir_name

# Set up finetuning mode
if finetune_mode == 'electrode_embeddings':
    log(f"Finetuning mode: Only electrode embeddings (freezing main model)", priority=0)
    
    # Debug: Check initial state
    log(f"Before freezing - Main model trainable params: {sum(p.numel() for p in training_setup.model.parameters() if p.requires_grad):,}", priority=0)
    if 'electrode_embeddings' in training_setup.model_components:
        log(f"Before freezing - Electrode embeddings trainable params: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters() if p.requires_grad):,}", priority=0)
    
    # Freeze the main model parameters
    for param in training_setup.model.parameters():
        param.requires_grad = False
    
    # Keep electrode embeddings trainable
    if 'electrode_embeddings' in training_setup.model_components:
        for param in training_setup.model_components['electrode_embeddings'].parameters():
            param.requires_grad = True
        log(f"Frozen main model parameters, keeping electrode embeddings trainable", priority=0)
        log(f"Electrode embeddings parameters: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters()):,}", priority=0)
        
        # Debug: Check final state
        log(f"After freezing - Main model trainable params: {sum(p.numel() for p in training_setup.model.parameters() if p.requires_grad):,}", priority=0)
        log(f"After freezing - Electrode embeddings trainable params: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters() if p.requires_grad):,}", priority=0)
    else:
        log(f"Warning: No electrode_embeddings found in model_components", priority=1)
elif finetune_mode == 'alignment_components':
    log(f"Finetuning mode: Alignment components (freezing main model, keeping electrode embeddings and shared projection unfrozen)", priority=0)
    
    # Debug: Check initial state
    log(f"Before freezing - Main model trainable params: {sum(p.numel() for p in training_setup.model.parameters() if p.requires_grad):,}", priority=0)
    if 'electrode_embeddings' in training_setup.model_components:
        log(f"Before freezing - Electrode embeddings trainable params: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters() if p.requires_grad):,}", priority=0)
    if 'shared_projection' in training_setup.model_components:
        log(f"Before freezing - Shared projection trainable params: {sum(p.numel() for p in training_setup.model_components['shared_projection'].parameters() if p.requires_grad):,}", priority=0)
    
    # Freeze the main model parameters
    for param in training_setup.model.parameters():
        param.requires_grad = False
    
    # Keep electrode embeddings and shared projection trainable
    if 'electrode_embeddings' in training_setup.model_components:
        for param in training_setup.model_components['electrode_embeddings'].parameters():
            param.requires_grad = True
    if 'shared_projection' in training_setup.model_components:
        for param in training_setup.model_components['shared_projection'].parameters():
            param.requires_grad = True
    
    log(f"Frozen main model parameters, keeping electrode embeddings and shared projection trainable", priority=0)
    log(f"Electrode embeddings parameters: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters()):,}", priority=0)
    log(f"Shared projection parameters: {sum(p.numel() for p in training_setup.model_components['shared_projection'].parameters()):,}", priority=0)
    
    # Debug: Check final state
    log(f"After freezing - Main model trainable params: {sum(p.numel() for p in training_setup.model.parameters() if p.requires_grad):,}", priority=0)
    log(f"After freezing - Electrode embeddings trainable params: {sum(p.numel() for p in training_setup.model_components['electrode_embeddings'].parameters() if p.requires_grad):,}", priority=0)
    log(f"After freezing - Shared projection trainable params: {sum(p.numel() for p in training_setup.model_components['shared_projection'].parameters() if p.requires_grad):,}", priority=0)
else:
    log(f"Finetuning mode: Full model", priority=0)
    log(f"All model parameters will be trained", priority=0)

log(f"Loading dataloaders...", priority=0)
training_setup.load_dataloaders()

### LOAD OPTIMIZER AND LEARNING RATE SCHEDULER ###

# Get trainable parameters based on finetune mode
if finetune_mode == 'electrode_embeddings':
    trainable_params = []
    if 'electrode_embeddings' in training_setup.model_components:
        trainable_params.extend(list(training_setup.model_components['electrode_embeddings'].parameters()))
    log(f"Trainable parameters (electrode embeddings only): {sum(p.numel() for p in trainable_params):,}", priority=0)
    
    # Debug: Verify that only electrode embeddings are trainable
    main_model_params = list(training_setup.model.parameters())
    trainable_main_params = [p for p in main_model_params if p.requires_grad]
    log(f"Debug: Main model trainable params in optimizer list: {sum(p.numel() for p in trainable_main_params):,}", priority=0)
    
    # Double-check that no main model params are in trainable_params
    main_model_param_ids = set(id(p) for p in main_model_params)
    trainable_param_ids = set(id(p) for p in trainable_params)
    overlap = main_model_param_ids.intersection(trainable_param_ids)
    if overlap:
        log(f"WARNING: Found {len(overlap)} main model parameters in trainable_params!", priority=1)
    else:
        log(f"✓ Confirmed: No main model parameters in trainable_params", priority=0)
elif finetune_mode == 'alignment_components':
    trainable_params = []
    if 'electrode_embeddings' in training_setup.model_components:
        trainable_params.extend(list(training_setup.model_components['electrode_embeddings'].parameters()))
    if 'shared_projection' in training_setup.model_components:
        trainable_params.extend(list(training_setup.model_components['shared_projection'].parameters()))
    log(f"Trainable parameters (electrode embeddings and shared projection): {sum(p.numel() for p in trainable_params):,}", priority=0)
    
    # Debug: Verify that only electrode embeddings and shared projection are trainable
    main_model_params = list(training_setup.model.parameters())
    trainable_main_params = [p for p in main_model_params if p.requires_grad]
    log(f"Debug: Main model trainable params in optimizer list: {sum(p.numel() for p in trainable_main_params):,}", priority=0)
    
    # Double-check that no main model params are in trainable_params
    main_model_param_ids = set(id(p) for p in main_model_params)
    trainable_param_ids = set(id(p) for p in trainable_params)
    overlap = main_model_param_ids.intersection(trainable_param_ids)
    if overlap:
        log(f"WARNING: Found {len(overlap)} main model parameters in trainable_params!", priority=1)
    else:
        log(f"✓ Confirmed: No main model parameters in trainable_params", priority=0)
else:
    trainable_params = training_setup.model_parameters(verbose=True)

optimizers = []
if config['training']['optimizer'] == 'Muon': # Muon is like the newest and coolest optimizer that works better than Adam
    # Muon only supports matrix parameters, so we use adam for the other parameters
    matrix_params = [p for p in trainable_params if p.ndim == 2] 
    other_params = [p for p in trainable_params if p.ndim != 2]
    optimizers.append(Muon(matrix_params, lr=config['training']['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=config['training']['weight_decay']))
    if len(other_params) > 0:
        optimizers.append(torch.optim.AdamW(other_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95)))
else:
    optimizers = [torch.optim.AdamW(trainable_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95))]

schedulers = [] # Learning rate scheduling (warmup and falloff, both optional)
for optimizer in optimizers:
    total_steps = config['training']['n_epochs'] * len(training_setup.train_dataloader)
    warmup = (torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=config['training']['warmup_steps']) if config['training']['warmup_steps'] > 0
             else None)
    main = (torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-5, total_iters=total_steps) if config['training']['lr_schedule'] == 'linear' 
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) if config['training']['lr_schedule'] == 'cosine'
        else None)
    if warmup is not None and main is not None: schedulers.append(ChainedScheduler([warmup, main]))
    elif warmup is not None: schedulers.append(warmup)
    elif main is not None: schedulers.append(main)

# Below for all the tasks in Neuroprobe
# eval_tasks = ['frame_brightness', 'global_flow', 'local_flow', 'global_flow_angle', 'local_flow_angle', 'face_num', 'volume', 'pitch', 'delta_volume', 
#               'delta_pitch', 'speech', 'onset', 'gpt2_surprisal', 'word_length', 'word_gap', 'word_index', 'word_head_pos', 'word_part_speech', 'speaker']
eval_tasks = config['training']['eval_tasks'].split(',')
# Filter out empty strings
eval_tasks = [task for task in eval_tasks if task.strip()]

if eval_tasks:  # Only create evaluation if there are tasks
    evaluation = FrozenModelEvaluation_SS_SM(
        # model evaluation function
        model_preprocess_functions=training_setup.get_preprocess_functions(pretraining=False),
        model_evaluation_function=training_setup.generate_frozen_features,
        eval_aggregation_method=config['cluster']['eval_aggregation_method'],
        # benchmark parameters 
        eval_names=eval_tasks, lite=True,
        subject_trials=[(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in config['training']['eval_subject_trials']],
        # dataloader parameters
        device=device,
        dtype=config['training']['data_dtype'],
        batch_size=config['training']['batch_size'],
        num_workers_eval=config['cluster']['num_workers_eval'],
        prefetch_factor=config['cluster']['prefetch_factor'],
    )
else:
    evaluation = None

### WANDB SETUP ###

if wandb: 
    log(f"Initializing wandb with project: {config['cluster']['wandb_project']}, name: {config['cluster']['wandb_name']}, entity: {config['cluster']['wandb_entity']}", priority=0)
    os.makedirs("runs/wandb", exist_ok=True)
    # Create a unique wandb run ID using the random_string
    unique_run_id = f"{config['cluster']['wandb_name']}_{config['training']['random_string']}"
    log(f"Using unique wandb run ID: {unique_run_id}", priority=0)
    
    # added option for entity since you might be inside the andrii-mit organization
    if len(config['cluster']['wandb_entity']) > 0:
        wandb.init(project=config['cluster']['wandb_project'], name=config['cluster']['wandb_name'], id=unique_run_id,
                    entity=config['cluster']['wandb_entity'],
                    config=config, settings=wandb.Settings(init_timeout=1000), dir="runs/wandb")
    else:
        wandb.init(project=config['cluster']['wandb_project'], name=config['cluster']['wandb_name'], id=unique_run_id,
                    config=config, settings=wandb.Settings(init_timeout=1000), dir="runs/wandb")
    log(f"Wandb run initialized successfully", priority=0)
else:
    log(f"Wandb disabled - project: '{config['cluster']['wandb_project']}', length: {len(config['cluster']['wandb_project'])}", priority=0)

### EVALUATION OF THE MODEL BEFORE TRAINING ###

eval_results = {}
if config['cluster']['eval_at_beginning'] and evaluation is not None:
    log(f"Evaluating model...", priority=0)
    training_setup.eval_mode()
    # 
    eval_raw = evaluation.evaluate_on_all_metrics(quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average', raw_data=True, key_prefix="raw_")
    eval_results.update(eval_raw)
    log(f"eval_raw: {eval_raw}", priority=0)
    #
    eval_full_model = evaluation.evaluate_on_all_metrics(quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average')
    log(f"eval_full_model: {eval_full_model}", priority=0)
    eval_results.update(eval_full_model)
    #
    if wandb: wandb.log(eval_results, step=1)
    del eval_full_model, eval_raw
    torch.cuda.empty_cache()
    gc.collect()

### TRAINING ###

training_setup.save_model(epoch=0)

training_statistics_store = []
for epoch_i in range(config['training']['n_epochs']):
    epoch_start_time = time.time()
    training_setup.train_mode()

    # Main training loop
    epoch_losses = {}
    for batch_idx, batch in enumerate(training_setup.train_dataloader):
        subject_identifier, trial_id = batch['subject_trial'][0]

        for optimizer in optimizers: optimizer.zero_grad()

        # Use autocast with specified dtype
        with autocast(device_type='cuda', dtype=config['model']['amp_dtype'], enabled=config['model']['use_mixed_precision']):
            loss_dict = training_setup.calculate_pretrain_loss(batch)
            loss = sum([v for k, v in loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in loss_dict.items() if 'accuracy' not in k]) 
        loss.backward()

        for key, _loss in loss_dict.items():
            epoch_losses[key] = epoch_losses.get(key, 0) + _loss.item()

        for optimizer in optimizers:
            optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

        training_statistics_store.append({
            'epoch': epoch_i+1,
            'batch': batch_idx+1,
            'subject_identifier': subject_identifier,
            'trial_id': trial_id,
            'batch_loss': loss.item(),
            'timestamp': time.time(),
            **{f"batch_{k}": v.item() for k, v in loss_dict.items()}
        })

        if batch_idx % 1 == 0:
            losses_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in loss_dict.items() if 'accuracy' not in k])
            log(f"Epoch {epoch_i+1}/{config['training']['n_epochs']}, " + \
                f"Batch {batch_idx+1}/{len(training_setup.train_dataloader)} ({subject_identifier}_{trial_id}), " + \
                f"LR: {optimizers[0].param_groups[0]['lr']:.6f}, " + \
                f"Loss: {loss.item():.4f} ({losses_string}), " + \
                (f"Temp {torch.exp(training_setup.model.temperature_param).item():.4f}" if hasattr(training_setup.model, 'temperature_param') else ""), priority=0)
        
        if batch_idx % 20 == 0: # Clear cache every 20 batches
            del loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()
        
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(training_setup.train_dataloader)

    # Evaluate the model
    training_setup.eval_mode()
    eval_results = {f"train_{k}": v for k, v in epoch_losses.items()}
    eval_results['train_loss'] = sum([v for k, v in epoch_losses.items() if 'accuracy' not in k]) / len([v for k, v in epoch_losses.items() if 'accuracy' not in k])
    with torch.no_grad():
        test_loss_dict = training_setup.calculate_pretrain_test_loss()
        eval_results.update({f"test_{k}": v.item() for k, v in test_loss_dict.items()})
        # Check if there are any non-accuracy losses in the test_loss_dict
        non_accuracy_losses = [v.item() for k, v in test_loss_dict.items() if 'accuracy' not in k]
        if len(non_accuracy_losses) > 0:
            eval_results['test_loss'] = sum(non_accuracy_losses) / len(non_accuracy_losses)
        else:
            # If no test data or only accuracy metrics, set test_loss to a default value
            log("Warning: No test data available or test dataloader is empty. Setting test_loss to inf.", priority=1)
            eval_results['test_loss'] = float('inf')  # or 0.0, depending on your preference
        losses_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' not in k])
        accuracy_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' in k])
        log(f"Test loss: {eval_results['test_loss']:.4f} ({losses_string}), Accuracies: {accuracy_string}", priority=0)
        if (epoch_i+1) % config['cluster']['eval_model_every_n_epochs'] == 0 and evaluation is not None:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average')
            eval_results.update(evaluation_results_strings)
            log("eval_full_model" + str(evaluation_results_strings))
            del evaluation_results_strings
            torch.cuda.empty_cache()
            gc.collect()
        time_remaining = (time.time() - epoch_start_time) * (config['training']['n_epochs'] - (epoch_i + 1))
        days = int(time_remaining // (24 * 3600))
        log(f"Epoch {epoch_i+1}/{config['training']['n_epochs']}, Estimated time remaining: {days}d, {time.strftime('%H:%M:%S', time.gmtime(time_remaining % (24 * 3600)))}", priority=0)
    if wandb: wandb.log(eval_results, step=epoch_i+2) # XXX adding step=1 to the first log
    training_statistics_store[-1].update(eval_results)

    # Save the model every N steps or at the end of the training
    if (epoch_i+1) % config['cluster']['save_model_every_n_epochs'] == 0 or epoch_i+1 == config['training']['n_epochs']:
        training_setup.save_model(epoch=epoch_i+1, eval_results=eval_results, training_statistics_store=training_statistics_store)

if wandb: wandb.finish()