import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast
import gc

from utils.muon_optimizer import Muon
from subject.dataset import load_subjects
from evaluation.neuroprobe_tasks import FrozenModelEvaluation_SS_SM
from training_setup.training_config import log, update_dir_name, update_random_seed, parse_config_from_args, get_default_config, parse_subject_trials_from_config
from torch.optim.lr_scheduler import ChainedScheduler

### LOADING CONFIGS ###

config = get_default_config(random_string="TEMP", wandb_project="") # Outputs a dictionary, see utils/training_config.py for how it looks like
parse_config_from_args(config) # Parses the command line arguments and updates the config dictionary
parse_subject_trials_from_config(config) # Parses the subject trials from the config dictionary

dir_name = update_dir_name(config) # This is used to name the directory where the model is saved, based on the training parameters (the config)
update_random_seed(config) # This is used to set the random seed for the model (for reproducibility)
log(f"Directory name: {dir_name}", priority=0)

# Weights & Biases dashboard setup
config['cluster']['wandb_name'] = config['cluster']['dir_name'] # This is used to name the wandb run
if len(config['cluster']['wandb_project'])==0: wandb = False # If wandb project is not set, we do not use wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['device'] = device
log(f"Using device: {device}", priority=0)

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
training_setup_dir = os.path.join('runs/data', dir_name, 'training_setup')
os.makedirs(training_setup_dir, exist_ok=True)
shutil.copy2(setup_file, training_setup_dir)    

### LOAD MODEL ###

log(f"Loading model...", priority=0)
training_setup.initialize_model()

log(f"Loading dataloaders...", priority=0)
training_setup.load_dataloaders()

### LOAD OPTIMIZER AND LEARNING RATE SCHEDULER ###

all_params = training_setup.model_parameters(verbose=True)

optimizers = []
if config['training']['optimizer'] == 'Muon': # Muon is like the newest and coolest optimizer that works better than Adam
    # Muon only supports matrix parameters, so we use adam for the other parameters
    matrix_params = [p for p in all_params if p.ndim == 2] 
    other_params = [p for p in all_params if p.ndim != 2]
    optimizers.append(Muon(matrix_params, lr=config['training']['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=config['training']['weight_decay']))
    if len(other_params) > 0:
        optimizers.append(torch.optim.AdamW(other_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95)))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95))]

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
evaluation = FrozenModelEvaluation_SS_SM(
    # model evaluation function
    model_preprocess_functions=training_setup.get_preprocess_functions(pretraining=False),
    model_evaluation_function=training_setup.generate_frozen_features,
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

### WANDB SETUP ###

if wandb: 
    os.makedirs("runs/wandb", exist_ok=True)
    # added option for entity since you might be inside the andrii-mit organization
    if len(config['cluster']['wandb_entity']) > 0:
        wandb.init(project=config['cluster']['wandb_project'], name=config['cluster']['wandb_name'], id=config['cluster']['wandb_name'],
                    entity=config['cluster']['wandb_entity'],
                    config=config, settings=wandb.Settings(init_timeout=1000), dir="runs/wandb")
    else:
        wandb.init(project=config['cluster']['wandb_project'], name=config['cluster']['wandb_name'], id=config['cluster']['wandb_name'],
                    config=config, settings=wandb.Settings(init_timeout=1000), dir="runs/wandb")

### EVALUATION OF THE MODEL BEFORE TRAINING ###

eval_results = {}
if config['cluster']['eval_at_beginning']:
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
        eval_results['test_loss'] = sum([v.item() for k, v in test_loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in test_loss_dict.items() if 'accuracy' not in k])
        losses_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' not in k])
        accuracy_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' in k])
        log(f"Test loss: {eval_results['test_loss']:.4f} ({losses_string}), Accuracies: {accuracy_string}", priority=0)
        if (epoch_i+1) % config['cluster']['eval_model_every_n_epochs'] == 0:
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