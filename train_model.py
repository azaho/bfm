import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast, GradScaler, autocast_mode
import gc

from muon import Muon
from model_model import FFTaker, OriginalModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from dataset import load_dataloaders, load_subjects
from evaluation_neuroprobe import FrozenModelEvaluation_SS_SM
from train_utils import log, update_dir_name, update_random_seed, convert_dtypes, parse_config_from_args, get_default_config, get_shared_memory_info, parse_subject_trials_from_config
from torch.optim.lr_scheduler import ChainedScheduler

### LOADING CONFIGS ###

config = get_default_config(random_string="TEMP", wandb_project="")
parse_config_from_args(config)
parse_subject_trials_from_config(config)

dir_name = update_dir_name(config)
update_random_seed(config)
config['cluster']['wandb_name'] = config['cluster']['dir_name']
log(f"Directory name: {dir_name}", priority=0)

wandb = len(config['cluster']['wandb_project']) > 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}", priority=0)

### LOAD SUBJECTS ###

log(f"Loading subjects...", priority=0)
all_subjects = load_subjects(config['training']['train_subject_trials'], config['training']['eval_subject_trials'], config['training']['data_dtype'], 
                             cache=config['cluster']['cache_subjects'], allow_corrupted=False)

### LOAD MODEL ###

assert config['model']['electrode_embedding']['spectrogram'] == True, "For the moment, we only support spectrogram"
assert config['model']['electrode_embedding']['spectrogram_power'] == True, "For the moment, we only support spectrogram power"

model = OriginalModel(
    d_model=config['model']['transformer']['d_model'],
    n_layers_electrode=config['model']['transformer']['n_layers_electrode'],
    n_layers_time=config['model']['transformer']['n_layers_time'],
    n_heads=config['model']['transformer']['n_heads'],
    dropout=config['model']['transformer']['dropout']
).to(device, dtype=config['model']['dtype'])

### LOAD ELECTRODE EMBEDDINGS ###

electrode_embeddings = {
    'learned': ElectrodeEmbedding_Learned,
    'zero': ElectrodeEmbedding_Zero,
    'coordinate_init': ElectrodeEmbedding_Learned_CoordinateInit,
    'noisy_coordinate': ElectrodeEmbedding_NoisyCoordinate,
}[config['model']['electrode_embedding']['type']](
    config['model']['transformer']['d_model'], 
    embedding_dim=config['model']['electrode_embedding']['embedding_dim'],
    coordinate_noise_std=config['model']['electrode_embedding']['coordinate_noise_std'],
).to(device, dtype=config['model']['dtype'])

for subject in all_subjects.values():
    log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
    electrode_embeddings.add_subject(subject)
electrode_embeddings = electrode_embeddings.to(device, dtype=config['model']['dtype']) # moving to device again to ensure the new parameters are on the correct device

### LOAD DATALOADERS ###

log(f"Loading dataloaders...", priority=0)
train_dataloader, test_dataloader = load_dataloaders(
    all_subjects, config['training']['train_subject_trials'], config['training']['p_test'], 
    config['model']['context_length'], config['training']['data_dtype'], 
    config['training']['batch_size'],
    num_workers_dataloaders=config['cluster']['num_workers_dataloaders'], 
    prefetch_factor=config['cluster']['prefetch_factor'],
    max_n_electrodes=config['model']['max_n_electrodes'],
    output_embeddings_map=electrode_embeddings.embeddings_map
)

### LOAD EVALUATION ###

eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in config['training']['eval_subject_trials']]
# Below for all the tasks in Neuroprobe
eval_tasks = ['frame_brightness', 'global_flow', 'local_flow', 'global_flow_angle', 'local_flow_angle', 'face_num', 'volume', 'pitch', 'delta_volume', 
              'delta_pitch', 'speech', 'onset', 'gpt2_surprisal', 'word_length', 'word_gap', 'word_index', 'word_head_pos', 'word_part_speech', 'speaker']
# Below for just two tasks in Neuroprobe (minimal eval)
eval_tasks = ['gpt2_surprisal', 'speech']
evaluation = FrozenModelEvaluation_SS_SM(
    eval_tasks, eval_subject_trials, 
    config['training']['data_dtype'], config['training']['batch_size'], # Can have a bigger batch size here if that speeds things up
    electrode_embeddings.embeddings_map,
    num_workers_eval=config['cluster']['num_workers_eval'],
    prefetch_factor=config['cluster']['prefetch_factor'],
    feature_aggregation_method=config['cluster']['eval_aggregation_method'],
    max_n_electrodes=config['model']['max_n_electrodes'],
    laplacian_rereference=config['model']['laplacian_rereference'],
    voltage_normalization=False
)

### LOAD OPTIMIZER AND LEARNING RATE SCHEDULER ###

all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
n_model_params = sum(p.numel() for p in model.parameters())
n_embed_params = sum(p.numel() for p in electrode_embeddings.parameters())
log(f"Model parameters: {n_model_params:,}", priority=0)
log(f"Embedding parameters: {n_embed_params:,}", priority=0)
log(f"Total parameters: {n_model_params + n_embed_params:,}", priority=0)
config['model']['n_params'] = {'model': n_model_params, 'embeddings': n_embed_params, 'total': n_model_params + n_embed_params}

optimizers = []
if config['training']['optimizer'] == 'Muon':
    # Muon only supports matrix parameters, so we use adam for the other parameters
    matrix_params = [p for p in all_params if p.ndim == 2] 
    other_params = [p for p in all_params if p.ndim != 2]
    optimizers.append(Muon(matrix_params, lr=config['training']['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=config['training']['weight_decay']))
    if len(other_params) > 0:
        optimizers.append(torch.optim.AdamW(other_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95)))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.95))]

schedulers = []
for optimizer in optimizers:
    total_steps = config['training']['n_epochs'] * len(train_dataloader)
    warmup = (torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=config['training']['warmup_steps']) if config['training']['warmup_steps'] > 0
             else None)
    main = (torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-5, total_iters=total_steps) if config['training']['lr_schedule'] == 'linear' 
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) if config['training']['lr_schedule'] == 'cosine'
        else None)
    if warmup is not None and main is not None: schedulers.append(ChainedScheduler([warmup, main]))
    elif warmup is not None: schedulers.append(warmup)
    elif main is not None: schedulers.append(main)

### LOSS FUNCTION CALCULATION ###

def calculate_loss_function(batch, output_accuracy=True):
    # batch['data'] shape: (batch_size, n_electrodes, n_timebins)
    # batch['electrode_index'] shape: (batch_size, n_electrodes)
    losses = {}
    def _add_to_loss_contrastive(output, target, loss_suffix):
        # output and target shape: (batch_size, n_timebins-future_bin_idx, d_model)
        if config['training']['normalize_features']:
            output_ = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
            target_ = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)
        similarity = output_.permute(1, 0, 2) @ target_.permute(1, 2, 0) # shape: (n_timebins-future_bin_idx, batch_size, batch_size)
        if config['training']['use_temperature_param']:
            similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(config['training']['max_temperature_param'], device=model.device, dtype=model.dtype))
        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(output.shape[1], 1).to(model.device, dtype=torch.long).reshape(-1)

        loss = torch.nn.functional.cross_entropy(similarity[:, :, :].view(-1, batch_size), expanded_arange)
        losses[f'contrastive_{loss_suffix}'] = loss
        if output_accuracy:
            accuracy_bin = (similarity[:, :, :].view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
            losses[f'accuracy_{loss_suffix}'] = accuracy_bin
        return losses
    future_bin_idx = config['training']['future_bin_idx']

    # Note that due to the RandomELectrodeCollator in the dataset class, the electrodes are already shuffled and cut to max_n_electrodes
    batch_size, n_electrodes, n_samples = batch['data'].shape
    
    if config['model']['laplacian_rereference']:
        from laplacian_rereferencing import laplacian_rereference_batch
        batch = laplacian_rereference_batch(batch, remove_non_laplacian=False)

    electrode_data_a = batch['data'][:, :n_electrodes//2, :]
    embeddings_a = electrode_embeddings(batch['electrode_index'][:, :n_electrodes//2])
    electrode_data_b = batch['data'][:, n_electrodes//2:, :]
    embeddings_b = electrode_embeddings(batch['electrode_index'][:, n_electrodes//2:])

    electrode_transformed_data_a, time_transformed_data_a = model(electrode_data_a, embeddings_a) # shape: (batch_size, n_timebins, d_model)
    electrode_transformed_data_b, time_transformed_data_b = model(electrode_data_b, embeddings_b) # shape: (batch_size, n_timebins, d_model)

    # add two symmetric loss components
    _add_to_loss_contrastive(time_transformed_data_a[:, :-future_bin_idx], electrode_transformed_data_b[:, future_bin_idx:], 'a')
    _add_to_loss_contrastive(time_transformed_data_b[:, :-future_bin_idx], electrode_transformed_data_a[:, future_bin_idx:], 'b')

    return losses

def calculate_pretrain_test_loss():
    losses = {}
    n_batches = 0
    for batch in test_dataloader:
        batch['data'] = batch['data'].to(model.device, dtype=config['model']['dtype'], non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(model.device, dtype=torch.long, non_blocking=True)

        loss = calculate_loss_function(batch)
        
        for key, value in loss.items():
            if key not in losses: losses[key] = 0
            losses[key] += value
        n_batches += 1
    return {k: v / n_batches for k, v in losses.items()}

### PREPARATION FOR TRAINING ###

log(f"Evaluating model...", priority=0)
eval_results = {}
model.eval()
electrode_embeddings.eval()
# 
eval_raw = evaluation.evaluate_on_all_metrics(model, electrode_embeddings, quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average', raw_data=True, key_prefix="raw_")
eval_results.update(eval_raw)
print("eval_raw", eval_raw)
#
eval_full_model = evaluation.evaluate_on_all_metrics(model, electrode_embeddings, quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average')
print("eval_full_model", eval_full_model)
eval_results.update(eval_full_model)
#
if wandb: wandb.log(eval_results, step=1)
del eval_full_model, eval_raw
torch.cuda.empty_cache()
gc.collect()
# eval_results = {}

if wandb: 
    wandb.init(project=config['cluster']['wandb_project'], name=config['cluster']['wandb_name'], id=config['cluster']['wandb_name'],
    config=config, settings=wandb.Settings(init_timeout=480))

def save_model(eval_results, epoch):
    model_path = f"models_data/{config['cluster']['dir_name']}/model_epoch_{epoch}.pth"
    os.makedirs(f"models_data/{config['cluster']['dir_name']}", exist_ok=True)
    torch.save({
        'eval_results': eval_results,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
        'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
        'config': convert_dtypes(config),
    }, model_path)
save_model(eval_results, 0)

### TRAINING ###
# The data starts out as (batch_size, n_electrodes, n_timesamples)
# 1. (batch_size, n_electrodes, n_timesamples) -> FFT + linear layer -> (batch_size, n_electrodes, n_timebins, d_model)
# 2. (batch_size, n_electrodes, n_timebins, d_model) -> electrode transformer -> (batch_size, n_timebins, d_model)
# 3. (batch_size, n_electrodes, n_timebins, sample_timebin_size) -> time transformer -> (batch_size, n_timebins, d_model)
# loss function: compare the output of the time transformer to the input of the time transformer

training_statistics_store = []
for epoch_i in range(config['training']['n_epochs']):
    epoch_start_time = time.time()

    model.train()
    electrode_embeddings.train()

    # Main training loop
    epoch_losses = {}
    for batch_idx, batch in enumerate(train_dataloader):
        batch['data'] = batch['data'].to(device, dtype=config['model']['dtype'], non_blocking=True) # (batch_size, n_electrodes, n_timesamples)
        batch['electrode_index'] = batch['electrode_index'].to(device, dtype=torch.long, non_blocking=True)
        subject_identifier, trial_id = batch['subject_trial'][0]

        for optimizer in optimizers: optimizer.zero_grad()

        # Use autocast with specified dtype
        if config['model']['use_mixed_precision']:
            with autocast(device_type='cuda', dtype=config['model']['amp_dtype'], enabled=config['model']['use_mixed_precision']):
                loss_dict = calculate_loss_function(batch)
                loss = sum([v for k, v in loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in loss_dict.items() if 'accuracy' not in k]) 
            loss.backward()
        else:
            loss_dict = calculate_loss_function(batch)
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
            log(f"Epoch {epoch_i+1}/{config['training']['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), LR: {optimizers[0].param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f} ({losses_string}), Temp {torch.exp(model.temperature_param).item():.4f}", priority=0)
        
        if batch_idx % 20 == 0: # Clear cache every 20 batches
            del loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()
        
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    electrode_embeddings.eval()
    eval_results = {f"train_{k}": v for k, v in epoch_losses.items()}
    eval_results['train_loss'] = sum([v for k, v in epoch_losses.items() if 'accuracy' not in k]) / len([v for k, v in epoch_losses.items() if 'accuracy' not in k])
    with torch.no_grad():
        test_loss_dict = calculate_pretrain_test_loss()
        eval_results.update({f"test_{k}": v.item() for k, v in test_loss_dict.items()})
        eval_results['test_loss'] = sum([v.item() for k, v in test_loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in test_loss_dict.items() if 'accuracy' not in k])
        losses_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' not in k])
        accuracy_string = f" / ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in test_loss_dict.items() if 'accuracy' in k])
        log(f"Test loss: {eval_results['test_loss']:.4f} ({losses_string}), Accuracies: {accuracy_string}", priority=0)
        if (epoch_i+1) % config['cluster']['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, electrode_embeddings, quick_eval=config['cluster']['quick_eval'], only_keys_containing='auroc/average')
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

    # Save the model
    if (epoch_i+1) % config['cluster']['save_model_every_n_epochs'] == 0 or epoch_i+1 == config['training']['n_epochs']:
        save_model(eval_results, epoch_i+1)
        statistics_path = f"models_data/{config['cluster']['dir_name']}/training_statistics.json"
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()