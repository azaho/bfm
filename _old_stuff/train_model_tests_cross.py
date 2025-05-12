import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast, GradScaler, autocast_mode

from muon import Muon
from model_model import GranularModel, BinTransformer, CrossModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit
from dataset import load_dataloaders, load_subjects
from evaluation_btbench import FrozenModelEvaluation_SS_SM
from train_utils import log, update_dir_name, update_random_seed, convert_dtypes, parse_configs_from_args, get_default_configs, get_shared_memory_info

training_config, model_config, cluster_config = get_default_configs(random_string="TEMP", wandb_project="")
parse_configs_from_args(training_config, model_config, cluster_config)
dir_name = update_dir_name(model_config, training_config, cluster_config)
update_random_seed(training_config)
cluster_config['wandb_name'] = cluster_config['dir_name']
log(f"Directory name: {dir_name}", priority=0)

if len(cluster_config['wandb_project'])==0: wandb = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}", priority=0)

log(f"Loading subjects...", priority=0)
all_subjects = load_subjects(training_config['train_subject_trials'], training_config['eval_subject_trials'], training_config['data_dtype'], 
                             cache=cluster_config['cache_subjects'], allow_corrupted=False)

n_downsample_factor = 1
assert n_downsample_factor == 1, "n_downsample_factor must be 1, not supported in the CrossModel class yet."

log(f"Loading model...", priority=0)
bin_transformer = BinTransformer(
    first_kernel=int(model_config['sample_timebin_size']*2048)//16, 
    d_model=192,#model_config['transformer']['d_model'],
    n_layers=4,
    n_heads=8,
    overall_sampling_rate=2048,
    sample_timebin_size=model_config['sample_timebin_size'],
    n_downsample_factor=n_downsample_factor
).to(device, dtype=model_config['dtype'])

model = GranularModel(
    int(model_config['sample_timebin_size'] * 2048 // n_downsample_factor),
    model_config['transformer']['d_model'],  
    n_layers=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads'],
).to(device, dtype=model_config['dtype'])

cross_model = CrossModel(
    int(model_config['sample_timebin_size'] * 2048 // n_downsample_factor),
    model_config['transformer']['d_model'],
    n_layers=model_config['transformer']['n_layers_electrode'],
    n_heads=model_config['transformer']['n_heads'],
).to(device, dtype=model_config['dtype'])

if model_config['electrode_embedding']['type'] == 'learned' or model_config['electrode_embedding']['type'] == 'zero':
    electrode_embeddings = ElectrodeEmbedding_Learned(
        model_config['transformer']['d_model'], 
        embedding_dim=model_config['electrode_embedding']['embedding_dim'],
        embedding_requires_grad=model_config['electrode_embedding']['type'] != 'zero'
    )
elif model_config['electrode_embedding']['type'] == 'coordinate_init':
    electrode_embeddings = ElectrodeEmbedding_Learned_CoordinateInit(
        model_config['transformer']['d_model'], 
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
elif model_config['electrode_embedding']['type'] == 'noisy_coordinate':
    electrode_embeddings = ElectrodeEmbedding_NoisyCoordinate(
        model_config['transformer']['d_model'], 
        coordinate_noise_std=model_config['electrode_embedding']['coordinate_noise_std'],
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
else:
    raise ValueError(f"Invalid electrode embedding type: {model_config['electrode_embedding']['type']}")
electrode_embeddings = electrode_embeddings.to(device, dtype=model_config['dtype'])

# if model_config['electrode_embedding']['spectrogram']:
#     electrode_data_embeddings = ElectrodeDataEmbeddingFFT(
#         electrode_embeddings, model_config['sample_timebin_size'], model_config['transformer']['d_model'], 
#         max_frequency_bin=model_config['max_frequency_bin']
#     ).to(device, dtype=model_config['dtype'])
# else:
#     electrode_data_embeddings = ElectrodeDataEmbedding(
#         electrode_embeddings, model_config['sample_timebin_size'], model_config['transformer']['d_model'], 
#         overall_sampling_rate=next(iter(all_subjects.values())).get_sampling_rate(0) # XXX remove this once figured out how to be flexible here regarding the sampling rate
#     ).to(device, dtype=model_config['dtype'])

from btbench.btbench_config import BTBENCH_LITE_ELECTRODES
np.random.seed(training_config['random_seed'])
for subject_identifier, subject in all_subjects.items():
    consider_electrode_names = list(BTBENCH_LITE_ELECTRODES[subject_identifier])
    electrode_subset = list(np.random.choice(consider_electrode_names, size=training_config['n_electrodes_subset'], replace=False))
    subject.set_electrode_subset(electrode_subset)

eval_electrode_subset = {
    #'btbank3': ['T1cIe11'],
}

for subject in all_subjects.values():
    log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
    this_subject_trials = [trial_id for (sub_id, trial_id) in training_config['train_subject_trials'] if sub_id == subject.subject_identifier]
    electrode_embeddings.add_subject(subject)
electrode_embeddings = electrode_embeddings.to(device, dtype=model_config['dtype']) # moving to device again to ensure the new parameters are on the correct device

log(f"Loading dataloaders...", priority=0)
n_samples = model_config['max_n_timebins'] * model_config['sample_timebin_size']
train_dataloader, test_dataloader = load_dataloaders(
    all_subjects, training_config['train_subject_trials'], training_config['p_test'], 
    model_config['sample_timebin_size'], model_config['max_n_timebins'], training_config['data_dtype'], 
    training_config['batch_size'],
    num_workers_dataloaders=cluster_config['num_workers_dataloaders'], 
    prefetch_factor=cluster_config['prefetch_factor'],
    max_n_electrodes=model_config['max_n_electrodes'],
    output_embeddings_map=electrode_embeddings.embeddings_map
)

eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in training_config['eval_subject_trials']]
eval_tasks = ['onset', 'gpt2_surprisal', 'volume', 'word_part_speech', 'pitch', 'speech']
evaluation = FrozenModelEvaluation_SS_SM(
    eval_tasks, eval_subject_trials, 
    training_config['data_dtype'], training_config['batch_size'], # Can have a bigger batch size here if that speeds things up
    electrode_embeddings.embeddings_map,
    num_workers_eval=cluster_config['num_workers_eval'],
    prefetch_factor=cluster_config['prefetch_factor'],
    feature_aggregation_method=cluster_config['eval_aggregation_method'],
    electrode_subset=eval_electrode_subset
)

# After model initialization
amp_dtype = model_config['amp_dtype']
use_amp = model_config['use_mixed_precision'] and torch.cuda.is_available()
scaler = None  # No scaler needed for bfloat16

all_params = list(model.parameters()) + list(bin_transformer.parameters()) + list(electrode_embeddings.parameters()) + list(cross_model.parameters())
# filter to only include parameters that require grad
#all_params = [p for p in all_params if p.requires_grad]
n_model_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in bin_transformer.parameters()) + sum(p.numel() for p in cross_model.parameters())
n_embed_params = sum(p.numel() for p in electrode_embeddings.parameters())
log(f"Model parameters: {n_model_params:,}", priority=0)
log(f"Embedding parameters: {n_embed_params:,}", priority=0)
log(f"Total parameters: {n_model_params + n_embed_params:,}", priority=0)
model_config['n_params'] = {
    'model': n_model_params,
    'embeddings': n_embed_params,
    'total': n_model_params + n_embed_params
}

optimizers = []
if training_config['optimizer'] == 'Muon':
    #all_params = list(model.parameters())
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]
    #other_params += list(electrode_data_embeddings.parameters()) # use adam for electrode data embeddings

    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
    optimizers.append(torch.optim.AdamW(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'], betas=(0.9, 0.95)))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'], betas=(0.9, 0.95))]

schedulers = []
if training_config['lr_schedule'] == 'linear':
    for optimizer in optimizers:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=training_config['n_epochs'] * len(train_dataloader)))
elif training_config['lr_schedule'] == 'cosine':
    for optimizer in optimizers:
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['n_epochs'] * len(train_dataloader)))

def calculate_loss_function(batch, output_accuracy=True):
    # batch['data'] shape: (batch_size, n_electrodes, n_timebins)
    # batch['electrode_index'] shape: (batch_size, n_electrodes)
    losses = {}

    # Randomly select a subset of electrodes
    batch_size, n_electrodes, n_samples = batch['data'].shape
    random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']]
    n_electrodes = min(model_config['max_n_electrodes'], n_electrodes)
    batch['data'] = batch['data'][:, random_electrodes, :]
    batch['electrode_index'] = batch['electrode_index'][:, random_electrodes]

    electrodes_a = random_electrodes[:n_electrodes//2]
    electrodes_b = random_electrodes[n_electrodes//2:]

    embeddings = electrode_embeddings.forward(batch['electrode_index']) # shape: (batch_size, n_electrodes, d_model)
    embeddings_a = embeddings[:, electrodes_a, :] # shape: (batch_size, n_electrodes, d_model)
    embeddings_b = embeddings[:, electrodes_b, :] # shape: (batch_size, n_electrodes, d_model)

    bin_transformed_data_a = bin_transformer(batch['data'][:, electrodes_a, :]) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
    bin_transformed_data_b = bin_transformer(batch['data'][:, electrodes_b, :]) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
    n_timebins = bin_transformed_data_a.shape[2]


    output_a = model(bin_transformed_data_a[:, :, :-1, :], embeddings_a) # shape: (batch_size, n_electrodes, n_timebins-1, sample_timebin_size*SR//n_downsample_factor)
    target_a = bin_transformed_data_a[:, :, 1:, :] # shape: (batch_size, n_electrodes, n_timebins-1, sample_timebin_size*SR//n_downsample_factor)

    if training_config['normalize_features']:
        output_a = output_a / (torch.norm(output_a, dim=-1, keepdim=True) + 0.001)
        target_a = target_a / (torch.norm(target_a, dim=-1, keepdim=True) + 0.001)

    similarity_a = output_a.permute(1, 2, 0, 3) @ target_a.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
    if training_config['use_temperature_param']:
        similarity_a = similarity_a * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))

    expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(len(electrodes_a), n_timebins-1, 1).to(model.device, dtype=torch.long).reshape(-1)
    loss_a = torch.nn.functional.cross_entropy(similarity_a.view(-1, batch_size), expanded_arange)
    losses['contrastive_t'] = loss_a
    if output_accuracy:
        accuracy_a = (similarity_a.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
        losses['accuracy_t'] = accuracy_a



    output_b = cross_model(embeddings_b, output_a, positions_offset=1) # shape: (batch_size, n_electrodes, n_timebins-1, sample_timebin_size*SR//n_downsample_factor)
    target_b = bin_transformed_data_b[:, :, 1:, :] # shape: (batch_size, n_electrodes, n_timebins-1, sample_timebin_size*SR//n_downsample_factor)

    if training_config['normalize_features']:
        output_b = output_b / (torch.norm(output_b, dim=-1, keepdim=True) + 0.001)
        target_b = target_b / (torch.norm(target_b, dim=-1, keepdim=True) + 0.001)

    similarity_b = output_b.permute(1, 2, 0, 3) @ target_b.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
    if training_config['use_temperature_param']:
        similarity_b = similarity_b * torch.minimum(torch.exp(cross_model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))

    expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(len(electrodes_b), n_timebins-1, 1).to(model.device, dtype=torch.long).reshape(-1)
    loss_b = torch.nn.functional.cross_entropy(similarity_b.view(-1, batch_size), expanded_arange)
    losses['contrastive_x'] = loss_b

    if output_accuracy:
        accuracy_b = (similarity_b.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
        losses['accuracy_x'] = accuracy_b

    if "NX" in training_config['random_string']: #XXX
        losses['contrastive_x'] = losses['contrastive_t']
        if output_accuracy:
            losses['accuracy_x'] = losses['accuracy_t']
    if "NT" in training_config['random_string']: #XXX
        losses['contrastive_t'] = losses['contrastive_x']
        if output_accuracy:
            losses['accuracy_t'] = losses['accuracy_x']


    return losses

def calculate_persistence_baseline_loss():
    losses = {}
    n_batches = 0
    for batch in test_dataloader:
        batch_size, n_electrodes, n_samples = batch['data'].shape
        random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']]
        batch['data'] = batch['data'][:, random_electrodes, :]

        batch['data'] = batch['data'].to(model.device, dtype=model_config['dtype'], non_blocking=True)
        normalized_batch = batch['data']
        normalized_batch = normalized_batch - torch.mean(normalized_batch, dim=[0, 2], keepdim=True)
        normalized_batch = normalized_batch / (torch.std(normalized_batch, dim=[0, 2], keepdim=True) + 1) # note values are in range [-180, 180]
        batch['data'] = normalized_batch

        batch['data'] = batch['data'].reshape(bin_transformer(batch['data']).shape) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
        n_timebins = batch['data'].shape[2]


        output = batch['data'][:, :, :-1, :]
        target = batch['data'][:, :, 1:, :]

        if training_config['normalize_features']:
            output = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
            target = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)

        similarity = output.permute(1, 2, 0, 3) @ target.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
        if training_config['use_temperature_param']:
            similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))

        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_electrodes, n_timebins-1, 1).to(model.device, dtype=torch.long).reshape(-1)
        loss = {'contrastive': torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange),
                'accuracy': (similarity.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()}
    
        for key, value in loss.items():
            if key not in losses: losses[key] = 0
            losses[key] += value
        n_batches += 1
    return {k: v / n_batches for k, v in losses.items()}

def calculate_pretrain_test_loss():
    losses = {}
    n_batches = 0
    for batch in test_dataloader:
        batch['data'] = batch['data'].to(model.device, dtype=model_config['dtype'], non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(model.device, dtype=torch.long, non_blocking=True)

        normalized_batch = batch['data']
        normalized_batch = normalized_batch - torch.mean(normalized_batch, dim=[0, 2], keepdim=True)
        normalized_batch = normalized_batch / (torch.std(normalized_batch, dim=[0, 2], keepdim=True) + 1) # note values are in range [-180, 180]
        batch['data'] = normalized_batch

        loss = calculate_loss_function(batch)
        
        for key, value in loss.items():
            if key not in losses: losses[key] = 0
            losses[key] += value
        n_batches += 1
    return {k: v / n_batches for k, v in losses.items()}

# After all model components are created and before the training loop
if cluster_config['save_model_every_n_epochs'] > 0:  # Only save if we're saving models at all
    model_path = f"models_data/{cluster_config['dir_name']}/model_epoch_0.pth"
    os.makedirs(f"models_data/{cluster_config['dir_name']}", exist_ok=True)
    
    torch.save({
        'eval_results': {},  # Empty since no evaluation has happened yet
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'bin_transformer_state_dict': bin_transformer.state_dict(),
        'cross_model_state_dict': cross_model.state_dict(),
        'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
        'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
        'training_config': convert_dtypes(training_config),
        'model_config': convert_dtypes(model_config), 
        'cluster_config': convert_dtypes(cluster_config),
    }, model_path)
    
log(f"Calculating the baseline persistence loss...", priority=0)
bin_transformer.train()
model.train()
cross_model.train()
electrode_embeddings.train()
baseline_loss = calculate_persistence_baseline_loss()
log(f"Baseline persistence loss: {baseline_loss['contrastive']:.4f}, Accuracy: {baseline_loss['accuracy']:.4f}", priority=0)

log(f"Evaluating model...", priority=0)
bin_transformer.eval()
model.eval()
cross_model.eval()
electrode_embeddings.eval()
print(evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average'))

training_statistics_store = []
if wandb: 
    wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'],
    config={"training_config": training_config, "model_config": model_config, "cluster_config": cluster_config}, settings=wandb.Settings(init_timeout=480))
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()

    model.train()
    bin_transformer.train()
    cross_model.train()
    electrode_embeddings.train()
    # Main training loop
    epoch_losses = {}
    for batch_idx, batch in enumerate(train_dataloader):
        batch['data'] = batch['data'].to(device, dtype=model_config['dtype'], non_blocking=True) # (batch_size, n_electrodes, n_timesamples)
        batch['electrode_index'] = batch['electrode_index'].to(device, dtype=torch.long, non_blocking=True)
        subject_identifier, trial_id = batch['subject_trial'][0]

        normalized_batch = batch['data']
        normalized_batch = normalized_batch - torch.mean(normalized_batch, dim=[0, 2], keepdim=True)
        normalized_batch = normalized_batch / (torch.std(normalized_batch, dim=[0, 2], keepdim=True) + 1) # note values are in range [-180, 180]
        batch['data'] = normalized_batch

        for optimizer in optimizers: optimizer.zero_grad()

        # Use autocast with specified dtype
        if use_amp:
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                loss_dict = calculate_loss_function(batch)
                loss = (loss_dict['contrastive_t'] + loss_dict['contrastive_x']) / 2
            loss.backward()  # Direct backward pass without scaling
        else:
            log(f"Using standard backward pass", priority=0)
            loss_dict = calculate_loss_function(batch)
            loss = (loss_dict['contrastive_t'] + loss_dict['contrastive_x']) / 2
            loss.backward()

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
            log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), Loss: {loss.item():.4f} ({loss_dict['contrastive_t'].item():.4f}/{loss_dict['contrastive_x'].item():.4f}), Temperature: {torch.exp(model.temperature_param).item():.2f}/{torch.exp(cross_model.temperature_param).item():.2f}", priority=0)
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    bin_transformer.eval()
    cross_model.eval()
    electrode_embeddings.eval()
    eval_results = {f"train_{k}": v for k, v in epoch_losses.items()}
    #eval_results['train_loss'] = sum(epoch_losses.values()) / len(epoch_losses)
    with torch.no_grad():
        test_loss_dict = calculate_pretrain_test_loss()
        eval_results.update({f"test_{k}": v.item() for k, v in test_loss_dict.items()})
        log(f"Test loss: {(test_loss_dict['contrastive_t'] + test_loss_dict['contrastive_x'])/2:.4f} ({test_loss_dict['contrastive_t']:.4f}/{test_loss_dict['contrastive_x']:.4f}), Test Accuracy: {test_loss_dict['accuracy_t']:.4f}/{test_loss_dict['accuracy_x']:.4f}", priority=0)
        #eval_results['test_loss'] = sum(test_loss_dict.values()).item() / len(test_loss_dict)
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average')
            eval_results.update(evaluation_results_strings)
            print(evaluation_results_strings)
        time_remaining = (time.time() - epoch_start_time) * (training_config['n_epochs'] - (epoch_i + 1))
        days = int(time_remaining // (24 * 3600))
        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Estimated time remaining: {days}d, {time.strftime('%H:%M:%S', time.gmtime(time_remaining % (24 * 3600)))}", priority=0)
    if wandb: wandb.log(eval_results, step=epoch_i+1)
    training_statistics_store[-1].update(eval_results)

    # Save the model
    if (epoch_i+1) % cluster_config['save_model_every_n_epochs'] == 0 or epoch_i+1 == training_config['n_epochs']:
        model_path = f"models_data/{cluster_config['dir_name']}/model_epoch_{epoch_i+1}.pth"
        statistics_path = f"models_data/{cluster_config['dir_name']}/training_statistics.json"
        os.makedirs(f"models_data/{cluster_config['dir_name']}", exist_ok=True)
            
        torch.save({
            'eval_results': eval_results,
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'bin_transformer_state_dict': bin_transformer.state_dict(),
            'cross_model_state_dict': cross_model.state_dict(),
            'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()