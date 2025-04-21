import torch
import wandb, os, json
import time

from muon import Muon
from model_model import TransformerModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeDataEmbeddingFFT, ElectrodeDataEmbedding

from dataset import load_dataloaders, load_subjects
from evaluation_btbench import FrozenModelEvaluation_SS_SM
from train_utils import log, update_dir_name, update_random_seed, convert_dtypes, parse_configs_from_args, get_default_configs, get_shared_memory_info

training_config, model_config, cluster_config = get_default_configs(random_string="TEMP", wandb_project="btbank_fft_exp")
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

log(f"Loading model...", priority=0)
model = TransformerModel(
    model_config['transformer']['d_model'],  
    n_layers_electrode=model_config['transformer']['n_layers_electrode'], 
    n_layers_time=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads'],
    use_cls_token=model_config['transformer']['use_cls_token']
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

if model_config['electrode_embedding']['spectrogram']:
    electrode_data_embeddings = ElectrodeDataEmbeddingFFT(
        electrode_embeddings, model_config['sample_timebin_size'], 
        max_frequency_bin=model_config['max_frequency_bin']
    ).to(device, dtype=model_config['dtype'])
else:
    electrode_data_embeddings = ElectrodeDataEmbedding(
        electrode_embeddings, model_config['sample_timebin_size'], 
        overall_sampling_rate=next(iter(all_subjects.values())).get_sampling_rate(0) # XXX remove this once figured out how to be flexible here regarding the sampling rate
    ).to(device, dtype=model_config['dtype'])

for subject in all_subjects.values():
    this_subject_trials = [trial_id for (sub_id, trial_id) in training_config['train_subject_trials'] if sub_id == subject.subject_identifier]
    electrode_data_embeddings.add_subject(subject, subject.get_sampling_rate(this_subject_trials[0]))
    log(f"Adding subject {subject.subject_identifier} to electrode data embeddings...", priority=0)
    if model_config['init_normalization']:
        for trial_id in this_subject_trials:
            log(f"Initializing normalization for subject {subject.subject_identifier} trial {trial_id}...", priority=1, indent=1)
            electrode_data_embeddings.initialize_normalization(subject, trial_id, init_normalization_window_to=int(subject.get_sampling_rate(trial_id) * 60 * 5))
electrode_data_embeddings = electrode_data_embeddings.to(device, dtype=model_config['dtype']) # moving to device again to ensure the new parameters are on the correct device

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
evaluation = FrozenModelEvaluation_SS_SM(
    ['speech', 'volume', 'gpt2_surprisal', 'word_part_speech'], eval_subject_trials, 
    training_config['data_dtype'], training_config['batch_size'] * 2, # Can have a bigger batch size here if that speeds things up
    electrode_embeddings.embeddings_map,
    num_workers_eval=cluster_config['num_workers_eval'],
    prefetch_factor=cluster_config['prefetch_factor'],
    feature_aggregation_method=cluster_config['eval_aggregation_method'],
)


all_params = list(model.parameters()) + list(electrode_data_embeddings.parameters())
# filter to only include parameters that require grad
#all_params = [p for p in all_params if p.requires_grad]
n_model_params = sum(p.numel() for p in model.parameters())
n_embed_params = sum(p.numel() for p in electrode_data_embeddings.parameters())
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

def calculate_loss_function(batch):
    electrode_embedded_data = electrode_data_embeddings.forward(batch['data'], batch['electrode_index'])

    batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape

    permutation = torch.randperm(n_electrodes)
    electrode_embedded_data = electrode_embedded_data[:, permutation]

    n_electrodes_per_stream = int(n_electrodes * training_config['p_electrodes_per_stream'] * training_config['p_unmasked'])
    #print(f"n_electrodes_per_stream: {n_electrodes_per_stream}")

    o1_e, o1_t = model(electrode_embedded_data[:, :n_electrodes_per_stream, :, :]) # shape: (batch_size, n_timebins, d_model)
    o2_e, o2_t = model(electrode_embedded_data[:, -n_electrodes_per_stream:, :, :]) # shape: (batch_size, n_timebins, d_model)
    
    if training_config['projection_type'] == 'random_batch':
        random_matrix = torch.randn(d_model, d_model, device=model.device, dtype=model_config['dtype']) / (d_model ** 0.5)
        o1_t = torch.matmul(o1_t, random_matrix)
        o2_t = torch.matmul(o2_t, random_matrix)
        o1_e = torch.matmul(o1_e, random_matrix)
        o2_e = torch.matmul(o2_e, random_matrix)

    similarity_1 = torch.matmul(o1_t[:, :-training_config['future_bin_idx']].permute(1, 0, 2), o2_e[:, training_config['future_bin_idx']:].permute(1, 2, 0)) * (model.temperature_param)
    expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_timebins-training_config['future_bin_idx'], 1).to(model.device, dtype=torch.long).reshape(-1)
    loss = torch.nn.functional.cross_entropy(similarity_1.view(-1, batch_size), expanded_arange)

    if training_config['symmetric_loss']:
        similarity_2 = torch.matmul(o2_t[:, :-training_config['future_bin_idx']].permute(1, 0, 2), o1_e[:, training_config['future_bin_idx']:].permute(1, 2, 0)) * (model.temperature_param)
        loss += torch.nn.functional.cross_entropy(similarity_2.view(-1, batch_size), expanded_arange)
        loss /= 2
    
    return {'contrastive': loss}

def calculate_pretrain_test_loss():
    losses = {}
    n_batches = 0
    for batch in test_dataloader:
        batch['data'] = batch['data'].to(model.device, dtype=model_config['dtype'], non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(model.device, dtype=torch.long, non_blocking=True)
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
        'electrode_data_embeddings_state_dict': electrode_data_embeddings.state_dict(),
        'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
        'training_config': convert_dtypes(training_config),
        'model_config': convert_dtypes(model_config), 
        'cluster_config': convert_dtypes(cluster_config),
    }, model_path)

training_statistics_store = []
if wandb: 
    wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'],
    config={"training_config": training_config, "model_config": model_config, "cluster_config": cluster_config}, settings=wandb.Settings(init_timeout=480))
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()
    model.train()

    # Main training loop
    epoch_losses = {}
    for batch_idx, batch in enumerate(train_dataloader):
        batch['data'] = batch['data'].to(device, dtype=model_config['dtype'], non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(device, dtype=torch.long, non_blocking=True)
        subject_identifier, trial_id = batch['subject_trial'][0]

        for optimizer in optimizers: optimizer.zero_grad()

        loss_dict = calculate_loss_function(batch)

        for key, loss in loss_dict.items():
            if key not in epoch_losses: epoch_losses[key] = 0
            epoch_losses[key] += loss.item()

        loss = sum(loss_dict.values()) / len(loss_dict)
        loss.backward()
        for optimizer in optimizers: optimizer.step()
        for scheduler in schedulers: scheduler.step()

        training_statistics_store.append({
            'epoch': epoch_i+1,
            'batch': batch_idx+1,
            'subject_identifier': subject_identifier,
            'trial_id': trial_id,
            'batch_loss': loss.item(),
            'timestamp': time.time(),
            **{f"batch_{k}": v.item() for k, v in loss_dict.items()}
        })

        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), LR: {optimizers[0].param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f}", priority=0)
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    eval_results = {f"train_{k}": v for k, v in epoch_losses.items()}
    eval_results['train_loss'] = sum(epoch_losses.values()) / len(epoch_losses)
    with torch.no_grad():
        test_loss_dict = calculate_pretrain_test_loss()
        eval_results.update({f"test_{k}": v.item() for k, v in test_loss_dict.items()})
        eval_results['test_loss'] = sum(test_loss_dict.values()).item() / len(test_loss_dict)
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, electrode_data_embeddings, log_priority=1, quick_eval=True)
            eval_results.update(evaluation_results_strings)
        time_remaining = (time.time() - epoch_start_time) * (training_config['n_epochs'] - (epoch_i + 1))
        days = int(time_remaining // (24 * 3600))
        log(f"Estimated time remaining: {days}d, {time.strftime('%H:%M:%S', time.gmtime(time_remaining % (24 * 3600)))}", priority=0)
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
            'electrode_data_embeddings_state_dict': electrode_data_embeddings.state_dict(),
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()