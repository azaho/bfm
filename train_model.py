import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast, GradScaler, autocast_mode
import gc

from muon import Muon
from model_model import GranularModel, LinearBinTransformer, BinTransformer, LinearKernelTransformer
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit
from dataset import load_dataloaders, load_subjects
from evaluation_neuroprobe import FrozenModelEvaluation_SS_SM
from train_utils import log, update_dir_name, update_random_seed, convert_dtypes, parse_configs_from_args, get_default_configs, get_shared_memory_info
from torch.optim.lr_scheduler import ChainedScheduler

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
assert n_downsample_factor == 1, "n_downsample_factor must be 1, not supported in the LinearBinTransformer class yet."

log(f"Loading model...", priority=0)
if model_config['bin_encoder'] == "linear":
    bin_embed_transformer = LinearBinTransformer(
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
        identity_init=model_config['init_identity']
    )
elif model_config['bin_encoder'] == "transformer":
    bin_embed_transformer = BinTransformer(
        d_input=model_config['first_kernel'],
        d_model=model_config['transformer']['d_model'],
        n_layers=model_config['transformer']['n_layers_electrode'],
        n_heads=12,
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
        dropout=model_config['transformer']['dropout']
    ).to(device, dtype=model_config['dtype'])
    bin_unembed_transformer = bin_embed_transformer
    
bin_unembed_transformer = bin_embed_transformer
if model_config['separate_unembed']:
    bin_unembed_transformer = LinearKernelTransformer(
        d_input=model_config['first_kernel'],
        d_output=model_config['transformer']['d_model'],
    )
else:
    bin_unembed_transformer = torch.nn.Identity()
bin_unembed_transformer = bin_unembed_transformer.to(device, dtype=model_config['dtype'])
bin_embed_transformer = bin_embed_transformer.to(device, dtype=model_config['dtype'])

model = GranularModel(
    model_config['transformer']['d_model'] * model_config['second_kernel'],
    model_config['transformer']['d_model'], 
    model_config['transformer']['d_model'] * model_config['second_kernel'] if model_config['separate_unembed'] else model_config['first_kernel'] * model_config['second_kernel'],
    n_layers=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads'],
    n_cls_tokens=0,
    dropout=model_config['transformer']['dropout']
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
#         max_frequency=model_config['max_frequency']
#     ).to(device, dtype=model_config['dtype'])
# else:
#     electrode_data_embeddings = ElectrodeDataEmbedding(
#         electrode_embeddings, model_config['sample_timebin_size'], model_config['transformer']['d_model'], 
#         overall_sampling_rate=next(iter(all_subjects.values())).get_sampling_rate(0) # XXX remove this once figured out how to be flexible here regarding the sampling rate
#     ).to(device, dtype=model_config['dtype'])

# from btbench.btbench_config import BTBENCH_LITE_ELECTRODES # Only tempporal lobe electrodes for now
# for subject_identifier, subject in all_subjects.items():
#     consider_electrode_names = list(BTBENCH_LITE_ELECTRODES[subject_identifier])
#     electrode_subset = [electrode_label for electrode_label in consider_electrode_names if electrode_label.startswith('T') or electrode_label.startswith('P')]
#     subject.set_electrode_subset(electrode_subset)
#     log(f"Subject {subject_identifier} has {len(electrode_subset)} temporal and parietal lobe electrodes", priority=0)

eval_electrode_subset = {
    #'btbank3': ['T1cIe11'],
}

for subject in all_subjects.values():
    #subject.set_electrode_subset(['T1cIe11'])
    log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
    this_subject_trials = [trial_id for (sub_id, trial_id) in training_config['train_subject_trials'] if sub_id == subject.subject_identifier]
    electrode_embeddings.add_subject(subject)
electrode_embeddings = electrode_embeddings.to(device, dtype=model_config['dtype']) # moving to device again to ensure the new parameters are on the correct device

log(f"Loading dataloaders...", priority=0)
train_dataloader, test_dataloader = load_dataloaders(
    all_subjects, training_config['train_subject_trials'], training_config['p_test'], 
    model_config['context_length'], training_config['data_dtype'], 
    training_config['batch_size'],
    num_workers_dataloaders=cluster_config['num_workers_dataloaders'], 
    prefetch_factor=cluster_config['prefetch_factor'],
    max_n_electrodes=model_config['max_n_electrodes'],
    output_embeddings_map=electrode_embeddings.embeddings_map
)

eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in training_config['eval_subject_trials']]
#eval_tasks = ['gpt2_surprisal', 'volume', 'word_part_speech', 'pitch', 'speech']
eval_tasks = ['gpt2_surprisal', 'speech']
evaluation = FrozenModelEvaluation_SS_SM(
    eval_tasks, eval_subject_trials, 
    training_config['data_dtype'], training_config['batch_size'], # Can have a bigger batch size here if that speeds things up
    electrode_embeddings.embeddings_map,
    num_workers_eval=cluster_config['num_workers_eval'],
    prefetch_factor=cluster_config['prefetch_factor'],
    feature_aggregation_method=cluster_config['eval_aggregation_method'],
    electrode_subset=eval_electrode_subset,
    max_n_electrodes=model_config['max_n_electrodes']
)

# After model initialization
amp_dtype = model_config['amp_dtype']
use_amp = model_config['use_mixed_precision'] and torch.cuda.is_available()
scaler = None  # No scaler needed for bfloat16

bin_transformer_params = list(bin_embed_transformer.parameters())
if model_config['separate_unembed']: 
    bin_transformer_params += list(bin_unembed_transformer.parameters())
all_params = list(model.parameters()) + bin_transformer_params + list(electrode_embeddings.parameters())
# filter to only include parameters that require grad
#all_params = [p for p in all_params if p.requires_grad]
n_model_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in bin_transformer_params)
n_embed_params = sum(p.numel() for p in electrode_embeddings.parameters())
log(f"Model parameters: {n_model_params:,}", priority=0)
log(f"Embedding parameters: {n_embed_params:,}", priority=0)
log(f"Total parameters: {n_model_params + n_embed_params:,}", priority=0)
model_config['n_params'] = {
    'model': n_model_params,
    'embeddings': n_embed_params,
    'total': n_model_params + n_embed_params
}

torch.autograd.set_detect_anomaly(True)

optimizers = []
if training_config['optimizer'] == 'Muon':
    #all_params = list(model.parameters())
    matrix_params = [p for p in all_params if p.ndim == 2]
    other_params = [p for p in all_params if p.ndim != 2]
    #other_params += list(electrode_data_embeddings.parameters()) # use adam for electrode data embeddings

    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
    if len(other_params) > 0:
        optimizers.append(torch.optim.AdamW(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'], betas=(0.9, 0.95)))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'], betas=(0.9, 0.95))]
schedulers = []
# Apply warmup if specified
warmup_steps = training_config.get('warmup_steps', 0)
if warmup_steps > 0:
    for optimizer in optimizers:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
        
        main_scheduler = None
        # Create main scheduler
        if training_config['lr_schedule'] == 'linear':
            main_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-5, 
                                                             total_iters=training_config['n_epochs'] * len(train_dataloader))
        elif training_config['lr_schedule'] == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                       T_max=training_config['n_epochs'] * len(train_dataloader))
        
        # Chain the schedulers together
        schedulers.append(ChainedScheduler([warmup_scheduler, main_scheduler]) if main_scheduler is not None else warmup_scheduler)
else:
    # If no warmup, just use the main scheduler
    for optimizer in optimizers:
        if training_config['lr_schedule'] == 'linear':
            schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-5, 
                                                              total_iters=training_config['n_epochs'] * len(train_dataloader)))
        elif training_config['lr_schedule'] == 'cosine':
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                        T_max=training_config['n_epochs'] * len(train_dataloader)))

def calculate_loss_function(batch, output_accuracy=True):
    # batch['data'] shape: (batch_size, n_electrodes, n_timebins)
    # batch['electrode_index'] shape: (batch_size, n_electrodes)
    losses = {}
    def _add_to_loss(output, target, loss_suffix):
        if training_config['normalize_features']:
            output_ = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
            target_ = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)
        similarity = output_.permute(1, 2, 0, 3) @ target_.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
        if training_config['use_temperature_param']:
            similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))
        expanded_arange_bin = torch.arange(batch_size).unsqueeze(0).repeat(output.shape[1], output.shape[2], 1).to(model.device, dtype=torch.long).reshape(-1)
        loss_bin = torch.nn.functional.cross_entropy(similarity[:, :, :, :].view(-1, batch_size), expanded_arange_bin)
        losses[f'contrastive_{loss_suffix}'] = loss_bin
        if output_accuracy:
            accuracy_bin = (similarity[:, :, :, :].view(-1, batch_size).argmax(dim=-1) == expanded_arange_bin).float().mean()
            losses[f'accuracy_{loss_suffix}'] = accuracy_bin
        return losses

    future_bin_idx = training_config['future_bin_idx']

    # Randomly select a subset of electrodes
    batch_size, n_electrodes, n_samples = batch['data'].shape
    random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']]
    n_electrodes = min(model_config['max_n_electrodes'], n_electrodes)
    batch['data'] = batch['data'][:, random_electrodes, :]
    batch['electrode_index'] = batch['electrode_index'][:, random_electrodes]

    n_timebins = n_samples//model_config['first_kernel']
    batch['data'] = batch['data'].reshape(batch_size, n_electrodes, n_timebins, model_config['first_kernel'])

    masked_batch_data = batch['data'].clone()

    zero_time_indices = np.random.choice(n_timebins, size=int(n_timebins*training_config['p_masked_timebins']), replace=False)
    mask = torch.ones_like(masked_batch_data)
    mask[:, :, zero_time_indices, :] = 0
    masked_batch_data = masked_batch_data * mask

    bin_embed_transformed_data = bin_embed_transformer(masked_batch_data) # shape: (batch_size, n_electrodes, n_timebins, d_model)
    bin_unembed_transformed_data = bin_unembed_transformer(batch['data']) # shape: (batch_size, n_electrodes, n_timebins, d_model or first_kernel)

    # if future_bin_idx > 0:
    #     _add_to_loss(output=bin_embed_transformed_data[:, :, :-future_bin_idx, :], target=bin_unembed_transformed_data[:, :, future_bin_idx:, :], loss_suffix='bin')
    # else:
    #     _add_to_loss(output=bin_embed_transformed_data, target=bin_unembed_transformed_data, loss_suffix='bin')

    # reshape 
    bin_embed_transformed_data = bin_embed_transformed_data.reshape(batch_size, n_electrodes, (n_timebins)//model_config['second_kernel'], model_config['transformer']['d_model']*model_config['second_kernel'])
    bin_unembed_transformed_data = bin_unembed_transformed_data.reshape(batch_size, n_electrodes, (n_timebins)//model_config['second_kernel'], model_config['transformer']['d_model']*model_config['second_kernel'] if model_config['separate_unembed'] else model_config['first_kernel']*model_config['second_kernel'])
    embeddings = electrode_embeddings.forward(batch['electrode_index']).unsqueeze(-2).repeat(1, 1, (n_timebins)//model_config['second_kernel'], 1) # shape: (batch_size, n_electrodes, d_model, n_timebins-future_bin_idx)

    if future_bin_idx > 0:
        bin_embed_transformed_data = bin_embed_transformed_data[:, :, :-future_bin_idx, :].clone()
        bin_unembed_transformed_data = bin_unembed_transformed_data[:, :, future_bin_idx:, :].clone()
        embeddings = embeddings[:, :, :-future_bin_idx, :].clone()

    # Split electrodes into two halves
    electrodes_a = np.arange(n_electrodes//2)
    electrodes_b = np.arange(n_electrodes//2, n_electrodes)
    
    # Create a new tensor instead of modifying in-place
    modified_data = bin_embed_transformed_data.clone()
    modified_data[:, electrodes_b, :, :] = 0  # Zero out the second half

    # Process the first half (A electrodes)
    model_output = model(modified_data, embeddings=embeddings)

    model_output_a = model_output[:, electrodes_a, :, :].clone()
    target_a = bin_unembed_transformed_data[:, electrodes_a, :, :].clone()

    model_output_b = model_output[:, electrodes_b, :, :].clone()
    target_b = bin_unembed_transformed_data[:, electrodes_b, :, :].clone()

    _add_to_loss(output=model_output_a, target=target_a, loss_suffix='time_a')
    _add_to_loss(output=model_output_b, target=target_b, loss_suffix='time_b')
    
    return losses

def calculate_persistence_baseline_loss(stop_at_batch=5):
    losses = {}
    n_batches = 0
    future_bin_idx = training_config['future_bin_idx']
    for batch in test_dataloader:
        if n_batches >= stop_at_batch: break

        batch_size, n_electrodes, n_samples = batch['data'].shape
        random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']//2]
        batch['data'] = batch['data'][:, random_electrodes, :]
        n_electrodes = min(model_config['max_n_electrodes']//2, n_electrodes)

        batch['data'] = batch['data'].to(model.device, dtype=model_config['dtype'], non_blocking=True)
        normalized_batch = batch['data']
        normalized_batch = normalized_batch - torch.mean(normalized_batch, dim=[0, 2], keepdim=True)
        normalized_batch = normalized_batch / (torch.std(normalized_batch, dim=[0, 2], keepdim=True) + 1) # note values are in range [-180, 180]
        batch['data'] = normalized_batch

        first_kernel = model_config['first_kernel']
        sample_timebin_size = first_kernel #* int(2048//n_downsample_factor * model_config['sample_timebin_size'])
        batch['data'] = batch['data'].reshape(batch_size, n_electrodes, n_samples//sample_timebin_size, sample_timebin_size) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
        n_timebins = batch['data'].shape[2]


        output = torch.flip(batch['data'][:, :, :-future_bin_idx, :], dims=[-1]) if future_bin_idx > 0 else batch['data'][:, :, :, :]
        target = batch['data'][:, :, future_bin_idx:, :] if future_bin_idx > 0 else batch['data'][:, :, :, :]

        if training_config['normalize_features']:
            output = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
            target = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)

        similarity = output.permute(1, 2, 0, 3) @ target.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
        if training_config['use_temperature_param']:
            similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))

        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_electrodes, n_timebins-future_bin_idx, 1).to(model.device, dtype=torch.long).reshape(-1)
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
        'bin_embed_transformer_state_dict': bin_embed_transformer.state_dict(),
        'bin_unembed_transformer_state_dict': bin_unembed_transformer.state_dict(),
        'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
        'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
        'training_config': convert_dtypes(training_config),
        'model_config': convert_dtypes(model_config), 
        'cluster_config': convert_dtypes(cluster_config),
    }, model_path)


if wandb: 
    wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'],
    config={"training_config": training_config, "model_config": model_config, "cluster_config": cluster_config}, settings=wandb.Settings(init_timeout=480))
    
log(f"Calculating the baseline persistence loss...", priority=0)
bin_embed_transformer.train()
bin_unembed_transformer.train()
model.train()
electrode_embeddings.train()
baseline_loss = calculate_persistence_baseline_loss()
log(f"Baseline persistence loss: {baseline_loss['contrastive']:.4f}, Accuracy: {baseline_loss['accuracy']:.4f}", priority=0)
del baseline_loss
torch.cuda.empty_cache()
gc.collect()

if not training_config['no_initial_init']:
    log(f"Evaluating model...", priority=0)
    bin_embed_transformer.eval()
    model.eval()
    electrode_embeddings.eval()
    eval_results = {}

    eval_raw = evaluation.evaluate_on_all_metrics(model, bin_embed_transformer, electrode_embeddings, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average', raw_data=True, key_prefix="raw_")
    eval_results.update(eval_raw)
    print("eval_raw", eval_raw)

    eval_bin_transformer = evaluation.evaluate_on_all_metrics(model, bin_embed_transformer, electrode_embeddings, quick_eval=cluster_config['quick_eval'], only_bin_transformer=True, only_keys_containing='auroc/average', key_prefix="bin_")
    eval_results.update(eval_bin_transformer)
    print("eval_bin_transformer", eval_bin_transformer)

    eval_full_model = evaluation.evaluate_on_all_metrics(model, bin_embed_transformer, electrode_embeddings, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average')
    print("eval_full_model", eval_full_model)
    eval_results.update(eval_full_model)

    if wandb: wandb.log(eval_results, step=1)
    del eval_full_model, eval_bin_transformer
    torch.cuda.empty_cache()
    gc.collect()

training_statistics_store = []
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()

    model.train()
    bin_embed_transformer.train()
    bin_unembed_transformer.train()
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
                loss = sum([v for k, v in loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in loss_dict.items() if 'accuracy' not in k])  # Only use contrastive_a loss
            loss.backward()
        else:
            loss_dict = calculate_loss_function(batch)
            loss = sum([v for k, v in loss_dict.items() if 'accuracy' not in k]) / len([v for k, v in loss_dict.items() if 'accuracy' not in k])  # Only use contrastive_a loss
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
            log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), LR: {optimizers[0].param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f} ({losses_string}), Temp {torch.exp(model.temperature_param).item():.4f}", priority=0)
        
        if batch_idx % 20 == 0:
            del loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()
        
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    bin_embed_transformer.eval()
    bin_unembed_transformer.eval()
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
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings_bin_transformer = evaluation.evaluate_on_all_metrics(model, bin_embed_transformer, electrode_embeddings, quick_eval=cluster_config['quick_eval'], only_bin_transformer=True, only_keys_containing='auroc/average', key_prefix="bin_")
            eval_results.update(evaluation_results_strings_bin_transformer)
            log("eval_bin_transformer" + str(evaluation_results_strings_bin_transformer))
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, bin_embed_transformer, electrode_embeddings, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average')
            eval_results.update(evaluation_results_strings)
            log("eval_full_model" + str(evaluation_results_strings))
            del evaluation_results_strings, evaluation_results_strings_bin_transformer
            torch.cuda.empty_cache()
            gc.collect()
        time_remaining = (time.time() - epoch_start_time) * (training_config['n_epochs'] - (epoch_i + 1))
        days = int(time_remaining // (24 * 3600))
        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Estimated time remaining: {days}d, {time.strftime('%H:%M:%S', time.gmtime(time_remaining % (24 * 3600)))}", priority=0)
    if wandb: wandb.log(eval_results, step=epoch_i+2) # XXX adding step=1 to the first log
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
            'bin_embed_transformer_state_dict': bin_embed_transformer.state_dict(),
            'bin_unembed_transformer_state_dict': bin_unembed_transformer.state_dict(),
            'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()