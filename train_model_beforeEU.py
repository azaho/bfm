import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast, GradScaler, autocast_mode
import gc

from muon import Muon
from model_model import GranularModel, LinearBinTransformer, CrossModel, BinTransformer
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit
from dataset import load_dataloaders, load_subjects
from evaluation_btbench import FrozenModelEvaluation_SS_SM
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
    bin_transformer = LinearBinTransformer(
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
        identity_init=model_config['init_identity']
    )
elif model_config['bin_encoder'] == "transformer":
    bin_transformer = BinTransformer(
        first_kernel=int(model_config['sample_timebin_size']*2048)//16, 
        d_model=64,#model_config['transformer']['d_model'],
        n_layers=2,
        n_heads=4,
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
        n_downsample_factor=n_downsample_factor,
        identity_init=model_config['init_identity']
    ).to(device, dtype=model_config['dtype'])
bin_transformer = bin_transformer.to(device, dtype=model_config['dtype'])

model = GranularModel(
    int(model_config['sample_timebin_size'] * 2048 // n_downsample_factor),
    model_config['transformer']['d_model'],  
    n_layers=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads'],
    identity_init=model_config['init_identity'],
    n_cls_tokens=1
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
#eval_tasks = ['gpt2_surprisal', 'volume', 'word_part_speech', 'pitch', 'speech']
eval_tasks = ['gpt2_surprisal', 'volume', 'speech']
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

all_params = list(model.parameters()) + list(bin_transformer.parameters()) + list(electrode_embeddings.parameters())
# filter to only include parameters that require grad
#all_params = [p for p in all_params if p.requires_grad]
n_model_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in bin_transformer.parameters())
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
    matrix_params = [p for p in all_params if p.ndim == 2]
    other_params = [p for p in all_params if p.ndim != 2]
    #other_params += list(electrode_data_embeddings.parameters()) # use adam for electrode data embeddings

    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
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

    future_bin_idx = training_config['future_bin_idx']

    # Randomly select a subset of electrodes
    batch_size, n_electrodes, n_samples = batch['data'].shape
    random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']]
    n_electrodes = min(model_config['max_n_electrodes'], n_electrodes)
    batch['data'] = batch['data'][:, random_electrodes, :]
    batch['electrode_index'] = batch['electrode_index'][:, random_electrodes]

    electrodes_a = np.arange(n_electrodes//2)
    electrodes_b = np.arange(n_electrodes//2, n_electrodes)

    embeddings = electrode_embeddings.forward(batch['electrode_index']) # shape: (batch_size, n_electrodes, d_model)

    bin_transformed_data = bin_transformer(batch['data']) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
    batch_size, n_electrodes, n_timebins, sample_timebin_size = bin_transformed_data.shape

    bin_transformed_data_copy = bin_transformed_data[:, :, :, :].clone()


    masked_tokens = torch.zeros(batch_size, n_electrodes, n_timebins, device=model.device, dtype=model.dtype)
    bin_transformed_data[:, electrodes_b, :, :] = 0 # mask out the second half of the electrodes
    masked_tokens[:, electrodes_b, :] = 0 # removed the mask tokens again 


    output, output_a_cls = model(bin_transformed_data[:, :, :-future_bin_idx, :], embeddings, masked_tokens=masked_tokens[:, :, :-future_bin_idx], return_cls_token=True) # shape: (batch_size, n_electrodes, n_timebins-future_bin_idx, sample_timebin_size*SR//n_downsample_factor)
    target = bin_transformed_data_copy[:, :, future_bin_idx:, :] # shape: (batch_size, n_electrodes, n_timebins-future_bin_idx, sample_timebin_size*SR//n_downsample_factor)

    if training_config['normalize_features']:
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
        target = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)

    similarity = output.permute(1, 2, 0, 3) @ target.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
    if training_config['use_temperature_param']:
        similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype))

    expanded_arange_a = torch.arange(batch_size).unsqueeze(0).repeat(len(electrodes_a), n_timebins-future_bin_idx, 1).to(model.device, dtype=torch.long).reshape(-1)
    loss_a = torch.nn.functional.cross_entropy(similarity[electrodes_a, :, :, :].view(-1, batch_size), expanded_arange_a)
    losses['contrastive_a'] = loss_a
    if output_accuracy:
        accuracy_a = (similarity[electrodes_a, :, :, :].view(-1, batch_size).argmax(dim=-1) == expanded_arange_a).float().mean()
        losses['accuracy_a'] = accuracy_a

    expanded_arange_b = torch.arange(batch_size).unsqueeze(0).repeat(len(electrodes_b), n_timebins-future_bin_idx, 1).to(model.device, dtype=torch.long).reshape(-1)
    loss_b = torch.nn.functional.cross_entropy(similarity[electrodes_b, :, :, :].view(-1, batch_size), expanded_arange_b)
    losses['contrastive_b'] = loss_b
    if output_accuracy:
        accuracy_b = (similarity[electrodes_b, :, :, :].view(-1, batch_size).argmax(dim=-1) == expanded_arange_b).float().mean()
        losses['accuracy_b'] = accuracy_b

    _, output_b_cls = model(bin_transformed_data_copy[:, electrodes_b, future_bin_idx:, :], embeddings[:, electrodes_b, :], return_cls_token=True)
    if training_config['normalize_features'] and False: # XXX removing normalization of the output features here
        output_a_cls = output_a_cls / (torch.norm(output_a_cls, dim=-1, keepdim=True) + 0.001)
        output_b_cls = output_b_cls / (torch.norm(output_b_cls, dim=-1, keepdim=True) + 0.001)
    similarity = output_a_cls.permute(1, 2, 0, 3) @ output_b_cls.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-1, batch_size, batch_size)
    if training_config['use_temperature_param']:
        similarity = similarity * torch.minimum(torch.exp(model.temperature_param), torch.tensor(training_config['max_temperature_param'], device=model.device, dtype=model.dtype)) # XXX going back to temp1 for both
    expanded_arange_cls = torch.arange(batch_size).unsqueeze(0).repeat(model.n_cls_tokens, n_timebins-future_bin_idx, 1).to(model.device, dtype=torch.long).reshape(-1)
    loss_cls = torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange_cls)
    losses['contrastive_cls'] = loss_cls
    if output_accuracy:
        accuracy_cls = (similarity.view(-1, batch_size).argmax(dim=-1) == expanded_arange_cls).float().mean()
        losses['accuracy_cls'] = accuracy_cls

    return losses

def calculate_persistence_baseline_loss():
    losses = {}
    n_batches = 0
    future_bin_idx = training_config['future_bin_idx']
    for batch in test_dataloader:
        batch_size, n_electrodes, n_samples = batch['data'].shape
        random_electrodes = torch.randperm(n_electrodes)[:model_config['max_n_electrodes']//2]
        batch['data'] = batch['data'][:, random_electrodes, :]
        n_electrodes = min(model_config['max_n_electrodes']//2, n_electrodes)

        batch['data'] = batch['data'].to(model.device, dtype=model_config['dtype'], non_blocking=True)
        normalized_batch = batch['data']
        normalized_batch = normalized_batch - torch.mean(normalized_batch, dim=[0, 2], keepdim=True)
        normalized_batch = normalized_batch / (torch.std(normalized_batch, dim=[0, 2], keepdim=True) + 1) # note values are in range [-180, 180]
        batch['data'] = normalized_batch


        sample_timebin_size = int(2048//n_downsample_factor * model_config['sample_timebin_size'])
        batch['data'] = batch['data'].reshape(batch_size, n_electrodes, n_samples//sample_timebin_size, sample_timebin_size) # shape: (batch_size, n_electrodes, n_timebins, sample_timebin_size*SR//n_downsample_factor)
        n_timebins = batch['data'].shape[2]


        output = torch.flip(batch['data'][:, :, :-future_bin_idx, :], dims=[-1])
        target = batch['data'][:, :, future_bin_idx:, :]

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
        'bin_transformer_state_dict': bin_transformer.state_dict(),
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
bin_transformer.train()
model.train()
electrode_embeddings.train()
baseline_loss = calculate_persistence_baseline_loss()
log(f"Baseline persistence loss: {baseline_loss['contrastive']:.4f}, Accuracy: {baseline_loss['accuracy']:.4f}", priority=0)
del baseline_loss
torch.cuda.empty_cache()
gc.collect()

log(f"Evaluating model...", priority=0)
bin_transformer.eval()
model.eval()
electrode_embeddings.eval()
eval_results = {}
eval_full_model = evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average')
eval_results.update(eval_full_model)
eval_bin_transformer = evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_bin_transformer=True, only_keys_containing='auroc/average', key_prefix="bin_")
eval_results.update(eval_bin_transformer)
print("eval_full_model", eval_full_model)
print("eval_bin_transformer", eval_bin_transformer)
if wandb: wandb.log(eval_results, step=1)
del eval_full_model, eval_bin_transformer
torch.cuda.empty_cache()
gc.collect()

training_statistics_store = []
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()

    model.train()
    bin_transformer.train()
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
            log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), LR: {optimizers[0].param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f} ({losses_string}), Temp {torch.exp(model.temperature_param).item():.4f} / {torch.exp(model.temperature_param2).item():.4f}", priority=0)
        
        if batch_idx % 10 == 0:
            del loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()
        
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    bin_transformer.eval()
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
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_keys_containing='auroc/average')
            eval_results.update(evaluation_results_strings)
            evaluation_results_strings_bin_transformer = evaluation.evaluate_on_all_metrics(model, bin_transformer, electrode_embeddings, log_priority=1, quick_eval=cluster_config['quick_eval'], only_bin_transformer=True, only_keys_containing='auroc/average', key_prefix="bin_")
            eval_results.update(evaluation_results_strings_bin_transformer)
            print("eval_full_model", evaluation_results_strings)
            print("eval_bin_transformer", evaluation_results_strings_bin_transformer)
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
            'bin_transformer_state_dict': bin_transformer.state_dict(),
            'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()