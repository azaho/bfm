import torch
import wandb, os, json
import time

from muon import Muon
from model_model import TransformerModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeDataEmbeddingFFT, ElectrodeEmbedding_Learned_FixedVocabulary

from dataset import load_dataloaders
from evaluation_btbench import FrozenModelEvaluation_SS_SM
from evaluation_moabb import FrozenModelEvaluation_MOABB
from train_utils import log, update_dir_name, update_random_seed, convert_dtypes, parse_configs_from_args, get_default_configs, get_shared_memory_info


training_config, model_config, cluster_config = get_default_configs(random_string="benchtest2", wandb_project="eeg_ieeg_experiments")
parse_configs_from_args(training_config, model_config, cluster_config)
update_random_seed(training_config)

# EEG channels
EEG_channels = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
training_config['train_subject_trials'] = [('mgh1', 3), ('mgh1', 2)]
training_config['eval_subject_trials'] = []

training_config['n_epochs'] *= 5

batch_size_multiplier = 1
training_config['batch_size'] *= batch_size_multiplier
training_config['learning_rate'] *= batch_size_multiplier
training_config['n_epochs'] *= batch_size_multiplier

cluster_config['eval_model_every_n_epochs'] = 3

model_size_multiplier = 1
model_config['transformer']['n_layers_electrode'] *= model_size_multiplier
model_config['transformer']['n_layers_time'] *= model_size_multiplier
model_config['transformer']['d_model'] *= model_size_multiplier

cluster_config['cache_subjects'] = True

dir_name = update_dir_name(model_config, training_config, cluster_config)
cluster_config['wandb_name'] = cluster_config['dir_name']
log(f"Directory name: {dir_name}", priority=0)

if len(cluster_config['wandb_project'])==0: wandb = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}", priority=0)


log(f"Loading dataloaders...", priority=0)
n_samples = model_config['max_n_timebins'] * model_config['sample_timebin_size']
all_subjects, train_dataloader, test_dataloader = load_dataloaders(
    training_config['train_subject_trials'], training_config['eval_subject_trials'], training_config['p_test'], 
    model_config['sample_timebin_size'], model_config['max_n_timebins'], training_config['data_dtype'], 
    training_config['batch_size'],
    num_workers_dataloaders=cluster_config['num_workers_dataloaders'], 
    cache=cluster_config['cache_subjects'], allow_corrupted=False,
    prefetch_factor=cluster_config['prefetch_factor'],
)

model = TransformerModel(
    model_config['transformer']['d_model'],  
    n_layers_electrode=model_config['transformer']['n_layers_electrode'], 
    n_layers_time=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads']
).to(device, dtype=model_config['dtype'])

if model_config['electrode_embedding']['type'] == 'learned':
    electrode_embeddings = ElectrodeEmbedding_Learned(
        model_config['transformer']['d_model'], 
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
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

electrode_data_embeddings = ElectrodeDataEmbeddingFFT(
    electrode_embeddings, model_config['sample_timebin_size'], 
    max_frequency_bin=model_config['max_frequency_bin']
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


model_eeg = TransformerModel(
    model_config['transformer']['d_model'],  
    n_layers_electrode=model_config['transformer']['n_layers_electrode'], 
    n_layers_time=model_config['transformer']['n_layers_time'],
    n_heads=model_config['transformer']['n_heads']
).to(device, dtype=model_config['dtype'])
electrode_embeddings_eeg = ElectrodeEmbedding_Learned_FixedVocabulary(
    model_config['transformer']['d_model'], 
    vocabulary_channels=EEG_channels,
    embedding_dim=model_config['electrode_embedding']['embedding_dim']
)
electrode_embeddings_eeg = electrode_embeddings_eeg.to(device, dtype=model_config['dtype'])
electrode_data_embeddings_eeg = ElectrodeDataEmbeddingFFT(
    electrode_embeddings_eeg, model_config['sample_timebin_size'], 
    max_frequency_bin=model_config['max_frequency_bin']
).to(device, dtype=model_config['dtype'])
for subject in all_subjects.values():
    this_subject_trials = [trial_id for (sub_id, trial_id) in training_config['train_subject_trials'] if sub_id == subject.subject_identifier]
    electrode_data_embeddings_eeg.add_subject(subject, subject.get_sampling_rate(this_subject_trials[0]))
    if model_config['init_normalization']:
        # Copying over from the full model to the EEG model, because this data is literally the same
        electrode_data_embeddings_eeg.normalization_means[subject.subject_identifier] = electrode_data_embeddings.normalization_means[subject.subject_identifier]
        electrode_data_embeddings_eeg.normalization_stds[subject.subject_identifier] = electrode_data_embeddings.normalization_stds[subject.subject_identifier]
electrode_data_embeddings_eeg = electrode_data_embeddings_eeg.to(device, dtype=model_config['dtype'])


electrode_embeddings_eeg.add_raw_embedding('moabb', EEG_channels)
electrode_data_embeddings_eeg.add_raw_subject('moabb', 1024)
evaluation = FrozenModelEvaluation_MOABB(
    EEG_channels, sampling_rate=1024, max_subjects=3,
)

all_params = list(model.parameters()) + list(electrode_data_embeddings.parameters()) + list(model_eeg.parameters()) + list(electrode_data_embeddings_eeg.parameters())
n_model_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model_eeg.parameters())
n_embed_params = sum(p.numel() for p in electrode_data_embeddings.parameters()) + sum(p.numel() for p in electrode_data_embeddings_eeg.parameters())
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
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]

    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
    optimizers.append(torch.optim.AdamW(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay']))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])]


def calculate_loss_function(batch, subject_identifier, trial_id):
    all_electrode_indices = all_subjects[subject_identifier].get_electrode_indices(trial_id)
    EEG_electrode_indices = torch.tensor([i for i in all_electrode_indices if all_subjects[subject_identifier].get_electrode_labels()[i] in EEG_channels])
    other_electrode_indices = torch.tensor([i for i in all_electrode_indices if i not in EEG_electrode_indices])

    batch_EEG_electrode_indices = torch.tensor([torch.where(all_electrode_indices == i)[0][0] for i in EEG_electrode_indices]) # getting the indices of the EEG electrodes in the batch
    batch_other_electrode_indices = torch.tensor([torch.where(all_electrode_indices == i)[0][0] for i in other_electrode_indices])

    electrode_embedded_data = electrode_data_embeddings.forward(subject_identifier, other_electrode_indices, batch[:, batch_other_electrode_indices, :], max_n_electrodes=model_config['max_n_electrodes'])
    electrode_embedded_data_eeg = electrode_data_embeddings_eeg.forward(subject_identifier, EEG_electrode_indices, batch[:, batch_EEG_electrode_indices, :], max_n_electrodes=model_config['max_n_electrodes'])

    # Randomly permute the electrodes
    permutation = torch.randperm(electrode_embedded_data.shape[1])
    permutation_eeg = torch.randperm(electrode_embedded_data_eeg.shape[1])
    electrode_embedded_data = electrode_embedded_data[:, permutation]
    electrode_embedded_data_eeg = electrode_embedded_data_eeg[:, permutation_eeg]

    # Put both halves of the batch through the models (eeg and ieeg). The outputs are of shape (batch_size, max_n_timebins, d_model)
    # Note that each output has the "electrode transformer" output, and the "time transformer" output. Those are the o_e and o_t outputs.
    # We only use the time outputs for predicting the future timestep of the electrode transformer
    o_eeg_e_1, o_eeg_t_1 = model_eeg(electrode_embedded_data_eeg[:, :len(permutation_eeg)//2])
    o_eeg_e_2, o_eeg_t_2 = model_eeg(electrode_embedded_data_eeg[:, len(permutation_eeg)//2:])
    o_e_1, o_t_1 = model(electrode_embedded_data[:, :len(permutation)//2])
    o_e_2, o_t_2 = model(electrode_embedded_data[:, len(permutation)//2:])

    # Loss component 1: EEG model predicting the iEEG model, in the same timestep. Since we are splitting the data into halves, we have 4 possible combinations
    similarity_1 = torch.matmul(o_eeg_e_1[:, :].permute(1, 0, 2), o_e_1[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_eeg_e_2[:, :].permute(1, 0, 2), o_e_2[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_3 = torch.matmul(o_eeg_e_1[:, :].permute(1, 0, 2), o_e_2[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_4 = torch.matmul(o_eeg_e_2[:, :].permute(1, 0, 2), o_e_1[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins'], 1).to(device, dtype=torch.long).reshape(-1)
    loss_eeg_ieeg = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_3.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_4.view(-1, training_config['batch_size']), expanded_arange)) / 4
    
    # Loss component 2: iEEG model predicting the EEG model, in the same timestep. Since we are splitting the data into halves, we have 4 possible combinations
    # (note that this computation is a bit redundant, because the similarity matrices are symmetric, but we compute it for clarity for now)
    similarity_1 = torch.matmul(o_e_1[:, :].permute(1, 0, 2), o_eeg_e_1[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_e_2[:, :].permute(1, 0, 2), o_eeg_e_2[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_3 = torch.matmul(o_e_1[:, :].permute(1, 0, 2), o_eeg_e_2[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_4 = torch.matmul(o_e_2[:, :].permute(1, 0, 2), o_eeg_e_1[:, :].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins'], 1).to(device, dtype=torch.long).reshape(-1)
    loss_ieeg_eeg = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_3.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_4.view(-1, training_config['batch_size']), expanded_arange)) / 4
    
    # Loss component 3: EEG model predicting the future timestep of the EEG model. There are 2 possible combinations
    similarity_1 = torch.matmul(o_eeg_t_1[:, :-1].permute(1, 0, 2), o_eeg_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_eeg_t_2[:, :-1].permute(1, 0, 2), o_eeg_e_1[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    loss_eeg_eeg_t = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange)) / 2
    
    # Loss component 4: iEEG model predicting the future timestep of the iEEG model. There are 2 possible combinations
    similarity_1 = torch.matmul(o_t_1[:, :-1].permute(1, 0, 2), o_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_t_2[:, :-1].permute(1, 0, 2), o_e_1[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    loss_ieeg_ieeg_t = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange)) / 2
    
    # Loss component 5: EEG model predicting the future timestep of the iEEG model. There are 4 possible combinations
    similarity_1 = torch.matmul(o_eeg_t_1[:, :-1].permute(1, 0, 2), o_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_eeg_t_2[:, :-1].permute(1, 0, 2), o_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_3 = torch.matmul(o_eeg_t_1[:, :-1].permute(1, 0, 2), o_e_1[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_4 = torch.matmul(o_eeg_t_2[:, :-1].permute(1, 0, 2), o_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    loss_eeg_ieeg_t = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_3.view(-1, training_config['batch_size']), expanded_arange) + \
                    torch.nn.functional.cross_entropy(similarity_4.view(-1, training_config['batch_size']), expanded_arange)) / 4
    
    # Loss component 6: iEEG model predicting the future timestep of the EEG model. There are 4 possible combinations
    similarity_1 = torch.matmul(o_t_1[:, :-1].permute(1, 0, 2), o_eeg_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_2 = torch.matmul(o_t_2[:, :-1].permute(1, 0, 2), o_eeg_e_2[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_3 = torch.matmul(o_t_1[:, :-1].permute(1, 0, 2), o_eeg_e_1[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    similarity_4 = torch.matmul(o_t_2[:, :-1].permute(1, 0, 2), o_eeg_e_1[:, 1:].permute(1, 2, 0)) * model_eeg.temperature_param
    expanded_arange = torch.arange(training_config['batch_size']).unsqueeze(0).repeat(model_config['max_n_timebins']-1, 1).to(device, dtype=torch.long).reshape(-1)
    loss_ieeg_eeg_t = (torch.nn.functional.cross_entropy(similarity_1.view(-1, training_config['batch_size']), expanded_arange) + \
                        torch.nn.functional.cross_entropy(similarity_2.view(-1, training_config['batch_size']), expanded_arange) + \
                        torch.nn.functional.cross_entropy(similarity_3.view(-1, training_config['batch_size']), expanded_arange) + \
                        torch.nn.functional.cross_entropy(similarity_4.view(-1, training_config['batch_size']), expanded_arange)) / 4

    # TODO: need to double check that when i am applying the cross entropy loss, we are actually using the correct ordering of the outputs, such that
    # we have the current timestep predict the future, as opposed to the future predict the past. Need to double check this.

    return {'loss_eeg_ieeg': loss_eeg_ieeg, 
            'loss_ieeg_eeg': loss_ieeg_eeg, 
            'loss_eeg_eeg_t': loss_eeg_eeg_t, 
            'loss_ieeg_ieeg_t': loss_ieeg_ieeg_t, 
            'loss_eeg_ieeg_t': loss_eeg_ieeg_t, 
            'loss_ieeg_eeg_t': loss_ieeg_eeg_t}

training_statistics_store = []
if wandb: 
    wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'],
    config={"training_config": training_config, "model_config": model_config, "cluster_config": cluster_config}, settings=wandb.Settings(init_timeout=480))
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()
    model.train()

    # Main training loop
    epoch_losses = {}
    for batch_idx, (batch, (subject_identifiers, trial_ids)) in enumerate(train_dataloader):
        for optimizer in optimizers: optimizer.zero_grad()
        subject_identifier = subject_identifiers[0]
        trial_id = trial_ids[0].item()
        
        batch = batch.to(device, dtype=model_config['dtype'], non_blocking=True)
        loss_dict = calculate_loss_function(batch, subject_identifier, trial_id)

        for key, loss in loss_dict.items():
            if key not in epoch_losses: epoch_losses[key] = 0
            epoch_losses[key] += loss.item()

        loss = sum(loss_dict.values()) / len(loss_dict)
        loss.backward()
        for optimizer in optimizers: optimizer.step()

        training_statistics_store.append({
            'epoch': epoch_i+1,
            'batch': batch_idx+1,
            'subject_identifier': subject_identifier,
            'trial_id': trial_id,
            'batch_loss': loss.item(),
            'timestamp': time.time(),
            **{f"batch_{k}": v.item() for k, v in loss_dict.items()}
        })

        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), Loss: {loss.item():.4f} ({', '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])})", priority=0)
    for key, loss in epoch_losses.items():
        epoch_losses[key] /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    eval_results = {f"train_{k}": v for k, v in epoch_losses.items()}
    eval_results['train_loss'] = sum(epoch_losses.values()) / len(epoch_losses)
    with torch.no_grad():
        test_loss_dict = model.calculate_pretrain_test_loss(electrode_data_embeddings, test_dataloader, all_subjects, calculate_loss_function=calculate_loss_function)
        eval_results.update({f"test_{k}": v.item() for k, v in test_loss_dict.items()})
        eval_results['test_loss'] = sum(test_loss_dict.values()).item() / len(test_loss_dict)
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model_eeg, electrode_data_embeddings_eeg, log_priority=1, quick_eval=True)
            eval_results.update(evaluation_results_strings)
            log(f"Evaluation results: {evaluation_results_strings}", priority=0)
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
            'model_eeg_state_dict': model_eeg.state_dict(),
            'electrode_data_embeddings_eeg_state_dict': electrode_data_embeddings_eeg.state_dict(),
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)
        with open(statistics_path, 'w') as f:
            json.dump(training_statistics_store, f)
if wandb: wandb.finish()