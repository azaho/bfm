# %%
from train_model_single_electrode_new_lin import *
import json

def save_checkpoint(step, training_logs, model, embed, unembed, inverter, save_dir, filename_base):
    """Save both training logs and model checkpoint in a single function"""
    # Save training logs
    log_filename = f'{filename_base}.json'
    log_save_path = os.path.join(save_dir, log_filename)
    with open(log_save_path, 'w') as f:
        json.dump(training_logs, f, indent=2)
    log(f"Saved training logs to {log_save_path}")
    
    # Save model and state dictionaries
    model_filename = f'{filename_base}_model.pt'
    model_save_path = os.path.join(save_dir, model_filename)
    save_dict = {
        'model': model.state_dict(),
        'embed': embed.state_dict(),
        'unembed': unembed.state_dict(),
        'inverter': inverter.state_dict(),
        #'optimizer_states': [optimizer.state_dict() for optimizer in optimizers],
        'training_config': {
            'subject_id': subject.subject_id,
            'trial_id': trial_id,
            'window_size': window_size,
            'd_embed': d_embed,
            'resolution': resolution,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'initial_lr': initial_lr,
            'electrode_subset': electrode_subset,
            'eval_electrode_index': eval_electrode_index
        }
    }
    torch.save(save_dict, model_save_path)
    log(f"Saved model and state dictionaries to {model_save_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')
dtype = torch.float32

subject_id, trial_id = (3, 1) # changed from 3, 0
window_size = 2048
subject = BrainTreebankSubject(subject_id, cache=True)

log(f'Subject: {subject.subject_identifier}, Trial: {trial_id}, loading data...')
electrode_subset = subject.electrode_labels
eval_electrode_index = electrode_subset.index('T1cIe11')
subject.set_electrode_subset(electrode_subset)
dataset = SubjectTrialDataset_SingleElectrode(subject, trial_id, window_size=window_size, dtype=dtype, unsqueeze_electrode_dimension=False, electrodes_subset=electrode_subset)
log("Data shape: " + str(dataset[0]['data'].shape))

log("Loading the eval trial...")
subject.load_neural_data(0)
log("Done.")

# %%
d_embed = 128
n_steps = 3000
batch_size = 128
log_every_step = min(300, n_steps//10)

# Get resolution from argparse
import argparse
parser = argparse.ArgumentParser(description='Train model for single electrode')
parser.add_argument('--resolution', type=int, default=10, help='Resolution for embedder')
args = parser.parse_args()
resolution = args.resolution

save_dir = "eval_results/juno/"
os.makedirs(save_dir, exist_ok=True)

filename_base = f'{subject.subject_identifier}_{trial_id}_embed{d_embed}_resolution{resolution}'

log(f'Creating models...')
import itertools
dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
dataloader = iter(itertools.cycle(dataloader))

embed = EmbedderDiscretized(d_model=d_embed, resolution=resolution, range=(-3, 3))
unembed = EmbedderDiscretized(d_model=d_embed, resolution=resolution, range=(-3, 3))

model = ContrastiveModel(d_input=n_samples_per_bin, embed=embed, unembed=unembed).to(device, dtype=dtype)
masker = NoneMasker()

# Create samples from 10 random indices of the dataset
samples = torch.cat([dataset[random.randint(0, len(dataset)-1)]['data'].flatten() for _ in range(n_samples_inverter)])
inverter = DistributionInverter(samples=samples).to(device, dtype=dtype)

evaluation = ModelEvaluation_BTBench(model, inverter, [(subject, 0)], ["speech", "gpt2_surprisal"], feature_aggregation_method='concat', 
                                        mean_collapse_factor=mean_collapse_factor, eval_electrode_index=eval_electrode_index)

log(f'Training model...')
initial_lr = 0.003
use_muon = True
optimizers = []
schedulers = []
if use_muon:
    from muon import Muon
    all_params = list(model.parameters())
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]
    optimizers.append(Muon(matrix_params, lr=initial_lr, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
    if len(other_params) > 0:
        optimizers.append(torch.optim.AdamW(other_params, lr=initial_lr, betas=(0.9, 0.95)))
    #schedulers.append(None)  # Muon doesn't support schedulers
    #schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizers[1], start_factor=1.0, end_factor=0.0, total_iters=n_steps))
else:
    optimizers = [torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95))]
    #schedulers = [torch.optim.lr_scheduler.LinearLR(optimizers[0], start_factor=1.0, end_factor=0.0, total_iters=n_steps)]

log("Evaluating the model before training...")
evaluation_results = evaluation.evaluate()
log(evaluation_results, indent=2)
evaluation_results['step'] = 0
evaluation_results['train_loss'] = -1
training_logs = [evaluation_results]

step = 1
for batch in dataloader:
    for optimizer in optimizers:
        optimizer.zero_grad()

    batch_data = batch['data'].to(device, dtype=dtype).reshape(batch_size, window_size//n_samples_per_bin, n_samples_per_bin) # shape (batch_size, seq_len, 1)
    batch_data = inverter(batch_data)
    masked_x, mask = masker.forward(batch_data)

    loss = model.calculate_loss(masked_x.unsqueeze(-2), mask=mask)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    
    # Step the schedulers
    for scheduler in schedulers:
        if scheduler is not None:
            scheduler.step()
    
    # Log metrics
    log_dict = {
        'train_loss': loss.item(),
        'step': step,
    }
    
    if step % log_every_step == 0:
        current_lr = optimizers[-1].param_groups[0]['lr']
        log(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Add evaluation results
        evaluation_results = evaluation.evaluate()
        log_dict.update(evaluation_results)
        log(log_dict, indent=2)
        
    training_logs.append(log_dict)
    
    # Save training results to file
    if step % log_every_step == 0 or step == n_steps:
        save_checkpoint(
            step=step,
            training_logs=training_logs,
            model=model,
            embed=embed,
            unembed=unembed,
            inverter=inverter,
            save_dir=save_dir,
            filename_base=filename_base,
        )
        
    if step == n_steps:
        break # Only process one batch per step
    step += 1


# %%


