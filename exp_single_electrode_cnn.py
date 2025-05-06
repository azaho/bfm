# %%
from train_model_single_electrode_new_lin import *
import json
import itertools

import btbench.btbench_config as btbench_config
from btbench.braintreebank_subject import BrainTreebankSubject as BTBench_BrainTreebankSubject

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')
dtype = torch.float32

train_subject_trials = [st for st in btbench_config.BTBENCH_FULL_SUBJECT_TRIALS if st not in btbench_config.BTBENCH_LITE_SUBJECT_TRIALS]
train_subject_trials = [(3, 1)]
window_size = 2048

all_subjects = {}
log("Loading the train subjects...")
for subject_id, trial_id in train_subject_trials:
    if subject_id not in all_subjects:
        all_subjects[subject_id] = BrainTreebankSubject(subject_id, cache=True)
    subject = all_subjects[subject_id]
    log(f'Subject: {subject.subject_identifier}, Trial: {trial_id}, loading data...', indent=1)
    subject.load_neural_data(trial_id)
log("Done.")

datasets = []
log("Loading the train datasets...")
for subject_id, trial_id in train_subject_trials:
    subject = all_subjects[subject_id]
    log(f"Loading subject {subject_id}, trial {trial_id}...", indent=1)
    dataset = SubjectTrialDataset_SingleElectrode(subject, trial_id, window_size=window_size, dtype=dtype, unsqueeze_electrode_dimension=False)
    datasets.append(dataset)
dataset = torch.utils.data.ConcatDataset(datasets)
log("Done.")

log("Data shape: " + str(dataset[0]['data'].shape) + "; Length: " + str(len(dataset)))

eval_subject_id, eval_trial_id = 3, 0
eval_subject = BTBench_BrainTreebankSubject(eval_subject_id, cache=True)
eval_electrode_index = eval_subject.electrode_labels.index('T1cIe11')

batch_size = 256 # up from 128
dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
dataloader = iter(itertools.cycle(dataloader))


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
            'n_steps': n_steps,
            'batch_size': batch_size,
            'eval_electrode_indices': [eval_electrode_index],
            'n_samples_per_bin': n_samples_per_bin
        }
    }
    torch.save(save_dict, model_save_path)
    log(f"Saved model and state dictionaries to {model_save_path}")

# %%
d_embed = 192
n_steps = 8000
batch_size = 256 # up from 128
log_every_step = min(100, n_steps//10)
save_every_step = min(1000, n_steps//10)
eval_every_step = 300

n_samples_per_bin = 1

# %%
n_samples_inverter = 100
mean_collapse_factor = 1//n_samples_per_bin

save_dir = 'eval_results/juno5/'
os.makedirs(save_dir, exist_ok=True)
# Get kernel sizes from argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--first_kernel', type=int, default=16, help='Size of the first convolutional kernel')
parser.add_argument('--second_kernel', type=int, default=8, help='Size of the second convolutional kernel')
args = parser.parse_args()
first_kernel = args.first_kernel
second_kernel = args.second_kernel

filename_base = f'{subject.subject_identifier}_{trial_id}_embed{d_embed}_nspb{n_samples_per_bin}_first_kernel{first_kernel}_second_kernel{second_kernel}'


log(f'Creating models...')
# Define a CNN model that produces a feature vector every 128 timesteps
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_features=d_embed, first_kernel=16, second_kernel=8):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(input_channels, 32, kernel_size=first_kernel, stride=first_kernel, padding=0),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv1d(32, 64, kernel_size=second_kernel, stride=second_kernel, padding=0),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Third convolutional layer
            # nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0),
            # # nn.BatchNorm1d(128),
            # nn.ReLU(),
        )
        
        # Final projection to output feature dimension
        self.projection = nn.Conv1d(64, output_features, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, n_channels, 1]
        # Reshape to [batch_size, 1, sequence_length] for 1D convolution
        x = x.unsqueeze(1).squeeze(-1).squeeze(-1)
        
        # Apply convolutional layers
        features = self.conv_layers(x)
        
        # Project to output dimension
        features = self.projection(features)
        
        # Transpose to get [batch_size, out_sequence_length, 1, output_features]
        features = features.transpose(1, 2).unsqueeze(-2)
        
        return features

# Test the CNN model with a batch
embed = CNNFeatureExtractor(first_kernel=first_kernel, second_kernel=second_kernel).to(device, dtype=dtype)
unembed = embed

model = ContrastiveModel(d_input=n_samples_per_bin, embed=embed, unembed=unembed,
                         d_model=d_embed, n_layers=6, n_heads=12, window_size=window_size//first_kernel//second_kernel).to(device, dtype=dtype)
masker = NoneMasker()

# Create samples from 10 random indices of the dataset
samples = torch.cat([dataset[random.randint(0, len(dataset)-1)]['data'].flatten() for _ in range(n_samples_inverter)])
inverter = DistributionInverter(samples=samples).to(device, dtype=dtype)

# %%
eval_tasks = ["gpt2_surprisal", "speech", "pitch"]
evaluation = ModelEvaluation_BTBench(model, inverter, [(eval_subject, eval_trial_id)], eval_tasks, feature_aggregation_method='concat', 
                                        mean_collapse_factor=mean_collapse_factor, eval_electrode_indices=[eval_electrode_index], n_samples_per_bin=n_samples_per_bin,
                                        lite=True)

log(f'Training model...')
initial_lr = 0.003
use_muon = False
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
evaluation_results = evaluation.evaluate(only_keys_containing='auroc/average')
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
        
    if step % eval_every_step == 0:
        # Add evaluation results
        evaluation_results = evaluation.evaluate(only_keys_containing='auroc/average')
        log_dict.update(evaluation_results)
        log(log_dict, indent=2)

    # Save training results to file
    if step % save_every_step == 0 or step == n_steps:
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
        
    training_logs.append(log_dict)
        
    if step == n_steps:
        break # Only process one batch per step
    step += 1

# %%



