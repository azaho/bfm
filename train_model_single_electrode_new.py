from dataset import SubjectTrialDataset_SingleElectrode
from subject_braintreebank import BrainTreebankSubject
from subject_ajile12 import AjileSubject
import torch
from model_model import BFModule
from model_transformers import Transformer
import torch.nn as nn
from train_utils import log
import random, os



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

n_samples_per_bin = 1
n_samples_inverter = 100
mean_collapse_factor = 128//n_samples_per_bin

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description='Train model for single electrode')
parser.add_argument('--d_hilbert', type=int, default=4, help='Dimension of Hilbert space')
parser.add_argument('--d_embed', type=int, default=128, help='Dimension of embedding')
parser.add_argument('--bits_per_sample', type=int, default=3, help='Number of bins per sample')
parser.add_argument('--n_steps', type=int, default=3000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--save_dir', type=str, default='eval_results/single_electrode_new/', help='Directory to save evaluation results')

# Parse arguments
args = parser.parse_args()
d_hilbert = args.d_hilbert
d_embed = args.d_embed
bits_per_sample = args.bits_per_sample
n_steps = args.n_steps
batch_size = args.batch_size
log_every_step = min(300, n_steps//10)

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

filename = f'{subject.subject_identifier}_{trial_id}_embed{d_embed}_hilbert{d_hilbert}_bits{bits_per_sample}.json'


class DistributionInverter(BFModule):
    def __init__(self, samples, eps=1e-5):
        super().__init__()
        self.samples = samples
        self.mean = nn.Parameter(samples.mean(dim=0).view(1, 1, -1))
        self.std = nn.Parameter(samples.std(dim=0).view(1, 1, -1))
        self.eps = eps

    def forward(self, x):
        # x is of shape (batch_size, seq_len, n_channels)
        # samples is of shape (n_samples, n_channels)
        return (x - self.mean) / (self.std + self.eps)
        # return torch.sum(self.samples.unsqueeze(0).unsqueeze(0) <= x.unsqueeze(-1), dim=-1) / len(self.samples)

class EmbedderInterpolation(BFModule):
    def __init__(self, d_model, resolution=10, range=(-3, 3)):
        super().__init__()
        self.d_model = d_model
        self.resolution = resolution
        self.range = range
        self.delta = (range[1] - range[0]) / (resolution-1)

        self.embedding_centers = torch.linspace(range[0], range[1], resolution)
        self.embedding_weights = torch.nn.Parameter(torch.randn(resolution, d_model) / d_model)
        
    def forward(self, x):
        # x is of shape (batch_size, seq_len, n_channels, d_input)
        # output is of shape (batch_size, seq_len, n_channels, d_model)
        batch_size, seq_len, n_channels, d_input = x.shape
        assert d_input == 1, "d_input must be 1, EmbedderInterpolation only supports single input channel"
        x = x.squeeze(-1)

        output = torch.zeros(*x.shape, self.d_model, device=x.device, dtype=x.dtype)
        for center_i, embedding_center in enumerate(self.embedding_centers):
            distance_multiplier = torch.maximum(1 - torch.abs(x - embedding_center) / self.delta, torch.tensor(0))
            output += distance_multiplier.unsqueeze(-1) * self.embedding_weights[center_i].view(1, 1, 1, -1)
        output[x<self.range[0]] = self.embedding_weights[0]
        output[x>self.range[1]] = self.embedding_weights[-1]
        return output

class EmbedderLinear(BFModule):
    def __init__(self, d_model, d_input):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.linear_embedding = nn.Linear(d_input, d_model)

    def forward(self, x):
        return self.linear_embedding(x)

from hilbert_decode import hilbert_decode
class EmbedderHilbert(BFModule):
    def __init__(self, d_model, d_hilbert=4, d_input=1, resolution=3, range=(-3, 3)):
        super().__init__()
        self.d_model = d_model
        self.d_hilbert = d_hilbert
        self.d_input = d_input
        self.linear_projection = nn.Linear(d_hilbert, d_model, bias=False) if d_model > 0 else lambda x: x
        self.bits_per_dimension = resolution
        self.range = range

    def forward(self, x):
        # x is of shape (batch_size, seq_len, n_channels, d_input)
        # output is of shape (batch_size, seq_len, n_channels, d_model)
        batch_size, seq_len, n_channels, d_input = x.shape
        assert d_input == 1, "d_input must be 1, EmbedderHilbert only supports single input channel"
        x = x.squeeze(-1)

        # Clamp to range first, then normalize to [0, 1]
        x = torch.clamp(x, self.range[0], self.range[1])
        x = (x - self.range[0]) / (self.range[1] - self.range[0])

        max_h = 2**(self.d_hilbert*self.bits_per_dimension)
        x = hilbert_decode(x * max_h, self.d_hilbert, self.bits_per_dimension).to(x.device, x.dtype) / max_h**0.5
        return self.linear_projection(x)

class Model(BFModule):
    def __init__():
        pass

class ContrastiveModel(Model):
    def __init__(self, d_model=128, n_layers=4, n_heads=8, d_input=1):
        super(Model, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, 
                                        n_layer=n_layers, n_head=n_heads, causal=True, 
                                        rope=True, cls_token=False, rope_base=window_size*2 // d_input)
        self.embed = EmbedderHilbert(d_model=d_model, d_hilbert=d_hilbert, d_input=d_input, resolution=bits_per_sample)
        self.unembed = EmbedderHilbert(d_model=d_model, d_hilbert=d_hilbert, d_input=d_input, resolution=bits_per_sample)

        self.mask_token = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x, y=None, mask=None):
        # x is of shape (batch_size, seq_len, n_channels, d_input) 
        # y is of shape (batch_size, seq_len, n_channels, d_input)
        batch_size, seq_len, n_channels, d_input = x.shape

        x = self.embed(x)# shape (batch_size, seq_len, n_channels, d_model)
        if mask is not None:
            x[:, mask, :] = self.mask_token.view(1, 1, 1, -1)
        
        positions = torch.arange(seq_len).repeat(1, n_channels).flatten().to(self.device)
        x = self.transformer(x.reshape(batch_size, seq_len*n_channels, self.d_model), positions=positions) # shape (batch_size, seq_len, d_model)
        x = x.reshape(batch_size, seq_len, n_channels, self.d_model)

        if y is not None:
            y = self.unembed(y) # shape (batch_size, seq_len, n_channels, d_model)
            return x, y
        return x
        
    def calculate_loss(self, batch, mask=None):
        # batch is of shape (batch_size, seq_len, n_channels, d_input)
        if mask is not None:
            x, y = self(batch, batch, mask=mask) # shape (batch_size, seq_len, n_channels, d_model)
        else:
            x, y = self(batch[:, :-1], batch[:, 1:]) # shape (batch_size, seq_len, n_channels, d_model)
        batch_size = batch.shape[0]

        if mask is not None:
            x = x[:, mask, :]
            y = y[:, mask, :]
        else:
            batch_size, seq_len, n_channels, d_model = x.shape
            x = x.reshape(batch_size, seq_len*n_channels, d_model)
            y = y.reshape(batch_size, seq_len*n_channels, d_model)
        # x shape: (batch_size, num_masked_intervals or seq_len*n_channels, d_model)
        # y shape: (batch_size, num_masked_intervals or seq_len*n_channels, d_model)

        similarity = torch.matmul(x[:, :].permute(1, 0, 2), y[:, :].permute(1, 2, 0))
        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(x.shape[1], 1).to(device, dtype=torch.long).reshape(-1)
        loss = torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
        return loss



import sklearn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import numpy as np

from btbench.btbench_train_test_splits import generate_splits_SS_SM
class ModelEvaluation_BTBench():
    def __init__(self, model, inverter, subject_trials, eval_names, batch_size=128, dtype=torch.float32, feature_aggregation_method='mean', mean_collapse_factor=1, start_neural_data_before_word_onset=0, end_neural_data_after_word_onset=2048, eval_electrode_index=eval_electrode_index):
        self.model = model
        self.inverter = inverter
        self.subject_trials = subject_trials
        self.eval_names = eval_names
        self.batch_size = batch_size
        self.dtype = dtype
        self.feature_aggregation_method = feature_aggregation_method
        self.mean_collapse_factor = mean_collapse_factor
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset
        self.eval_electrode_index = eval_electrode_index
        
        # Create evaluation datasets
        self.evaluation_datasets = {}
        for eval_name in self.eval_names:
            for subject, trial_id in self.subject_trials:
                splits = generate_splits_SS_SM(subject, trial_id, eval_name, dtype=self.dtype,
                                               start_neural_data_before_word_onset=self.start_neural_data_before_word_onset,
                                               end_neural_data_after_word_onset=self.end_neural_data_after_word_onset)
                self.evaluation_datasets[(eval_name, subject.subject_identifier, trial_id)] = splits

    def evaluate_on_dataset(self, train_dataset, test_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        device = next(self.model.parameters()).device
        
        # Get embeddings for train and test data
        X_train, y_train = [], []
        for batch_input, batch_label in train_dataloader:
            batch_input = batch_input.to(device, dtype=self.dtype)
            batch_size = batch_input.shape[0]
            window_size = batch_input.shape[2]

            batch_input = batch_input[:, self.eval_electrode_index] # select just the first channel

            batch_data = batch_input.to(device, dtype=dtype).reshape(batch_size, window_size//n_samples_per_bin, 1, n_samples_per_bin) # shape (batch_size, seq_len, 1)
            batch_data = self.inverter(batch_data)

            with torch.no_grad():
                x_embed = self.model(batch_data)
            
            batch_size, seq_len, _, d_model = x_embed.shape
            x_embed = x_embed.reshape(batch_size, seq_len//self.mean_collapse_factor, self.mean_collapse_factor, d_model)
            x_embed = x_embed.mean(dim=2)

            if self.feature_aggregation_method == 'mean':
                X_train.append(x_embed.mean(dim=1).cpu().float().numpy())  # Average across sequence length
            elif self.feature_aggregation_method == 'concat':
                X_train.append(x_embed.reshape(batch_size, -1).cpu().float().numpy())  # Concatenate across sequence length
            y_train.append(batch_label.numpy())        
        X_test, y_test = [], []
        for batch_input, batch_label in test_dataloader:
            batch_input = batch_input.to(device, dtype=self.dtype)
            batch_size = batch_input.shape[0]
            window_size = batch_input.shape[2]

            batch_input = batch_input[:, self.eval_electrode_index] # select just the first channel

            batch_data = batch_input.to(device, dtype=dtype).reshape(batch_size, window_size//n_samples_per_bin, 1, n_samples_per_bin) # shape (batch_size, seq_len, 1)
            batch_data = self.inverter(batch_data)

            with torch.no_grad():
                x_embed = self.model(batch_data)
            batch_size, seq_len, _, d_model = x_embed.shape
            x_embed = x_embed.reshape(batch_size, seq_len//self.mean_collapse_factor, self.mean_collapse_factor, d_model)
            x_embed = x_embed.mean(dim=2)

            if self.feature_aggregation_method == 'mean':
                X_test.append(x_embed.mean(dim=1).cpu().float().numpy())
            elif self.feature_aggregation_method == 'concat':
                X_test.append(x_embed.reshape(batch_size, -1).cpu().float().numpy())
            y_test.append(batch_label.numpy())

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

        # Train logistic regression classifier
        clf = LogisticRegression(random_state=42, max_iter=10000)
        clf.fit(X_train, y_train)

        # Get predictions and calculate metrics
        test_probs = clf.predict_proba(X_test)
        accuracy = clf.score(X_test, y_test)
        
        # Calculate AUROC
        y_test_onehot = np.zeros((len(y_test), len(clf.classes_)))
        for i, label in enumerate(y_test):
            class_idx = np.where(clf.classes_ == label)[0][0]
            y_test_onehot[i, class_idx] = 1
            
        if len(clf.classes_) > 2:
            auroc = sklearn.metrics.roc_auc_score(y_test_onehot, test_probs, multi_class='ovr', average='macro')
        else:
            auroc = sklearn.metrics.roc_auc_score(y_test_onehot, test_probs)
            
        return auroc, accuracy

    def evaluate(self, return_raw=False):
        results = {}
        for subject in set(subject for subject, _ in self.subject_trials):
            for eval_name in self.eval_names:
                trial_ids = [trial_id for _subject, trial_id in self.subject_trials if _subject == subject]
                for trial_id in trial_ids:
                    splits = self.evaluation_datasets[(eval_name, subject.subject_identifier, trial_id)]
                    auroc_list, acc_list = [], []
                    for train_dataset, test_dataset in zip(splits[0], splits[1]):
                        auroc, acc = self.evaluate_on_dataset(train_dataset, test_dataset)
                        auroc_list.append(auroc)
                        acc_list.append(acc)
                    
                    mean_auroc = np.mean(auroc_list)
                    mean_acc = np.mean(acc_list)
                    results[(eval_name, subject.subject_identifier, trial_id)] = (mean_auroc, mean_acc) if not return_raw else (auroc_list, acc_list)
                    
        return self._format_results(results) if not return_raw else results
        
    def _format_results(self, results):
        formatted_results = {}
        for eval_name in self.eval_names:
            auroc_values = []
            acc_values = []
            subject_aurocs = {}
            subject_accs = {}
            
            for (metric, subject_id, trial_id) in [k for k in results.keys() if k[0] == eval_name]:
                if subject_id not in subject_aurocs:
                    subject_aurocs[subject_id] = []
                    subject_accs[subject_id] = []
                    
                auroc, acc = results[(eval_name, subject_id, trial_id)]
                subject_aurocs[subject_id].append(auroc)
                subject_accs[subject_id].append(acc)
                
                formatted_results[f"eval_auroc/{subject_id}_{trial_id}_{eval_name}"] = auroc
                formatted_results[f"eval_acc/{subject_id}_{trial_id}_{eval_name}"] = acc
                
            for subject_id in subject_aurocs:
                auroc_values.append(np.mean(subject_aurocs[subject_id]))
                acc_values.append(np.mean(subject_accs[subject_id]))
                
            if auroc_values:
                formatted_results[f"eval_auroc/average_{eval_name}"] = np.mean(auroc_values)
                formatted_results[f"eval_acc/average_{eval_name}"] = np.mean(acc_values)
                
        return formatted_results

class NoneMasker(BFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x, None



log(f'Creating models...')
import itertools
dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
dataloader = iter(itertools.cycle(dataloader))

model = ContrastiveModel(d_input=n_samples_per_bin).to(device, dtype=dtype)
masker = NoneMasker()

# Create samples from 10 random indices of the dataset
samples = torch.cat([dataset[random.randint(0, len(dataset)-1)]['data'].flatten() for _ in range(n_samples_inverter)])
inverter = DistributionInverter(samples=samples).to(device, dtype=dtype)

evaluation = ModelEvaluation_BTBench(model, inverter, [(subject, 0)], ["gpt2_surprisal", "onset"], feature_aggregation_method='concat', mean_collapse_factor=mean_collapse_factor)



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
        # Convert training logs to a format that can be saved as JSON
        import json
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'w') as f:
            json.dump(training_logs, f, indent=2)
        log(f"Saved training logs to {save_path}")
        
    if step == n_steps:
        break # Only process one batch per step
    step += 1
