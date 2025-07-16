import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from utils.muon_optimizer import Muon
from subject.dataset import load_subjects
from evaluation.neuroprobe_tasks import FrozenModelEvaluation_SS_SM
from training_setup.training_config import log, update_dir_name, update_random_seed, parse_config_from_args, get_default_config, parse_subject_trials_from_config, convert_dtypes
from torch.optim.lr_scheduler import ChainedScheduler
from training_setup.training_config import convert_dtypes, unconvert_dtypes, parse_subject_trials_from_config
from torch.utils.data import DataLoader
from training_setup.training_setup import TrainingSetup

from evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
import evaluation.neuroprobe.config as neuroprobe_config

### PARSE MODEL DIR ###

# python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --subject_id 3 --trial_id 1 --eval_tasks "onset,gpt2_surprisal" --model_epoch 100

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
parser.add_argument('--subject_id', type=str, required=True, help='Subject identifier')
parser.add_argument('--trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--eval_tasks', type=str, required=True, help='Tasks to evaluate on, comma-separated')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing key electrode analyses')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for analysis')
parser.add_argument('--analysis_type', type=str, default='attention', choices=['attention', 'activations'], 
                   help='Type of analysis to perform')
parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient processing (smaller batches, save to disk)')
args = parser.parse_args()

model_dir = args.model_dir
model_epoch = args.model_epoch
subject_id = args.subject_id
trial_id = args.trial_id
eval_tasks = args.eval_tasks.split(",")
overwrite = args.overwrite
batch_size = args.batch_size
analysis_type = args.analysis_type # making it modular for now
memory_efficient = args.memory_efficient

bins_start_before_word_onset_seconds = 0
bins_end_after_word_onset_seconds = 1.0

### LOAD CONFIG ###

# Load the checkpoint
if model_epoch < 0: model_epoch = "final"
checkpoint_path = os.path.join("runs/data", model_dir, f"model_epoch_{model_epoch}.pth")
checkpoint = torch.load(checkpoint_path)
config = unconvert_dtypes(checkpoint['config'])
log(f"Directory name: {model_dir}", priority=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['device'] = device
log(f"Using device: {device}", priority=0)

config['training']['train_subject_trials'] = ""
config['training']['eval_subject_trials'] = f"btbank{subject_id}_{trial_id}"
parse_subject_trials_from_config(config)

if 'setup_name' not in config['training']:
    config['training']['setup_name'] = "andrii0" # XXX: this is only here for backwards compatibility, can remove soon

### LOAD SUBJECTS ###

log(f"Loading subjects...", priority=0)
# all_subjects is a dictionary of subjects, with the subject identifier as the key and the subject object as the value
all_subjects = load_subjects(config['training']['train_subject_trials'], 
                             config['training']['eval_subject_trials'], config['training']['data_dtype'], 
                             cache=config['cluster']['cache_subjects'], allow_corrupted=False)
subject = all_subjects[f"btbank{subject_id}"] # we only really have one subject, so we can just get it by subject identifier

electrode_subset = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[f"btbank{subject_id}"]

### LOAD MODEL ###

# Import the training setup class dynamically based on config
try:
    setup_module = __import__(f'training_setup.{config["training"]["setup_name"].lower()}', fromlist=[config["training"]["setup_name"]])
    setup_class = getattr(setup_module, config["training"]["setup_name"])

    # Create a custom analysis training setup that inherits from the original
    class KeyElectrodeAnalysisSetup(setup_class):
        """
        Custom training setup for key electrode analysis.
        Inherits from the original training setup and overrides specific methods.
        """
        
        def __init__(self, all_subjects, config, verbose=True):
            super().__init__(all_subjects, config, verbose)
            self.analysis_type = analysis_type
            self.attention_scores = []
            self.activation_scores = []
            
        def compute_key_electrodes_attention(self, batch):
            """
            Extract attention scores from electrode transformer layers by using a custom Transformer class that returns attention matrices.
            This version preprocesses the batch using the model's fft_preprocessor and reshapes as in ElectrodeTransformer.
            """
            if hasattr(self.model, 'electrode_transformer'):
                import types
                from model.transformer_implementation import apply_rotary_emb
                # Custom CausalSelfAttention that returns attention weights
                class CausalSelfAttentionWithReturn(self.model.electrode_transformer.transformer.blocks[0].attn.__class__):
                    def forward(self, x, attention_mask=None, positions=None):
                        B, T, C = x.size()
                        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
                        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
                        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
                        if self.rope:
                            cos, sin = self.rotary(q, positions)
                            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                        q = q.transpose(1, 2)
                        k = k.transpose(1, 2)
                        v = v.transpose(1, 2)
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
                        if attention_mask is not None:
                            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
                        if self.causal and attention_mask is None:
                            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                        attention_weights = torch.softmax(scores, dim=-1)
                        attention_weights = self.dropout(attention_weights)
                        y = torch.matmul(attention_weights, v)
                        y = y.transpose(1, 2).contiguous().view(B, T, C)
                        y = self.c_proj(y)
                        return y, attention_weights
                # Custom Block that returns attention weights
                class BlockWithReturn(self.model.electrode_transformer.transformer.blocks[0].__class__):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.attn = CausalSelfAttentionWithReturn(self.attn.n_embd, self.attn.n_head, self.attn.causal, self.attn.rope, self.attn.rope_base, self.attn.dropout.p)
                        self.attn.load_state_dict(self.attn.state_dict())
                    def forward(self, x, attention_mask=None, positions=None):
                        L = self.n_layer
                        x_norm = torch.nn.functional.rms_norm(x, (x.size(-1),))
                        attn_out, attn_weights = self.attn(x_norm, attention_mask=attention_mask, positions=positions)
                        x = (2*L-1)/(2*L) * x + (1/(2*L)) * attn_out
                        x_norm2 = torch.nn.functional.rms_norm(x, (x.size(-1),))
                        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(x_norm2)
                        return x, attn_weights
                # Custom Transformer that returns all attention weights
                class TransformerWithReturn(self.model.electrode_transformer.transformer.__class__):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.blocks = torch.nn.ModuleList([BlockWithReturn(self.n_layer, self.d_model, self.n_head, self.causal, self.rope, self.rope_base, self.dropout.p) for _ in range(self.n_layer)])
                        self.embed = self.embed
                        self.output_proj = self.output_proj
                        self.dropout = self.dropout
                    def forward(self, x, attention_mask=None, positions=None, embeddings=None, strict_positions=False, stop_at_block=None):
                        batch_size, seq_len, d_input = x.shape
                        x = self.embed(x)
                        x = self.dropout(x)
                        if embeddings is not None:
                            x = x + embeddings
                        if attention_mask is None and positions is not None:
                            if strict_positions:
                                attention_mask = positions.unsqueeze(2) == positions.unsqueeze(1)
                            else:
                                attention_mask = positions.unsqueeze(2) >= positions.unsqueeze(1)
                        all_attn_weights = []
                        for block_i, block in enumerate(self.blocks):
                            x, attn_weights = block(x, attention_mask=attention_mask, positions=positions)
                            all_attn_weights.append(attn_weights)
                            if stop_at_block is not None and block_i+1 == stop_at_block:
                                return x, all_attn_weights
                        x = self.output_proj(x)
                        return x, all_attn_weights
                # Instantiate custom transformer and load weights
                transformer = TransformerWithReturn(self.model.electrode_transformer.transformer.d_input, self.model.electrode_transformer.transformer.d_model, self.model.electrode_transformer.transformer.d_output, self.model.electrode_transformer.transformer.n_layer, self.model.electrode_transformer.transformer.n_head, self.model.electrode_transformer.transformer.causal, self.model.electrode_transformer.transformer.rope, self.model.electrode_transformer.transformer.rope_base, self.model.electrode_transformer.transformer.dropout.p)
                transformer.load_state_dict(self.model.electrode_transformer.transformer.state_dict())
                # === Preprocess input as in model ===
                # 1. Run through fft_preprocessor
                x = self.model.fft_preprocessor(batch)  # (batch, n_electrodes, n_timebins, d_model)
                batch_size, n_electrodes, n_timebins, d_model = x.shape
                # 2. Reshape as in ElectrodeTransformer
                x = x.transpose(1, 2).reshape(batch_size * n_timebins, n_electrodes, d_model)
                # 3. Add CLS token as in ElectrodeTransformer
                cls_token = self.model.electrode_transformer.cls_token.unsqueeze(0).repeat(batch_size * n_timebins, 1, 1)
                x = torch.cat([cls_token, x], dim=1)  # (batch*n_timebins, n_electrodes+1, d_model)
                # 4. Move to correct device
                device = next(transformer.parameters()).device
                x = x.to(device)
                # === Forward pass ===
                x, all_attn_weights = transformer(x)
                # Print attention info
                for i, attn in enumerate(all_attn_weights):
                    # print(f"Block {i} attention shape: {attn.shape}")
                    if attn.shape[0] > 0 and attn.shape[1] > 0:
                        first_head_attn = attn[0, 0].cpu().numpy()
                        # print(f"First head attention matrix (first 5x5):\n{first_head_attn[:5, :5]}")
                return all_attn_weights
            else:
                log("Model does not have electrode_transformer attribute", priority=1)
                return None

    # Create the analysis training setup
    training_setup = KeyElectrodeAnalysisSetup(all_subjects, config, verbose=True)
    
except (ImportError, AttributeError) as e:
    print(f"Could not load training setup '{config['training']['setup_name']}'. Are you sure the filename and the class name are the same and correspond to the parameter? Error: {str(e)}")
    exit()

log(f"Loading model...", priority=0)
training_setup.initialize_model()

log(f"Loading model weights...", priority=0)
training_setup.load_model(model_epoch)

log(f"Computing key electrode analysis...", priority=0)
for eval_name in eval_tasks:
    save_file_path = os.path.join("runs/data", model_dir, "key_electrodes", f"model_epoch{model_epoch}", 
                                 f"key_electrodes_btbank{subject_id}_{trial_id}_{eval_name}_{analysis_type}.npy")
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if not overwrite and os.path.exists(save_file_path):
        log(f"Skipping {save_file_path} because it already exists", priority=0)
        continue
    
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name,
                                                        output_indices=False, lite=True,
                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE))

    # Create a temporary directory for streaming results
    temp_dir = os.path.join(os.path.dirname(save_file_path), "temp_attention")
    os.makedirs(temp_dir, exist_ok=True)
    
    batch_files = []  # Keep track of saved batch files

    # Pass through model to get analysis results
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, (batch_input, batch_label) in enumerate(dataloader):
            batch = {
                'data': batch_input.to(device, dtype=config['training']['data_dtype']),
                'electrode_labels': [electrode_subset],
                'metadata': {
                    'subject_identifier': subject.subject_identifier,
                    'trial_id': trial_id,
                    'eval_name': eval_name,
                }
            }
            
            # Apply preprocessing
            for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
                batch = preprocess_function(batch)
            
            # Perform analysis
            analysis_output = training_setup.compute_key_electrodes_attention(batch)
            log(f"Computed {analysis_type} analysis for batch {batch_idx+1} of {len(dataloader)}", priority=0, indent=1)

            if analysis_output is not None:
                # Save attention matrices to disk immediately and delete from memory
                batch_file = os.path.join(temp_dir, f"batch_{batch_idx:04d}.npy")
                
                if isinstance(analysis_output, torch.Tensor):
                    # Convert to numpy and save immediately
                    numpy_result = analysis_output.float().cpu().numpy()
                    np.save(batch_file, numpy_result)
                    del analysis_output, numpy_result
                elif isinstance(analysis_output, list):
                    # Handle list of tensors (e.g., attention scores from multiple layers)
                    numpy_results = []
                    for t in analysis_output:
                        if isinstance(t, torch.Tensor):
                            # Average across first two dimensions (sequences and heads)
                            # Shape: (n_seq, n_heads, seq_len, seq_len) -> (seq_len, seq_len)
                            averaged_t = t.float().cpu().numpy().mean(axis=(0, 1))
                            numpy_results.append(averaged_t)
                            del t  # Delete tensor immediately
                        else:
                            numpy_results.append(t)
                    np.save(batch_file, numpy_results)
                    del analysis_output, numpy_results
                else:
                    np.save(batch_file, analysis_output)
                    del analysis_output
                
                batch_files.append(batch_file)
                log(f"Saved batch {batch_idx+1} attention to {batch_file}", priority=0)
            
            # Clean up all batch-related memory
            del batch_input, batch_label, batch
            gc.collect()
            torch.cuda.empty_cache()
   
    # Combine all batch files into final result
    if batch_files:
        log(f"Combining {len(batch_files)} batch files into final result...", priority=0)
        combined_results = []
        
        for batch_file in batch_files:
            batch_data = np.load(batch_file, allow_pickle=True)
            combined_results.append(batch_data)
            # Clean up individual batch file
            os.remove(batch_file)
        
        # Save combined results
        np.save(save_file_path, {
            'analysis_results': combined_results,
            'analysis_type': analysis_type,
            'metadata': {
                'subject_id': subject.subject_identifier,
                'trial_id': trial_id,
                'eval_name': eval_name,
                'config': convert_dtypes(config),
            }
        })
        log(f"Saved {analysis_type} analysis results to {save_file_path}")
        
        # Clean up temp directory
        os.rmdir(temp_dir)
    else:
        log(f"No analysis results to save for {eval_name}", priority=1)
