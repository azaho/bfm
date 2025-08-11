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
from model.custom_attention_modules import (
    CausalSelfAttentionWithReturn,
    BlockWithReturn,
    TransformerWithReturn,
)

from evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
import evaluation.neuroprobe.config as neuroprobe_config

### PARSE MODEL DIR ###

# python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --subject_id 3 --trial_id 1 --model_epoch 100 --overwrite

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
parser.add_argument('--subject_id', type=str, help='Subject identifier (use --all_subjects to process all)')
parser.add_argument('--trial_id', type=int, help='Trial identifier (use --all_subjects to process all)')
parser.add_argument('--all_subjects', action='store_true', help='Process all subjects and trials from NEUROPROBE_FULL_SUBJECT_TRIALS')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing key electrode analyses')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for analysis')
parser.add_argument('--analysis_type', type=str, default='attention', choices=['attention', 'activations'], 
                   help='Type of analysis to perform')
parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient processing (smaller batches, save to disk)')
args = parser.parse_args()

# Validate arguments
if not args.all_subjects and (args.subject_id is None or args.trial_id is None):
    parser.error("Either --all_subjects or both --subject_id and --trial_id must be specified")

model_dir = args.model_dir
model_epoch = args.model_epoch
eval_name = "onset" # just need something to pass in
overwrite = args.overwrite
batch_size = args.batch_size
analysis_type = args.analysis_type # making it modular for now
memory_efficient = args.memory_efficient

# Determine which subjects/trials to process
if args.all_subjects:
    subject_trials = neuroprobe_config.NEUROPROBE_FULL_SUBJECT_TRIALS
    log(f"Processing all {len(subject_trials)} subject/trial pairs from NEUROPROBE_FULL_SUBJECT_TRIALS", priority=0)
else:
    subject_trials = [(args.subject_id, args.trial_id)]
    log(f"Processing single subject/trial pair: {args.subject_id}, {args.trial_id}", priority=0)

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
config['training']['eval_subject_trials'] = ""  # Will be set inside the loop
parse_subject_trials_from_config(config)

if 'setup_name' not in config['training']:
    config['training']['setup_name'] = "andrii0" # XXX: this is only here for backwards compatibility, can remove soon

### PROCESS EACH SUBJECT/TRIAL PAIR ###
for subject_id, trial_id in subject_trials:
    log(f"Processing subject {subject_id}, trial {trial_id}...", priority=0)
    
    # Update config for this specific subject/trial
    config['training']['eval_subject_trials'] = f"btbank{subject_id}_{trial_id}"
    parse_subject_trials_from_config(config)  # Re-parse with the new subject/trial
    
    ### LOAD SUBJECTS ###
    log(f"Loading subjects...", priority=0)
    # all_subjects is a dictionary of subjects, with the subject identifier as the key and the subject object as the value
    all_subjects = load_subjects(config['training']['train_subject_trials'], 
                                 config['training']['eval_subject_trials'], config['training']['data_dtype'], 
                                 cache=config['cluster']['cache_subjects'], allow_corrupted=False)
    subject = all_subjects[f"btbank{subject_id}"] # we only really have one subject, so we can just get it by subject identifier

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
                
            def _preprocess_add_electrode_indices(self, batch):
                """
                Override to handle electrodes that don't exist in the embeddings map.
                For new subjects, we'll assign sequential indices.
                """
                electrode_indices = []
                subject_identifier = batch['metadata']['subject_identifier']
                
                # Get the current size of embeddings map to assign new indices
                current_max_index = max(self.electrode_embeddings.embeddings_map.values()) if self.electrode_embeddings.embeddings_map else -1
                
                for electrode_label in batch['electrode_labels'][0]:
                    key = (subject_identifier, electrode_label)
                    if key in self.electrode_embeddings.embeddings_map:
                        electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
                    else:
                        # For new electrodes, assign the next available index
                        current_max_index += 1
                        electrode_indices.append(current_max_index)
                        # Add to embeddings map
                        self.electrode_embeddings.embeddings_map[key] = current_max_index
                
                batch['electrode_index'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1)
                return batch
                
            def compute_key_electrodes_attention(self, batch):
                """
                Extract attention scores from electrode transformer layers by using a custom Transformer class that returns attention matrices.
                This version preprocesses the batch using the model's fft_preprocessor and reshapes as in ElectrodeTransformer.
                """
                if hasattr(self.model, 'electrode_transformer'):
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
                    return all_attn_weights
                else:
                    log("Model does not have electrode_transformer attribute", priority=1)
                    return None

        # Create the analysis training setup for this subject
        training_setup = KeyElectrodeAnalysisSetup(all_subjects, config, verbose=True)
        
    except (ImportError, AttributeError) as e:
        print(f"Could not load training setup '{config['training']['setup_name']}'. Are you sure the filename and the class name are the same and correspond to the parameter? Error: {str(e)}")
        exit()

    log(f"Loading model...", priority=0)
    training_setup.initialize_model()

    log(f"Loading model weights...", priority=0)
    training_setup.load_model(model_epoch)

    log(f"Computing key electrode analysis...", priority=0)

    save_file_path = os.path.join("runs/data", model_dir, "key_electrodes", f"model_epoch{model_epoch}", 
                                    f"key_electrodes_btbank{subject_id}_{trial_id}_{eval_name}_{analysis_type}.npy")
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if not overwrite and os.path.exists(save_file_path):
        log(f"Skipping {save_file_path} because it already exists", priority=0)
        continue

    # Check if this subject is in the lite electrodes list
    subject_identifier = f"btbank{subject_id}"
    if subject_identifier not in neuroprobe_config.NEUROPROBE_LITE_ELECTRODES:
        log(f"Skipping subject {subject_identifier} because it's not in lite electrodes", priority=0)
        continue

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name,
                                                        output_indices=False, lite=True,
                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE))

    # Get the actual electrodes that the dataset is using
    dataset_electrode_labels = [subject.electrode_labels[i] for i in dataset.electrode_indices_subset]
    print(f"Dataset using {len(dataset_electrode_labels)} electrodes")
    print(f"Dataset electrode subset: {dataset_electrode_labels[:5]}...")

    # Use the dataset's electrode subset - the model should handle new electrodes through embeddings
    electrode_subset = dataset_electrode_labels
    print(f"Using all {len(electrode_subset)} electrodes from dataset")

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

                    # preserve ram by deleting tensors if no other layers
                    del analysis_output, numpy_result
                elif isinstance(analysis_output, list):
                    # Handle list of tensors (e.g., attention scores from multiple layers)
                    numpy_results = []
                    for layer_idx, t in enumerate(analysis_output):
                        if isinstance(t, torch.Tensor):
                            # Average across first two dimensions (sequences and heads)
                            # Shape: (n_seq, n_heads, n_electrodes+1, n_electrodes+1) -> (n_electrodes+1, n_electrodes+1)
                            averaged_t = t.float().cpu().numpy().mean(axis=(0, 1))
                            numpy_results.append(averaged_t)
                            
                            # Create and save electrode attention heatmap for this layer
                            electrode_matrix = averaged_t # Not removing CLS token for now
                            plt.figure(figsize=(12, 10))
                            seq_len = electrode_matrix.shape[0]
                            print(f"electrode_subset ({len(electrode_subset)}): {electrode_subset}")
                            labels = ["CLS"] + list(electrode_subset)
                            print(labels)
                            
                            sns.heatmap(electrode_matrix,
                                        xticklabels=labels,
                                        yticklabels=labels,
                                        cmap='viridis',
                                        annot=False,
                                        fmt='.3f',
                                        cbar_kws={'label': 'Attention Score'},
                                        square=True)
                            
                            plt.title(f'Electrode Attention - Layer {layer_idx}, Batch {batch_idx+1}')
                            plt.xlabel('Key Electrode')
                            plt.ylabel('Query Electrode')
                            
                            # Save the plot
                            plot_dir = os.path.join(os.path.dirname(save_file_path), "electrode_plots")
                            os.makedirs(plot_dir, exist_ok=True)
                            plot_path = os.path.join(plot_dir, f"electrode_attention_layer{layer_idx}_batch{batch_idx+1}.png")
                            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                            plt.close()  # Close to free memory
                            log(f"Saved electrode attention plot to {plot_path}", priority=0)
                            
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
            'electrode_labels': ["CLS"] + list(electrode_subset),
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
