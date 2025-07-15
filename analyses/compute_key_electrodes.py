import torch
import wandb, os, json
import time
import numpy as np
from torch.amp import autocast
import gc

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
args = parser.parse_args()

model_dir = args.model_dir
model_epoch = args.model_epoch
subject_id = args.subject_id
trial_id = args.trial_id
eval_tasks = args.eval_tasks.split(",")
overwrite = args.overwrite
batch_size = args.batch_size
analysis_type = args.analysis_type # making it modular for now

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
            Extract attention scores from electrode transformer layers.
            Override this method based on your specific model architecture.
            """
            # This is a placeholder - you'll need to implement based on your model
            if hasattr(self.model, 'electrode_transformer'):
                # Example: extract attention from transformer layers
                attention_scores = []
                for block in self.model.electrode_transformer.blocks:
                    # Accessing q, k, v directly and compute attention manually
                    q = block.attn.c_q(x)
                    k = block.attn.c_k(x) 
                    v = block.attn.c_v(x)
                    attention_scores.append(torch.softmax(q @ k.T / sqrt(d_k), dim=-1))
                return attention_scores
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

    analysis_results = []

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
                # Convert to numpy and store
                if isinstance(analysis_output, torch.Tensor):
                    analysis_results.append(analysis_output.float().cpu().numpy())
                elif isinstance(analysis_output, list):
                    # Handle list of tensors (e.g., attention scores from multiple layers)
                    analysis_results.append([t.float().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in analysis_output])
                else:
                    analysis_results.append(analysis_output)
            
            del batch_input, batch_label, analysis_output, batch
            gc.collect()
            torch.cuda.empty_cache()
   
    # Save results as npy file
    if analysis_results:
        np.save(save_file_path, {
            'analysis_results': analysis_results,
            'analysis_type': analysis_type,
            'metadata': {
                'subject_id': subject.subject_identifier,
                'trial_id': trial_id,
                'eval_name': eval_name,
                'config': convert_dtypes(config),
            }
        })
        log(f"Saved {analysis_type} analysis results to {save_file_path}")
    else:
        log(f"No analysis results to save for {eval_name}", priority=1)
