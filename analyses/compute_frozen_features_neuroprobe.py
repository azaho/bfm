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

from evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
import evaluation.neuroprobe.config as neuroprobe_config

### PARSE MODEL DIR ###

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
parser.add_argument('--subject_id', type=str, required=True, help='Subject identifier')
parser.add_argument('--trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--eval_tasks', type=str, required=True, help='Tasks to evaluate on, comma-separated')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing frozen features')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for feature computation')
args = parser.parse_args()

model_dir = args.model_dir
model_epoch = args.model_epoch
subject_id = args.subject_id
trial_id = args.trial_id
eval_tasks = args.eval_tasks.split(",")
overwrite = args.overwrite
batch_size = args.batch_size

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
    training_setup = setup_class(all_subjects, config, verbose=True)
except (ImportError, AttributeError) as e:
    print(f"Could not load training setup '{config['training']['setup_name']}'. Are you sure the filename and the class name are the same and correspond to the parameter? Error: {str(e)}")
    exit()

log(f"Loading model...", priority=0)
training_setup.initialize_model()

log(f"Loading model weights...", priority=0)
training_setup.load_model(model_epoch)

log(f"Computing frozen features...", priority=0)
for eval_name in eval_tasks:
    save_file_path = os.path.join("runs/data", model_dir, "frozen_features_neuroprobe", f"model_epoch{model_epoch}", f"frozen_population_btbank{subject_id}_{trial_id}_{eval_name}.npy")
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if not overwrite and os.path.exists(save_file_path):
        log(f"Skipping {save_file_path} because it already exists", priority=0)
        continue
    
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name,
                                                        output_indices=False, lite=True,
                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE))

    X_all_bins = []
    y = []

    # Pass through model to get all train and test outputs
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
            model_output = training_setup.generate_frozen_features(batch)
            log(f"Computed features for batch {batch_idx+1} of {len(dataloader)}: model output shape {model_output.shape}", priority=0, indent=1)

            X_all_bins.append(model_output.float().cpu().numpy())
            y.append(batch_label.float().cpu().numpy())
            
            del batch_input, batch_label, model_output, batch
            gc.collect()
            torch.cuda.empty_cache()
   
    X_all_bins = np.concatenate(X_all_bins, axis=0) # shape: (n_dataset, feature_vector_dimension)
    y = np.concatenate(y, axis=0)

    # Save results as npy file
    np.save(save_file_path, {
        'X': X_all_bins,
        'y': y,

        'metadata': {
            'subject_id': subject.subject_identifier,
            'trial_id': trial_id,
            'eval_name': eval_name,
            'config': convert_dtypes(config),
        }
    })
    log(f"Saved results to {save_file_path}")
