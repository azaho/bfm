import gc
import json
import os
import time

import numpy as np
import torch
import wandb
from torch.amp import autocast

from evaluation.neuroprobe import config as neuroprobe_config
from evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
from evaluation.neuroprobe_tasks import FrozenModelEvaluation_SS_SM
from subject.dataset import load_subjects
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import ChainedScheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from training_setup.training_config import (
    convert_dtypes,
    get_default_config,
    log,
    parse_config_from_args,
    parse_subject_trials_from_config,
    unconvert_dtypes,
    update_dir_name,
    update_random_seed,
)
from utils.muon_optimizer import Muon
# torch.set_float32_matmul_precision('high')

RUNS_DIR='runs/data'

### PARSE MODEL DIR ###

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
parser.add_argument('--train_subject_id', type=str, required=True, help='Subject identifier')
parser.add_argument('--train_trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--test_subject_id', type=str, required=True, help='Subject identifier')
parser.add_argument('--test_trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--eval_tasks', type=str, required=True, help='Tasks to evaluate on, comma-separated')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing frozen features')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for feature computation')
parser.add_argument('--finetuning_learning_rate', type=float, default=0.003, help='Learning rate for finetuning')
parser.add_argument('--finetuning_epochs', type=int, default=100, help='Number of epochs to train')
args = parser.parse_args()

model_dir = args.model_dir
model_epoch = args.model_epoch if args.model_epoch >= 0 else "final"
train_subject_id = args.train_subject_id
train_trial_id = args.train_trial_id
test_subject_id = args.test_subject_id
test_trial_id = args.test_trial_id
eval_tasks = args.eval_tasks.split(",")
overwrite = args.overwrite
batch_size = args.batch_size
finetuning_learning_rate = args.finetuning_learning_rate
finetuning_epochs = args.finetuning_epochs

bins_start_before_word_onset_seconds = 0
bins_end_after_word_onset_seconds = 1.0


### LOAD CONFIG ###

# Load the checkpoint
if model_epoch < 0: model_epoch = "final"

ckpt_dir = os.path.join(RUNS_DIR, model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(ckpt_dir, f"model_epoch_{model_epoch}.pth")
checkpoint = torch.load(checkpoint_path, map_location=device if torch.cuda.is_available() else torch.device('cpu'))

config = unconvert_dtypes(checkpoint['config'])
log(f"Directory name: {model_dir}", priority=0)

config['device'] = device
log(f"Using device: {device}", priority=0)

config['training']['train_subject_trials'] = f"btbank{train_subject_id}_{train_trial_id}"
config['training']['eval_subject_trials'] = f"btbank{test_subject_id}_{test_trial_id}"
parse_subject_trials_from_config(config)

if 'setup_name' not in config['training']:
    config['training']['setup_name'] = "andrii0" # XXX: this is only here for backwards compatibility, can remove soon

if 'signal_preprocessing' in config['model'] and 'spectrogram_parameters' not in config['model']['signal_preprocessing']:
    config['model']['signal_preprocessing']['spectrogram_parameters'] = {
    'max_frequency': config.get('spectrogram_max_frequency'),
    'time_bin_size': config.get('time_bin_size')
    }
# XXX: this is only here for backwards compatibility, can remove soon


### LOAD SUBJECTS ###

log(f"Loading subjects...", priority=0)
# all_subjects is a dictionary of subjects, with the subject identifier as the key and the subject object as the value
all_subjects = load_subjects(config['training']['train_subject_trials'],
                             config['training']['eval_subject_trials'],
                             config['training']['data_dtype'],
                             cache=config['cluster']['cache_subjects'],
                             allow_corrupted=False)

train_subject = all_subjects[f"btbank{train_subject_id}"]
test_subject = all_subjects[f"btbank{test_subject_id}"]

electrode_subset_train = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[f"btbank{train_subject_id}"]
electrode_subset_test = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[f"btbank{test_subject_id}"]


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

# andrii0_lr0.003_wd0.0_dr0.1_rTEMP_t20250812_133427

# Step 0. Define the linear layer

# Step 0.1: Define the train and test split

# Step 0.5 Parameters
#   - learning rate - this will be separate from the config
#   - number of epochs - this will be separate from the config
#   - batch size - can come from the config
#   - optimizer - can come from the config
#   - loss function - you will define your own, logistic regression loss

# Step 1. Set up the optimizer
# Step 2. Loop over the number of fine-tuning epochs
# Step 3. For each epoch, loop over the number of batches
# Step 4. For each batch, compute the features (NOTE: avoid the torch.no_grad() context manager)
# Step 5. Pass the features through the linear layer to get the predictions
# Step 6. Compute the loss (logistic regression loss)
# Step 7. Backpropagate the loss - loss.backward()
# Step 8. Update the weights - optimizer.step()
# Step 9. Log the loss - wandb.log()
# Step 10. Save the model - torch.save()

# What to save?
# - model weights every fine tuning epoch
# - loss every fine tuning epoch (train and test)
# - AUROC and accuracy on the test set every fine tuning epoch

# What do we want? A plot of train and test loss over the course of finetuning. + train and test auccracy and AUROC over the course of finetuning.
loss_fn = nn.BCEWithLogitsLoss()

for eval_name in eval_tasks:

    wandb.init(
        project="neuroprobe-finetuning",
        config={
            "model_dir": model_dir,
            "model_epoch": model_epoch,
            "train_subject": train_subject_id,
            "train_trial": train_trial_id,
            "test_subject": test_subject_id,
            "test_trial": test_trial_id,
            "task": eval_name,
            "learning_rate": finetuning_learning_rate,
            "epochs": finetuning_epochs,
            "batch_size": batch_size,
        },
        name=f"{model_dir}_ft_{eval_name}_t{train_subject_id}_{train_trial_id}_e{test_subject_id}_{test_trial_id}"
    )

    start_offset = int(bins_start_before_word_onset_seconds * neuroprobe_config.SAMPLING_RATE)
    end_offset = int(bins_end_after_word_onset_seconds * neuroprobe_config.SAMPLING_RATE)

    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        train_subject, train_trial_id, dtype=torch.float32, eval_name=eval_name,
        output_indices=False, lite=True,
        start_neural_data_before_word_onset=start_offset,
        end_neural_data_after_word_onset=end_offset,
    )
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        test_subject, test_trial_id, dtype=torch.float32, eval_name=eval_name,
        output_indices=False, lite=True,
        start_neural_data_before_word_onset=start_offset,
        end_neural_data_after_word_onset=end_offset,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    linear_head = nn.Linear(config['model']['transformer']['d_model'], 1).to(device)

    base_lr = finetuning_learning_rate
    head_lr = base_lr * 10

    optimizer = torch.optim.AdamW(
        [
        {"params": training_setup.model.parameters(), "lr": base_lr},
        {"params": linear_head.parameters(), "lr": head_lr}
        ]
    )

    finetune_run_name = f"{model_dir}_FT_{eval_name}_S{train_subject_id}-{train_trial_id}_T{test_subject_id}-{test_trial_id}"
    config['cluster']['dir_name'] = finetune_run_name

    training_setup.model_components['linear_head'] = linear_head
    # Step 1. Set up the optimizer
    # Step 2. Loop over the number of fine-tuning epochs
    # Step 3. For each epoch, loop over the number of batches
    # Step 4. For each batch, compute the features (NOTE: avoid the torch.no_grad() context manager)
    # Step 5. Pass the features through the linear layer to get the predictions
    # Step 6. Compute the loss (logistic regression loss)
    # Step 7. Backpropagate the loss - loss.backward()
    # Step 8. Update the weights - optimizer.step()
    # Step 9. Log the loss - wandb.log()
    # Step 10. Save the model - torch.save()

    # Pass through model to get all train and test outputs
    for epoch_idx in range(finetuning_epochs):
        # ---- TRAIN ----
        training_setup.model.train()
        linear_head.train()
        train_losses = []
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (batch_input, batch_label) in enumerate(train_loader):
            batch = {
                "data": batch_input.to(device, dtype=config['training']['data_dtype']),
                "electrode_labels": [electrode_subset_train],
                "metadata": {
                    "subject_identifier": train_subject.subject_identifier,
                    "trial_id": train_trial_id,
                    "eval_name": eval_name,
                    "sampling_rate": neuroprobe_config.SAMPLING_RATE
                }
            }
            for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
                batch = preprocess_function(batch)

            # forward through encoder (we are fine-tuning, so keep grads on)
            model_output = training_setup.generate_frozen_features(batch)
            cls = model_output[:, 0, :, :]
            pooled = cls.mean(dim=1)
            logits = linear_head(pooled).squeeze(-1)

            loss = loss_fn(logits, batch_label.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log("optimizer step")

            train_losses.append(loss.item())
            all_train_preds.append(logits.detach().cpu())
            all_train_labels.append(batch_label.detach().cpu())

            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch_idx,
                "batch_idx": batch_idx
            })

            del batch_input, batch_label, model_output, batch, logits, pooled, cls
            gc.collect()
            torch.cuda.empty_cache()

        # aggregate train metrics for the epoch
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_probs = torch.sigmoid(all_train_preds).numpy()
        train_bin = (train_probs > 0.5).astype(np.float32)
        train_true = all_train_labels.numpy()
        try:
            train_auroc = roc_auc_score(train_true, train_probs)
        except ValueError:
            train_auroc = float('nan')
        train_acc = accuracy_score(train_true, train_bin)
        train_loss_mean = float(np.mean(train_losses))

        # ---- EVAL / TEST ----
        training_setup.model.eval()
        linear_head.eval()
        test_losses = []
        all_test_preds = []
        all_test_labels = []

        with torch.no_grad():
            for batch_idx, (batch_input, batch_label) in enumerate(test_loader):
                batch = {
                    "data": batch_input.to(device, dtype=config['training']['data_dtype']),
                    "electrode_labels": [electrode_subset_test],
                    "metadata": {
                        "subject_identifier": test_subject.subject_identifier,
                        "trial_id": test_trial_id,
                        "eval_name": eval_name,
                        "sampling_rate": neuroprobe_config.SAMPLING_RATE
                    }
                }
                for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
                    batch = preprocess_function(batch)

                model_output = training_setup.generate_frozen_features(batch)
                cls = model_output[:, 0, :, :]
                pooled = cls.mean(dim=1)
                logits = linear_head(pooled).squeeze(-1)

                loss = loss_fn(logits, batch_label.float().to(device))
                test_losses.append(loss.item())
                all_test_preds.append(logits.cpu())
                all_test_labels.append(batch_label.cpu())

                del batch_input, batch_label, model_output, batch, logits, pooled, cls
                gc.collect()
                torch.cuda.empty_cache()

        # aggregate test metrics
        all_test_preds = torch.cat(all_test_preds, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
        test_probs = torch.sigmoid(all_test_preds).numpy()
        test_bin = (test_probs > 0.5).astype(np.float32)
        test_true = all_test_labels.numpy()
        try:
            test_auroc = roc_auc_score(test_true, test_probs)
        except ValueError:
            test_auroc = float('nan')
        test_acc = accuracy_score(test_true, test_bin)
        test_loss_mean = float(np.mean(test_losses)) if len(test_losses) > 0 else float('nan')

        # ---- Log per-epoch metrics ----
        wandb.log({
            "epoch": epoch_idx,
            "train/loss": train_loss_mean,
            "train/accuracy": train_acc,
            "train/auroc": train_auroc,
            "test/loss": test_loss_mean,
            "test/accuracy": test_acc,
            "test/auroc": test_auroc,
            "lr/encoder": optimizer.param_groups[0]["lr"],
            "lr/head": optimizer.param_groups[1]["lr"],
        })

        # ---- Save finetuned weights for this epoch ----
        eval_results = {
        "epoch": epoch_idx,
        "train/loss": train_loss_mean,
        "train/accuracy": train_acc,
        "train/auroc": train_auroc,
        "test/loss": test_loss_mean,
        "test/accuracy": test_acc,
        "test/auroc": test_auroc,
        }


        training_setup.save_model(
            epoch=epoch_idx,
            eval_results=eval_results,
            save_in_dir=RUNS_DIR,
            training_statistics_store=None,
        )

        log(
        f"[Epoch {epoch_idx+1}/{finetuning_epochs}] "
        f"Train - Loss: {train_loss_mean:.4f}, Acc: {train_acc:.4f}, AUROC: {train_auroc:.4f} | "
        f"Test - Loss: {test_loss_mean:.4f}, Acc: {test_acc:.4f}, AUROC: {test_auroc:.4f}",
        priority=0
        )
