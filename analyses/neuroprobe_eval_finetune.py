# Credit: Bhadra Rupesh, MIT; Andrii Zahorodnii, MIT

import gc
import json
import os
import time

import numpy as np
import torch
import wandb
from torch.amp import autocast

from bfm.evaluation.neuroprobe import config as neuroprobe_config
from bfm.evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
from bfm.subject.datasets.dataset import load_subjects
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR, StepLR, ExponentialLR
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from bfm.training.setup_registry import setups
from bfm.training.training_config import (
    convert_dtypes,
    get_default_config,
    parse_config_from_args,
    parse_subject_trials_from_config,
    unconvert_dtypes,
    update_dir_name,
    update_random_seed,
)
from bfm.core.logger import log
# torch.set_float32_matmul_precision('high')

RUNS_DIR='runs/data'

### PARSE MODEL DIR ###

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
# parser.add_argument('--train_subject_id', type=str, required=True, help='Subject identifier')
# parser.add_argument('--train_trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--subject_id', type=int, required=True, help='Subject identifier')
parser.add_argument('--trial_id', type=int, required=True, help='Trial identifier')
parser.add_argument('--eval_name', type=str, required=True, help='Tasks to evaluate on, comma-separated')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing frozen features')
parser.add_argument('--finetuning_batch_size', type=int, default=100, help='Batch size for feature computation')
parser.add_argument('--finetuning_learning_rate', type=float, default=0.003/2, help='Learning rate for finetuning')
parser.add_argument('--finetuning_lr_scheduler', type=str, default='linear', help='Learning rate scheduler')
parser.add_argument('--finetuning_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--finetuning_feature_aggregation_method', type=str, default='meanT_meanE', help='Feature aggregation method for finetuning')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

model_dir = args.model_dir
model_epoch = args.model_epoch if args.model_epoch >= 0 else "final"
# train_subject_id = args.train_subject_id
# train_trial_id = args.train_trial_id
test_subject_id = args.subject_id
test_trial_id = args.trial_id
eval_tasks = args.eval_name.split(",")
overwrite = args.overwrite
batch_size = args.finetuning_batch_size
finetuning_learning_rate = args.finetuning_learning_rate
finetuning_lr_scheduler = args.finetuning_lr_scheduler
finetuning_epochs = args.finetuning_epochs
finetuning_feature_aggregation_method = args.finetuning_feature_aggregation_method
random_seed = args.random_seed

# defaulting to the SS_DM split
train_subject_trial = [(s_i, t_i) for s_i, t_i in neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS if s_i == test_subject_id and t_i != test_trial_id][0]
train_subject_id, train_trial_id = train_subject_trial

bins_start_before_word_onset_seconds = 0
bins_end_after_word_onset_seconds = 1.0

### SET SEED ###

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

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


### LOAD MODEL ###

# Import the training setup class dynamically based on config
setup_name = config["training"]["setup_name"] # Name in registry
training_setup = setups.resolve(setup_name, all_subjects=all_subjects, config=config, verbose=True)


log(f"Loading model...", priority=0)
training_setup.initialize_model()

log(f"Loading model weights...", priority=0)
training_setup.load_model(model_epoch)
loss_fn = nn.BCEWithLogitsLoss()

def apply_feature_aggregation(features, aggregation_method):
    if 'meanT' in aggregation_method:
        features = features.mean(dim=2, keepdim=True) # shape: (batch_size, n_electrodes + 1, 1, d_model)
    if 'meanE' in aggregation_method:
        features = features.mean(dim=1, keepdim=True) # shape: (batch_size, 1, n_timebins, d_model)
    if 'cls' in aggregation_method:
        features = features[:, 0:1, :, :] # shape: (batch_size, 1, n_timebins, d_model) -- take just the cls token
    return features
def process_batch_and_predict(batch_input, batch_label, subject, trial_id, 
                             linear_head, just_output_features=False):
    """Helper function to process batch data and get predictions"""
    batch = {
        "data": batch_input.to(device, dtype=config['training']['data_dtype']),
        "electrode_labels": [neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]],
        "metadata": {
            "subject_identifier": subject.subject_identifier,
            "trial_id": trial_id,
            "sampling_rate": neuroprobe_config.SAMPLING_RATE
        }
    }
    for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
        batch = preprocess_function(batch)

    # forward through encoder
    model_output = training_setup.generate_frozen_features(batch)
    
    pooled = apply_feature_aggregation(model_output, finetuning_feature_aggregation_method).reshape(batch_input.shape[0], -1)
    if just_output_features:
        return pooled

    logits = linear_head(pooled).squeeze(-1)
    return logits

def evaluate_epoch(data_loader, subject, trial_id, 
                  linear_head, is_training=False, training_statistics_store=None, scheduler=None):
    """Helper function to evaluate model on a dataset"""
    losses = []
    all_preds = []
    all_labels = []
    
    context_manager = torch.no_grad() if not is_training else torch.enable_grad()
    
    with context_manager:
        for batch_idx, (batch_input, batch_label) in enumerate(data_loader):
            logits = process_batch_and_predict(
                batch_input, batch_label, subject, trial_id, 
                linear_head
            )
            
            loss = loss_fn(logits, batch_label.float().to(device))
            
            if is_training:
                # Training-specific operations
                training_setup.optimizer.zero_grad()
                loss.backward()
                training_setup.optimizer.step()
                
                # Step learning rate scheduler after each batch
                if scheduler is not None:
                    scheduler.step()
                
                # Collect batch-level statistics for training_statistics_store
                if training_statistics_store is not None:
                    
                    training_statistics_store.append({
                        'epoch': training_setup.current_epoch + 1,
                        'batch': batch_idx + 1,
                        'subject_identifier': subject.subject_identifier,
                        'trial_id': trial_id,
                        'batch_loss': loss.item(),
                        'timestamp': time.time(),
                    })
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": training_setup.current_epoch,
                    "batch_idx": batch_idx,
                    "learning_rate": training_setup.optimizer.param_groups[0]['lr']
                })
                log(f"Epoch {training_setup.current_epoch+1}/{finetuning_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, Learning Rate: {training_setup.optimizer.param_groups[0]['lr']:.6f}", priority=0)
            
            losses.append(loss.item())
            all_preds.append(logits.detach().cpu())
            all_labels.append(batch_label.detach().cpu())
    
    # Aggregate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_preds).numpy()
    binary_preds = (probs > 0.5).astype(np.float32)
    true_labels = all_labels.numpy()
    
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(true_labels, binary_preds),
        "auroc": roc_auc_score(true_labels, probs)
    }

for eval_name in eval_tasks:
    start_time = time.time()
    
    finetune_run_name = f"{model_dir}/model_epoch{model_epoch}_neuroprobe_finetuning_lr{finetuning_learning_rate}_{finetuning_feature_aggregation_method}_r{random_seed}/{eval_name}_t{train_subject_id}_{train_trial_id}_e{test_subject_id}_{test_trial_id}"
    config['cluster']['dir_name'] = finetune_run_name

    # Create results directory if it doesn't exist
    results_dir = os.path.join(RUNS_DIR, finetune_run_name, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file_save_path = os.path.join(results_dir, f"population_btbank{test_subject_id}_{test_trial_id}_{eval_name}.json")
    if os.path.exists(results_file_save_path) and not overwrite:
        log(f"Results file already exists at {results_file_save_path}, skipping finetuning", priority=0)
        continue

    wandb.init(
        project="neuroprobe-finetuning",
        config={
            "model_dir": model_dir,
            "model_epoch": model_epoch,
            "train_subject": train_subject_id,
            "train_trial": train_trial_id,
            "test_subject": test_subject_id,
            "test_trial": test_trial_id,
            "eval_name": eval_name,
            "learning_rate": finetuning_learning_rate,
            "lr_scheduler": finetuning_lr_scheduler,
            "epochs": finetuning_epochs,
            "batch_size": batch_size,
        },
        dir="runs/wandb/",
        name=f"{model_dir}_epoch{model_epoch}_ft_{finetuning_feature_aggregation_method}_{eval_name}_t{train_subject_id}_{train_trial_id}_e{test_subject_id}_{test_trial_id}_r{random_seed}"
    )

    start_offset = int(bins_start_before_word_onset_seconds * neuroprobe_config.SAMPLING_RATE)
    end_offset = int(bins_end_after_word_onset_seconds * neuroprobe_config.SAMPLING_RATE)

    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        train_subject, train_trial_id, dtype=torch.float32, eval_name=eval_name,
        output_indices=False, lite=True,
        start_neural_data_before_word_onset=start_offset,
        end_neural_data_after_word_onset=end_offset,
    )
    
    # Create full test dataset first
    full_test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        test_subject, test_trial_id, dtype=torch.float32, eval_name=eval_name,
        output_indices=False, lite=True,
        start_neural_data_before_word_onset=start_offset,
        end_neural_data_after_word_onset=end_offset,
    )
    
    # Split test dataset into val (10%) and test (90%)
    val_size = int(0.1 * len(full_test_dataset))
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    linear_head_input_dimension = process_batch_and_predict(*next(iter(train_loader)), train_subject, train_trial_id, None, just_output_features=True).shape[1]
    linear_head = nn.Linear(linear_head_input_dimension, 1).to(device)
    # Zero all weights in the linear head
    linear_head.weight.data.zero_()
    linear_head.bias.data.zero_()

    base_lr = finetuning_learning_rate * 0.1
    head_lr = finetuning_learning_rate

    optimizer = torch.optim.AdamW(
        [
        {"params": training_setup.model.parameters(), "lr": base_lr},
        {"params": linear_head.parameters(), "lr": head_lr}
        ]
    )
    
    # Initialize learning rate scheduler (scheduled per batch)
    if finetuning_lr_scheduler == 'linear':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=len(train_loader) * finetuning_epochs)
    else:
        scheduler = None

    training_setup.model_components['linear_head'] = linear_head
    
    # Store optimizer in training_setup for the helper function
    training_setup.optimizer = optimizer
    
    # Track best model
    best_val_auroc = -1
    best_epoch = -1
    best_train_results = {}
    best_test_results = {}
    
    # Initialize training statistics store
    training_statistics_store = []
    
    # Training loop
    for epoch_idx in range(finetuning_epochs):
        training_setup.current_epoch = epoch_idx  # Store current epoch for logging
        
        # ---- TRAIN ----
        training_setup.model.train()
        linear_head.train()
        
        train_results = evaluate_epoch(
            train_loader, train_subject, train_trial_id, 
            linear_head, is_training=True, training_statistics_store=training_statistics_store, scheduler=scheduler
        )

        # ---- VALIDATION ----
        training_setup.model.eval()
        linear_head.eval()
        
        val_results = evaluate_epoch(
            val_loader, test_subject, test_trial_id, 
            linear_head, is_training=False
        )

        # ---- TEST ----
        test_results = evaluate_epoch(
            test_loader, test_subject, test_trial_id, 
            linear_head, is_training=False
        )

        # Check if this is the best model so far (based on validation AUROC)
        if val_results["auroc"] > best_val_auroc:
            best_val_auroc = val_results["auroc"]
            best_epoch = epoch_idx
            best_train_results = train_results.copy()
            best_test_results = test_results.copy()
            # Save JSON results for best model
            end_time = time.time()
            regression_run_time = end_time - start_time

        # ---- Log per-epoch metrics ----
        wandb.log({
            "epoch": epoch_idx,
            "train/loss": train_results["loss"],
            "train/accuracy": train_results["accuracy"],
            "train/auroc": train_results["auroc"],
            "val/loss": val_results["loss"],
            "val/accuracy": val_results["accuracy"],
            "val/auroc": val_results["auroc"],
            "test/loss": test_results["loss"],
            "test/accuracy": test_results["accuracy"],
            "test/auroc": test_results["auroc"],
        })

        # ---- Save finetuned weights for this epoch ----
        eval_results = {
            "epoch": epoch_idx,
            "train/loss": train_results["loss"],
            "train/accuracy": train_results["accuracy"],
            "train/auroc": train_results["auroc"],
            "val/loss": val_results["loss"],
            "val/accuracy": val_results["accuracy"],
            "val/auroc": val_results["auroc"],
            "test/loss": test_results["loss"],
            "test/accuracy": test_results["accuracy"],
            "test/auroc": test_results["auroc"],
        }
        
        # Add epoch-level results to training statistics store
        if len(training_statistics_store) > 0:
            training_statistics_store[-1].update(eval_results)

        training_setup.save_model(
            epoch=epoch_idx,
            eval_results=eval_results,
            save_in_dir=RUNS_DIR,
            training_statistics_store=training_statistics_store,
        )

        log(f"Epoch {epoch_idx+1}/{finetuning_epochs}, " + \
            f"Train - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.4f}, AUROC: {train_results['auroc']:.4f} | " + \
            f"Val - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, AUROC: {val_results['auroc']:.4f} | " + \
            f"Test - Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}, AUROC: {test_results['auroc']:.4f}", priority=0)

        gc.collect()
        torch.cuda.empty_cache()

        
    results = {
        "model_name": f"{model_dir}_finetuned",
        "author": "Andrii Zahorodnii",
        "description": f"Finetuned {model_dir} on {eval_name} task using train subject {train_subject_id} trial {train_trial_id}, tested on subject {test_subject_id} trial {test_trial_id}.",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{test_subject.subject_identifier}_{test_trial_id}": {
                "population": {
                    "time_bins": [],
                    "one_second_after_onset": {
                        "time_bin_start": 0.0,
                        "time_bin_end": 1.0,
                        "folds": [
                            {
                                "train_accuracy": best_train_results["accuracy"],
                                "train_roc_auc": best_train_results["auroc"],
                                "test_accuracy": best_test_results["accuracy"],
                                "test_roc_auc": best_test_results["auroc"]
                            }
                        ]
                    }
                }
            }
        },

        "config": {
            "training_setup_config": convert_dtypes(config),
            "model_dir": model_dir,
            "model_epoch": model_epoch,
            "train_subject_id": train_subject_id,
            "train_trial_id": train_trial_id,
            "test_subject_id": test_subject_id,
            "test_trial_id": test_trial_id,
            "task": eval_name,
            "learning_rate": finetuning_learning_rate,
            "lr_scheduler": finetuning_lr_scheduler,
            "epochs": finetuning_epochs,
            "batch_size": batch_size,
            "val_split": 0.1,
        },

        "timing": {
            "total_training_time": regression_run_time,
        }
    }

    with open(results_file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    log(f"Results saved to {results_file_save_path}", priority=0)
    
    wandb.finish()

    # Clean up at end of each eval_name loop
    del train_dataset, val_dataset, test_dataset, full_test_dataset
    gc.collect()
