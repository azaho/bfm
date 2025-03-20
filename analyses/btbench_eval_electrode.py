import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path

from subject_braintreebank import BrainTreebankSubject
import btbench.btbench_train_test_splits as btbench_train_test_splits
import btbench.btbench_config as btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--eval_names', type=str, default='onset', help='Evaluation names list, separated by commas (e.g. onset, gpt2_surprisal)')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='analyses/eval_results', help='Directory to save results')
parser.add_argument('--model_dir', type=str, help='Directory containing model files')
parser.add_argument('--model_epoch', type=int, default=100, help='Model epoch to evaluate')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for evaluation')
parser.add_argument('--timebins_aggregation', type=str, default='concat', help='Timebins aggregation method (mean, concat)')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
args = parser.parse_args()

eval_names = args.eval_names.split(',')
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
model_dir = args.model_dir
model_epoch = args.model_epoch
batch_size = args.batch_size
timebins_aggregation = args.timebins_aggregation
overwrite = bool(args.overwrite)

# Loading the model
import torch
from model_model import TransformerModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeDataEmbeddingFFT
from train_utils import *
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(f"models_data/{model_dir}/model_epoch_{model_epoch}.pth", map_location=device)

training_config, model_config, cluster_config = checkpoint['training_config'], checkpoint['model_config'], checkpoint['cluster_config']
training_config = unconvert_dtypes(training_config) # convert string dtypes back to torch dtypes
model_config = unconvert_dtypes(model_config)
cluster_config = unconvert_dtypes(cluster_config)

bins_start_before_word_onset_seconds = 0.5
bins_end_after_word_onset_seconds = 2.5
bin_size_seconds = model_config['sample_timebin_size']

# Initialize model
model = TransformerModel(
    model_config['transformer']['d_model'],
    n_layers_electrode=model_config['transformer']['n_layers_electrode'],
    n_layers_time=model_config['transformer']['n_layers_time']
).to(device, dtype=model_config['dtype'])

# Initialize electrode embeddings based on config type
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
    electrode_embeddings,
    model_config['sample_timebin_size']
).to(device, dtype=model_config['dtype'])

# Load saved model checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
electrode_data_embeddings.load_state_dict(checkpoint['electrode_data_embeddings_state_dict'])
if verbose: print("Model loaded", model.eval(), "", sep="\n")

assert bin_size_seconds == model_config['sample_timebin_size'], f"Bin size {bin_size_seconds} does not match model config sample timebin size {model_config['sample_timebin_size']}"
assert int((bins_start_before_word_onset_seconds+bins_end_after_word_onset_seconds)/bin_size_seconds) <= model_config['max_n_timebins'], f"Time window {bins_start_before_word_onset_seconds+bins_end_after_word_onset_seconds} does is too big given the model config max_n_timebins {model_config['max_n_timebins']}"

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels

for eval_name in eval_names:
    save_file_path = os.path.join(save_dir, model_dir, f"frozen_electrode_btbank{subject_id}_{trial_id}_{eval_name}.json")
    if not overwrite and os.path.exists(save_file_path):
        if verbose:
            print(f"Results already exist at {save_file_path}, skipping...")
        continue

    results_electrode = {}
    for electrode_idx, electrode_label in enumerate(all_electrode_labels):
        subject.clear_neural_data_cache()
        subject.set_electrode_subset([electrode_label])
        subject.load_neural_data(trial_id)
        if verbose:
            print(f"Electrode {electrode_label} subject loaded")

        results_electrode[electrode_label] = {
            "time_bins": [],
        }

        # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, add_other_trials=False, k_folds=5, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE))


        # Prepare bin results dictionaries
        bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
        bin_ends = bin_starts + bin_size_seconds
        for bin_start, bin_end in zip(bin_starts, bin_ends):
            bin_results = {
                "time_bin_start": float(bin_start),
                "time_bin_end": float(bin_end),
                "folds": []
            }
            results_electrode[electrode_label]["time_bins"].append(bin_results)
        results_electrode[electrode_label]["whole_window"] = {
            "time_bin_start": float(-bins_start_before_word_onset_seconds),
            "time_bin_end": float(bins_end_after_word_onset_seconds),
            "folds": []
        }


        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            X_train_all_bins = []
            X_test_all_bins = []
            y_train = []
            y_test = []

            # Pass through model to get all train and test outputs
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch_input, batch_label in train_dataloader:
                    batch_input = batch_input.to(device, dtype=model_config['dtype'])
                    electrode_data = electrode_data_embeddings.forward(subject.subject_identifier, [electrode_idx], batch_input)
                    model_output = model(electrode_data, only_electrode_output=True)[0]
                    X_train_all_bins.append(model_output)
                    y_train.append(batch_label)
            with torch.no_grad():
                for batch_input, batch_label in test_dataloader:
                    batch_input = batch_input.to(device, dtype=model_config['dtype'])
                    electrode_data = electrode_data_embeddings.forward(subject.subject_identifier, [electrode_idx], batch_input)
                    model_output = model(electrode_data, only_electrode_output=True)[0]
                    X_test_all_bins.append(model_output)
                    y_test.append(batch_label)
            X_train_all_bins = torch.cat(X_train_all_bins, dim=0) # shape: (n_dataset, n_timebins, d_model)
            X_test_all_bins = torch.cat(X_test_all_bins, dim=0)
            y_train = torch.cat(y_train, dim=0).float().cpu().numpy()
            y_test = torch.cat(y_test, dim=0).float().cpu().numpy()

            for bin_idx in range(-1, len(bin_starts)):
                bin_start = bin_starts[bin_idx] if bin_idx != -1 else -bins_start_before_word_onset_seconds
                bin_end = bin_ends[bin_idx] if bin_idx != -1 else bins_end_after_word_onset_seconds

                if bin_idx == -1:
                    if timebins_aggregation == "concat":
                        X_train = X_train_all_bins.reshape(X_train_all_bins.shape[0], -1).float().cpu().numpy()
                        X_test = X_test_all_bins.reshape(X_test_all_bins.shape[0], -1).float().cpu().numpy()
                    else:
                        X_train = X_train_all_bins.mean(dim=1).float().cpu().numpy()
                        X_test = X_test_all_bins.mean(dim=1).float().cpu().numpy()
                else:
                    X_train = X_train_all_bins[:, bin_idx, :].float().cpu().numpy()
                    X_test = X_test_all_bins[:, bin_idx, :].float().cpu().numpy()

                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train logistic regression
                clf = LogisticRegression(random_state=42, max_iter=10000, tol=1e-3)
                clf.fit(X_train, y_train)

                # Evaluate model
                train_accuracy = clf.score(X_train, y_train)
                test_accuracy = clf.score(X_test, y_test)

                # Get predictions - for multiclass classification
                train_probs = clf.predict_proba(X_train)
                test_probs = clf.predict_proba(X_test)

                # Filter test samples to only include classes that were in training
                valid_class_mask = np.isin(y_test, clf.classes_)
                y_test_filtered = y_test[valid_class_mask]
                test_probs_filtered = test_probs[valid_class_mask]

                # Convert y_test to one-hot encoding
                y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
                for i, label in enumerate(y_test_filtered):
                    class_idx = np.where(clf.classes_ == label)[0][0]
                    y_test_onehot[i, class_idx] = 1

                y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
                for i, label in enumerate(y_train):
                    class_idx = np.where(clf.classes_ == label)[0][0]
                    y_train_onehot[i, class_idx] = 1

                # For multiclass ROC AUC, we need to calculate the score for each class
                n_classes = len(clf.classes_)
                if n_classes > 2:
                    train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                    test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
                else:
                    train_roc = roc_auc_score(y_train_onehot, train_probs)
                    test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

                fold_result = {
                    "train_accuracy": float(train_accuracy),
                    "train_roc_auc": float(train_roc),
                    "test_accuracy": float(test_accuracy),
                    "test_roc_auc": float(test_roc)
                }

                if bin_idx == -1:
                    results_electrode[electrode_label]["whole_window"]["folds"].append(fold_result)
                else:
                    results_electrode[electrode_label]["time_bins"][bin_idx]["folds"].append(fold_result)
                if verbose: print(f"Electrode {electrode_label} ({electrode_idx+1}/{len(all_electrode_labels)}), Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}")

    results = {
        "model_name": "Andrii's model",
        "author": "Andrii Zahorodnii",
        "description": "Linear regression on top of the output of the transformer model.",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "electrode": results_electrode
            }
        }
    }
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True) # Create save directory if it doesn't exist
    with open(save_file_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        print(f"Results saved to {save_file_path}\n\n")