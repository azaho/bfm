import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path

from subject_braintreebank import BrainTreebankSubject
import btbench.btbench_train_test_splits as btbench_train_test_splits
import btbench.btbench_config as btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--eval_names', type=str, default='onset', help='Evaluation names list, separated by commas (e.g. onset, gpt2_surprisal)')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='analyses/eval_results', help='Directory to save results')
parser.add_argument('--features_dir', type=str, default='analyses/eval_features', help='Directory containing pre-computed features')
parser.add_argument('--model_dir', type=str, help='Directory containing model files')
parser.add_argument('--model_epoch', type=int, default=100, help='Model epoch to evaluate')
parser.add_argument('--timebins_aggregation', type=str, default='mean', help='Timebins aggregation method (mean, concat)')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation splits')
args = parser.parse_args()

eval_names = args.eval_names.split(',')
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
features_dir = args.features_dir
model_dir = args.model_dir
model_epoch = args.model_epoch
timebins_aggregation = args.timebins_aggregation
overwrite = bool(args.overwrite)
n_splits = args.n_splits

bins_start_before_word_onset_seconds = 0.5
bins_end_after_word_onset_seconds = 2.5

results_population = {
    "time_bins": [],
}

for eval_name in eval_names:
    save_file_path = os.path.join(save_dir, model_dir+f"_epoch{model_epoch}", f"frozen_population_btbank{subject_id}_{trial_id}_{eval_name}.json")
    features_file_path = os.path.join(features_dir, model_dir+f"_epoch{model_epoch}", f"frozen_population_btbank{subject_id}_{trial_id}_{eval_name}.npy")
    
    if not overwrite and os.path.exists(save_file_path):
        if verbose:
            print(f"Results already exist at {save_file_path}, skipping...")
        continue

    # Load pre-computed features
    features_dict = np.load(features_file_path, allow_pickle=True).item()
    X_all_bins = features_dict['X']  # shape: (n_samples, n_timebins, 1, d_model)
    y = features_dict['y']
    metadata = features_dict['metadata']

    # Prepare bin results dictionaries
    bin_size_seconds = metadata['model_config']['sample_timebin_size']
    bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
    bin_ends = bin_starts + bin_size_seconds
    
    for bin_start, bin_end in zip(bin_starts, bin_ends):
        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }
        results_population["time_bins"].append(bin_results)
    
    results_population["whole_window"] = {
        "time_bin_start": float(-bins_start_before_word_onset_seconds),
        "time_bin_end": float(bins_end_after_word_onset_seconds),
        "folds": []
    }
    results_population["one_second_after_onset"] = {
        "time_bin_start": 0.0,
        "time_bin_end": 1.0,
        "folds": []
    }

    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=42)

    # Loop over all folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all_bins)):
        X_train_all_bins = X_all_bins[train_idx]
        X_test_all_bins = X_all_bins[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for bin_idx in range(-2, len(bin_starts)):
            if bin_idx == -2:  # 1 second window centered on onset
                if timebins_aggregation == "concat":
                    X_train = X_train_all_bins[:, 4:12, 0, :].reshape(len(train_idx), -1)
                    X_test = X_test_all_bins[:, 4:12, 0, :].reshape(len(test_idx), -1)
                elif timebins_aggregation == "mean":
                    X_train = X_train_all_bins[:, 4:12, 0, :].mean(axis=1)
                    X_test = X_test_all_bins[:, 4:12, 0, :].mean(axis=1)
            elif bin_idx == -1:  # whole window
                if timebins_aggregation == "concat":
                    X_train = X_train_all_bins[:, :, 0, :].reshape(len(train_idx), -1)
                    X_test = X_test_all_bins[:, :, 0, :].reshape(len(test_idx), -1)
                elif timebins_aggregation == "mean":
                    X_train = X_train_all_bins[:, :, 0, :].mean(axis=1)
                    X_test = X_test_all_bins[:, :, 0, :].mean(axis=1)
            else:  # individual time bins
                X_train = X_train_all_bins[:, bin_idx, 0, :]
                X_test = X_test_all_bins[:, bin_idx, 0, :]

            # Rest of the evaluation code remains the same
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train logistic regression
            clf = LogisticRegression(random_state=42, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)

            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            # Convert to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1
            
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            # Calculate ROC AUC
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

            if bin_idx == -2:
                results_population["one_second_after_onset"]["folds"].append(fold_result)
            elif bin_idx == -1:
                results_population["whole_window"]["folds"].append(fold_result)
            else:
                results_population["time_bins"][bin_idx]["folds"].append(fold_result)
            
            if verbose: 
                bin_start = bin_starts[bin_idx] if bin_idx >= 0 else -bins_start_before_word_onset_seconds
                bin_end = bin_ends[bin_idx] if bin_idx >= 0 else bins_end_after_word_onset_seconds
                print(f"Subject {subject_id} trial {trial_id}, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}")

    results = {
        "model_name": f"Andrii's model ({model_dir}_epoch{model_epoch})",
        "author": "Andrii Zahorodnii",
        "description": "Linear regression on top of the output of the transformer model.",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject_id}_{trial_id}": {
                "population": results_population
            }
        }
    }
    
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        print(f"Results saved to {save_file_path}\n\n")