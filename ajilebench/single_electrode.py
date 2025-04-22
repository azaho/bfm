from subject_ajile12 import AjileSubject
import ajilebench.ajilebench_train_test_splits as ajilebench_train_test_splits
import ajilebench.ajilebench_config as ajilebench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='reach_onset', help='Evaluation name(s) (e.g. reach_onset). If multiple, separate with commas.')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
only_1second = bool(args.only_1second)
seed = args.seed

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5 if not only_1second else 1
bin_size_seconds = 0.125

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = AjileSubject(subject_id, cache=True, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels

for eval_name in eval_names:
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
        train_datasets, test_datasets = ajilebench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, add_other_trials=False, k_folds=5, dtype=torch.float32, 
                                                                                            output_indices=False, 
                                                                                            start_neural_data_before_reach_onset=int(bins_start_before_word_onset_seconds*ajilebench_config.SAMPLING_RATE), 
                                                                                            end_neural_data_after_reach_onset=int(bins_end_after_word_onset_seconds*ajilebench_config.SAMPLING_RATE))

        if only_1second:
            bin_starts, bin_ends = [0], [1]
        else:
            # Loop over all time bins
            bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
            bin_ends = bin_starts + bin_size_seconds
            # Add a time bin for the whole window
            bin_starts = np.append(bin_starts, -bins_start_before_word_onset_seconds)
            bin_ends = np.append(bin_ends, bins_end_after_word_onset_seconds)
            # Add a time bin for 1 second after the word onset
            bin_starts = np.append(bin_starts, 0)
            bin_ends = np.append(bin_ends, 1)

        for bin_start, bin_end in zip(bin_starts, bin_ends):
            data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*ajilebench_config.SAMPLING_RATE)
            data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*ajilebench_config.SAMPLING_RATE)

            bin_results = {
                "time_bin_start": float(bin_start),
                "time_bin_end": float(bin_end),
                "folds": []
            }

            # Loop over all folds
            for fold_idx in range(len(train_datasets)):
                train_dataset = train_datasets[fold_idx]
                test_dataset = test_datasets[fold_idx]

                # Convert PyTorch dataset to numpy arrays for scikit-learn
                X_train = np.array([item[0][:, data_idx_from:data_idx_to].flatten() for item in train_dataset])
                y_train = np.array([item[1] for item in train_dataset])
                X_test = np.array([item[0][:, data_idx_from:data_idx_to].flatten() for item in test_dataset])
                y_test = np.array([item[1] for item in test_dataset])

                # Check for NaN values in the data arrays
                for array_name, array in [("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)]:
                    if np.isnan(array).any():
                        nan_indices = np.where(np.isnan(array))[0]
                        print(f"WARNING: Found NaN values in {array_name} for electrode {electrode_label}, trial {trial_id}, time bin [{bin_start}, {bin_end}]")
                        print(f"NaN indices: {nan_indices}")
                        
                        if array_name in ["X_train", "X_test"]:
                            for idx in nan_indices:
                                if array_name == "X_train":
                                    print(f"Train dataset item {idx}: {train_dataset[idx]}")
                                else:
                                    print(f"Test dataset item {idx}: {test_dataset[idx]}")

                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train logistic regression
                clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
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
                bin_results["folds"].append(fold_result)
                if verbose: print(f"Electrode {electrode_label} ({electrode_idx+1}/{len(all_electrode_labels)}), Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}")

            if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds:
                results_electrode[electrode_label]["whole_window"] = bin_results # whole window results
            elif bin_start == 0 and bin_end == 1:
                results_electrode[electrode_label]["one_second_after_onset"] = bin_results # one second after onset results
            else:
                results_electrode[electrode_label]["time_bins"].append(bin_results) # time bin results

    results = {
        "model_name": "Logistic Regression",
        "author": None,
        "description": "Simple linear regression.",
        "organization": "MIT",
        "organization_url": "https://mit.edu",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "electrode": results_electrode
            }
        },

        "random_seed": seed
    }
    os.makedirs(save_dir, exist_ok=True) # Create save directory if it doesn't exist
    with open(f"{save_dir}/linear_regression_electrode_{subject.subject_identifier}_{trial_id}_{eval_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        print(f"Results saved to {save_dir}/linear_regression_electrode_{subject.subject_identifier}_{trial_id}_{eval_name}.json")
