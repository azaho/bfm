from subject_ajile12 import AjileSubject
import ajilebench.ajilebench_datasets as ajilebench_datasets
import ajilebench.ajilebench_config as ajilebench_config

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

        dataset = ajilebench_datasets.AjileSubjectTrialBenchmarkDataset(subject, trial_id, torch.float32, eval_name, output_indices=False, 
                 start_neural_data_before_reach_onset=int(bins_start_before_word_onset_seconds*ajilebench_config.SAMPLING_RATE), 
                                                                                            end_neural_data_after_reach_onset=int(bins_end_after_word_onset_seconds*ajilebench_config.SAMPLING_RATE),
                                                                                            replace_nan_with=0)

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
                "activity": {}
            }

            # Use only the first fold
            fold_idx = 0

            # Extract data and labels
            X = np.array([item[0][:, data_idx_from:data_idx_to].flatten() for item in dataset])
            y = np.array([item[1] for item in dataset])

            # Check for NaN values in the data arrays
            if np.isnan(X).any():
                nan_indices = np.where(np.isnan(X))[0]
                print(f"WARNING: Found NaN values in data for electrode {electrode_label}, trial {trial_id}, time bin [{bin_start}, {bin_end}]")
                print(f"NaN indices: {nan_indices}")
                
                for idx in nan_indices:
                    print(f"Dataset item {idx}: {dataset[idx]}")

            # Get unique labels
            unique_labels = np.unique(y)
            
            # Calculate average activity for each label
            for label in unique_labels:
                label_mask = (y == label)
                label_data = X[label_mask]
                avg_activity = np.mean(label_data, axis=0)
                
                # Store the average activity for this label
                bin_results["activity"][f"label_{label}"] = avg_activity.tolist()
                
                if verbose:
                    print(f"Electrode {electrode_label} ({electrode_idx+1}/{len(all_electrode_labels)}), Bin {bin_start}-{bin_end}, Label {label}: Average activity calculated from {len(label_data)} samples")

            if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds:
                results_electrode[electrode_label]["whole_window"] = bin_results # whole window results
            elif bin_start == 0 and bin_end == 1:
                results_electrode[electrode_label]["one_second_after_onset"] = bin_results # one second after onset results
            else:
                results_electrode[electrode_label]["time_bins"].append(bin_results) # time bin results

    results = {
        "model_name": "Average Activity",
        "author": None,
        "description": "Average neural activity for each label in each time bin.",
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
    with open(f"{save_dir}/average_activity_electrode_{subject.subject_identifier}_{trial_id}_{eval_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        print(f"Results saved to {save_dir}/average_activity_electrode_{subject.subject_identifier}_{trial_id}_{eval_name}.json")
