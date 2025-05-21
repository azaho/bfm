from btbench.braintreebank_subject import BrainTreebankSubject
import btbench.btbench_train_test_splits as btbench_train_test_splits
import btbench.btbench_config as btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time, psutil
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--splits_type', type=str, choices=['SS_SM', 'SS_DM'], default='SS_SM', help='Type of splits to use (SS_SM or DM_SM)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--lite', action='store_true', help='Whether to use the lite eval for BTBench (which is the default)')
parser.add_argument('--time_granularity', type=int, default=1, help='Time granularity of the BrainBERT features (how many bins within a window)')

parser.add_argument('--model_dirname', type=str, default=None, help='Directory name containing the model checkpoint')
parser.add_argument('--model_epoch', type=int, default=None, help='Epoch of the model checkpoint to load')

args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
splits_type = args.splits_type
seed = args.seed
only_1second = bool(args.only_1second)
lite = bool(args.lite)
time_granularity = args.time_granularity
model_dirname = args.model_dirname
model_epoch = args.model_epoch


# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 2.5 if not only_1second else 1
bin_size_seconds = 0.125

if not only_1second:
    # Loop over all time bins
    bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
    bin_ends = bin_starts + bin_size_seconds
    # Add a time bin for the whole window and for 1 second after the word onset
    bin_starts = [0, -bins_start_before_word_onset_seconds][::-1] + list(bin_starts)
    bin_ends = [1, bins_end_after_word_onset_seconds][::-1] + list(bin_ends)
else:
    bin_starts = [0]
    bin_ends = [1]


max_log_priority = -1 if not verbose else 4
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Using device: {device}", priority=0)

log("Loading model...", priority=0)
# Construct the model path
model_path = os.path.join("models_data/", model_dirname, f'model_epoch_{model_epoch}.pth')
# Load the model dictionary
from train_utils import unconvert_dtypes
model_dict = torch.load(model_path, map_location=device)
training_config = unconvert_dtypes(model_dict['training_config'])
model_config = unconvert_dtypes(model_dict['model_config'])
cluster_config = unconvert_dtypes(model_dict['cluster_config'])

n_downsample_factor = 1

from model_model import LinearBinTransformer, BinTransformer, LinearKernelTransformer
# log(f"Loading model...", priority=0)
# if model_config['bin_encoder'] == "linear":
#     bin_embed_transformer = LinearBinTransformer(
#         overall_sampling_rate=2048,
#         sample_timebin_size=model_config['sample_timebin_size'],
#         identity_init=model_config['init_identity']
#     )
# elif model_config['bin_encoder'] == "transformer":
#     bin_embed_transformer = BinTransformer(
#         d_input=model_config['first_kernel'],
#         d_model=model_config['transformer']['d_model'],
#         n_layers=model_config['transformer']['n_layers_electrode'],
#         n_heads=12,
#         overall_sampling_rate=2048,
#         sample_timebin_size=model_config['sample_timebin_size'],
#     ).to(device, dtype=model_config['dtype'])
#     bin_unembed_transformer = bin_embed_transformer
    
# bin_unembed_transformer = bin_embed_transformer
# if model_config['separate_unembed']:
#     bin_unembed_transformer = LinearKernelTransformer(
#         d_input=model_config['first_kernel'],
#         d_output=model_config['transformer']['d_model'],
#     )
# bin_unembed_transformer = bin_unembed_transformer.to(device, dtype=model_config['dtype'])
# bin_embed_transformer = bin_embed_transformer.to(device, dtype=model_config['dtype'])

log(f"Loading model...", priority=0)
if model_config['bin_encoder'] == "linear":
    bin_embed_transformer = LinearBinTransformer(
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
        identity_init=model_config['init_identity']
    )
elif model_config['bin_encoder'] == "transformer":
    bin_embed_transformer = BinTransformer(
        d_input=model_config['first_kernel'],
        d_model=model_config['transformer']['d_model'],
        n_layers=model_config['transformer']['n_layers_electrode'],
        n_heads=12,
        overall_sampling_rate=2048,
        sample_timebin_size=model_config['sample_timebin_size'],
    ).to(device, dtype=model_config['dtype'])
    bin_unembed_transformer = bin_embed_transformer
bin_embed_transformer = bin_embed_transformer.to(device, dtype=model_config['dtype'])

bin_embed_transformer.load_state_dict(model_dict['bin_embed_transformer_state_dict'])

def get_model_features(x, fs=2048, max_batch_size=100):
    """
    x: np.ndarray of shape (batch_size, n_channels, n_samples)
    returns: np.ndarray of shape (batch_size, n_channels, n_timebins, sample_timebin_size)
    """
    assert len(x.shape) == 3, "x must be of shape (batch_size, n_channels, n_samples)"
    
    batch_size, n_channels, n_samples = x.shape
    out_array = None     # will hold the full output
    idx_start = 0        # pointer into out_array

    # normalizing the features
    log(f"Normalizing features", priority=3, indent=1)
    x = x - torch.mean(x, dim=[0, 2], keepdim=True)
    x = x / (torch.std(x, dim=[0, 2], keepdim=True) + 1)

    with torch.no_grad():
        for i in range(0, batch_size, max_batch_size):
            log(f"Processing batch {i//max_batch_size+1} of {(batch_size-1)//max_batch_size+1}", priority=3, indent=2)
            batch_x = x[i:i+max_batch_size].to(device, dtype=model_config['dtype']) # shape: (batch_i, n_channels, n_samples)

            electrode_data = bin_embed_transformer(batch_x) # shape: (batch_i, n_channels, n_timebins, sample_timebin_size)
            batch_i, n_channels, n_timebins, sample_timebin_size = electrode_data.shape

            log(f"Converting to numpy", priority=4, indent=3)
            batch_np = electrode_data.cpu().float().numpy()

            # On first batch, allocate the big output array
            if out_array is None:
                out_array = np.empty((batch_size, n_channels, n_timebins, sample_timebin_size), dtype=batch_np.dtype)

            # fill slice
            idx_end = idx_start + batch_i
            out_array[idx_start:idx_end] = batch_np
            idx_start = idx_end

            # cleanup - add f and t to the list of variables to delete
            del batch_np, batch_x, electrode_data
            torch.cuda.empty_cache()

    return out_array


# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels

for eval_name in eval_names:
    file_save_dir = f"{save_dir}/{model_dirname}/frozen_bin_epoch{model_epoch}"
    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    # Check if the file already exists, and if so, skip this evaluation
    if os.path.exists(file_save_path):
        log(f"File {file_save_path} already exists. Skipping {eval_name} for subject {subject.subject_identifier}, trial {trial_id}.")
        continue

    results_population = {
        "time_bins": [],
    }

    # Load all electrodes at once
    subject.clear_neural_data_cache()
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    if verbose:
        log("Subject loaded", priority=0)

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    if splits_type == "SS_SM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
    elif splits_type == "SS_DM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, max_other_trials=3, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]

    for bin_start, bin_end in zip(bin_starts, bin_ends):
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
            log("Preparing data...", priority=2, indent=1)

            # Convert PyTorch dataset to numpy arrays for scikit-learn
            X_train = torch.cat([item[0][:, data_idx_from:data_idx_to].unsqueeze(0) for item in train_dataset])
            y_train = np.array([item[1] for item in train_dataset])
            X_test = torch.cat([item[0][:, data_idx_from:data_idx_to].unsqueeze(0) for item in test_dataset])
            y_test = np.array([item[1] for item in test_dataset])

            log("Computing Model features...", priority=2, indent=1)
            X_train = get_model_features(X_train)
            X_test = get_model_features(X_test)

            # Flatten the data after preprocessing in-place
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            log(f"Standardizing data...", priority=2, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            log(f"Training model...", priority=2, indent=1)

            log(f"Fitting model...", priority=3, indent=2)
            # Train logistic regression
            clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)
            gc.collect()

            log(f"Evaluating model...", priority=3, indent=2)
            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            gc.collect()

            log(f"Getting predictions...", priority=3, indent=2)
            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()

            log(f"Filtering test samples...", priority=3, indent=2)
            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            log(f"Converting y_test to one-hot encoding...", priority=3, indent=2)
            # Convert y_test to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1

            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            log(f"Calculating ROC AUC...", priority=3, indent=2)
            # For multiclass ROC AUC, we need to calculate the score for each class
            n_classes = len(clf.classes_)
            if n_classes > 2:
                train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
            else:
                train_roc = roc_auc_score(y_train_onehot, train_probs)
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

            log(f"Saving results...", priority=3, indent=2)
            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc)
            }
            bin_results["folds"].append(fold_result)
            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results

        del X_train, X_test, y_train, y_test
        gc.collect()

    results = {
        "model_name": f"{model_dirname} (frozen) + logistic regression",
        "author": "Andrii Zahorodnii",
        "description": f"{model_dirname} features (frozen) + logistic regression, all electrodes' features concatenated, mean across all timesteps within a timebin (splitting the window into {time_granularity} timebins).",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },

        "only_1second": only_1second,
        "seed": seed
    }

    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist
    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)