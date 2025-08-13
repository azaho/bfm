import evaluation.neuroprobe.train_test_splits as neuroprobe_train_test_splits
import evaluation.neuroprobe.config as neuroprobe_config
from evaluation.neuroprobe.braintreebank_subject import BrainTreebankSubject

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time
import gc

from eval_utils import *

from training_setup.training_config import log, parse_subject_trials_from_config, unconvert_dtypes, convert_dtypes
from subject.dataset import load_subjects

splits_options = [
    'SS_SM', # same subject, same trial
    'SS_DM', # same subject, different trial    
    'DS_DM', # different subject, different trial
]

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--split_type', type=str, choices=splits_options, default='SS_SM', help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')

parser.add_argument('--silent', action='store_true', help='Whether to suppress progress messages')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--full', action='store_true', help='Whether to use the full eval for Neuroprobe (NOTE: Lite is the default!)')
parser.add_argument('--nano', action='store_true', help='Whether to use Neuroprobe Nano for faster evaluation')

parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch of the model to load')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for feature computation')

parser.add_argument('--feature_type', type=str, default='keepall', help='How to extract features from the model. Options: \'meanE\' (mean across electrodes), \'meanT\' (mean across timebins), \'cls\' (only take the first token of the electrode dimension), any combinations of these (you can use _ to concatenate them) or \'keepall\' (keep all tokens)')

parser.add_argument('--classifier_type', type=str, choices=['linear', 'cnn', 'transformer'], default='linear', help='Type of classifier to use for evaluation')
args = parser.parse_args()

eval_names = args.eval_name.split(',')
splits_type = args.split_type.upper()
subject_id = args.subject_id
trial_id = args.trial_id
assert splits_type != "DS_DM", "DS_DM is currently not supported"

verbose = not bool(args.silent)
overwrite = bool(args.overwrite)
save_dir = args.save_dir
seed = args.seed

only_1second = bool(args.only_1second)
lite = not bool(args.full)
nano = bool(args.nano)
assert (not nano) or (splits_type != "SS_DM"), "Nano only works with SS_SM or DS_DM splits; does not work with SS_DM."
assert (not nano) or lite, "--nano and --full cannot be used together. Neuroprobe Full and Neuroprobe Nano are different evaluations."

model_dir = args.model_dir
model_epoch = args.model_epoch
batch_size = args.batch_size
feature_type = args.feature_type

classifier_type = args.classifier_type

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)


### LOAD CONFIG ###

# Load the checkpoint
if model_epoch < 0: model_epoch = "final"
checkpoint_path = os.path.join("runs/data", model_dir, f"model_epoch_{model_epoch}.pth")
checkpoint = torch.load(checkpoint_path)
config = unconvert_dtypes(checkpoint['config'])
if verbose:
    log(f"Directory name: {model_dir}", priority=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['device'] = device
if verbose:
    log(f"Using device: {device}", priority=0)

config['training']['train_subject_trials'] = ""
config['training']['eval_subject_trials'] = f"btbank{subject_id}_{trial_id}"
parse_subject_trials_from_config(config)

if 'setup_name' not in config['training']:
    config['training']['setup_name'] = "andrii0" # XXX: this is only here for backwards compatibility, can remove soon

### LOAD SUBJECTS ###

if verbose:
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
    training_setup = setup_class(all_subjects, config, verbose=True)
except (ImportError, AttributeError) as e:
    print(f"Could not load training setup '{config['training']['setup_name']}'. Are you sure the filename and the class name are the same and correspond to the parameter? Error: {str(e)}")
    exit()

if verbose:
    log(f"Loading model...", priority=0)
training_setup.initialize_model()

if verbose:
    log(f"Loading model weights...", priority=0)
if model_epoch != 0:
    training_setup.load_model(model_epoch)

### SETUP FEATURE GENERATION FUNCTION ###

def load_dataset(dataset):
    X = []

    for item_start_i in range(0, len(dataset), batch_size):
        if verbose:
            log(f"Loading batch {item_start_i//batch_size+1} of {len(dataset)//batch_size}", priority=1, indent=2)

        item_end_i = min(item_start_i + batch_size, len(dataset))
        batch_input = torch.cat([dataset[i][0][:, data_idx_from:data_idx_to].unsqueeze(0) for i in range(item_start_i, item_end_i)], dim=0)
        batch = {
            'data': batch_input, # shape (batch_size, n_electrodes, n_samples),
            'electrode_labels': [all_electrode_labels],
            'metadata': {
                'subject_identifier': subject.subject_identifier,
                'trial_id': trial_id,
                'sampling_rate': subject.get_sampling_rate(trial_id),
            },
        }

        with torch.no_grad():
            for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
                batch = preprocess_function(batch)

            batch['data'] = batch['data'].to(training_setup.model.device, dtype=training_setup.model.dtype, non_blocking=True)
            training_setup.model.spectrogram_preprocessor.output_transform = torch.nn.Identity()
            preprocessed_input = training_setup.model.spectrogram_preprocessor(batch, mask=False)['data']
            features = preprocessed_input

            if 'meanT' in feature_type:
                features = features.mean(dim=2, keepdim=True) # shape: (batch_size, n_electrodes + 1, 1, d_model)
            if 'meanE' in feature_type:
                features = features.mean(dim=1, keepdim=True) # shape: (batch_size, 1, n_timebins, d_model)
            if 'cls' in feature_type:
                features = features[:, 0:1, :, :] # shape: (batch_size, 1, n_timebins, d_model) -- take just the cls token
                
            features = features.detach().cpu().float().numpy()
        X.append(features)
        if verbose and item_start_i == 0:
            log(f"Input shape: {batch['data'].shape}", priority=1, indent=3)
            log(f"Features shape: {features.shape}", priority=1, indent=3)
    y = [dataset[i][1] for i in range(len(dataset))]
    return np.concatenate(X, axis=0), np.array(y)

### CALCULATE TIME BINS ###

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5 if not only_1second else 1
bin_size_seconds = 0.25
bin_step_size_seconds = 0.125

bin_starts = []
bin_ends = []
if not only_1second:
    for bin_start in np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds-bin_size_seconds, bin_step_size_seconds):
        bin_end = bin_start + bin_size_seconds
        if bin_end > bins_end_after_word_onset_seconds: break

        bin_starts.append(bin_start)
        bin_ends.append(bin_end)
    bin_starts += [-bins_start_before_word_onset_seconds]
    bin_ends += [bins_end_after_word_onset_seconds]
bin_starts += [0]
bin_ends += [1]

### LOAD SUBJECT ###

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = all_subjects[f"btbank{subject_id}"]
if nano:
    all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
elif lite:
    all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
else:
    all_electrode_labels = subject.electrode_labels
subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
neural_data_loaded = False

for eval_name in eval_names:
    start_time = time.time()

    file_save_dir = f"{save_dir}/{classifier_type}_inputs_no_otf_{model_dir}_epoch{model_epoch}_{feature_type}"
    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist

    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    if os.path.exists(file_save_path) and not overwrite:
        if verbose:
            log(f"Skipping {file_save_path} because it already exists", priority=0)
        continue

    # Load neural data if it hasn't been loaded yet; NOTE: this is done here to avoid unnecessary loading of neural data if the file is going to be skipped.
    if not neural_data_loaded:
        subject.load_neural_data(trial_id)
        subject_load_time = time.time() - start_time
        if verbose:
            log(f"Subject loaded in {subject_load_time:.2f} seconds", priority=0)
        neural_data_loaded = True

    results_population = {
        "time_bins": [],
    }

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    if splits_type == "SS_SM":
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano, allow_partial_cache=True)
    elif splits_type == "SS_DM":
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]
    elif splits_type == "DS_DM":
        if verbose: log("Loading the training subject...", priority=0)
        train_subject_id = neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        train_subject_electrodes = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier] if lite else train_subject.electrode_labels
        train_subject.set_electrode_subset(train_subject_electrodes)
        all_subjects = {
            subject_id: subject,
            train_subject_id: train_subject,
        }
        if verbose: log("Subject loaded.", priority=0)
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_DS_DM(all_subjects, subject_id, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]


    for bin_start, bin_end in zip(bin_starts, bin_ends):
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            if verbose:
                log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
                log("Preparing and preprocessing data & Generating features...", priority=1, indent=1)

            start_time = time.time()
            X_train, y_train = load_dataset(train_dataset)
            gc.collect()  # Collect after creating large arrays
            X_test, y_test = load_dataset(test_dataset)
            gc.collect()  # Collect after creating large arrays

            if splits_type == "DS_DM":
                if verbose: log("Combining regions...", priority=1, indent=1)
                regions_train = get_region_labels(train_subject)
                regions_test = get_region_labels(subject)
                X_train, X_test, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)
            features_processing_time = time.time() - start_time
            if verbose:
                log(f"Features processing time: {features_processing_time:.2f} seconds", priority=0)

            start_time = time.time()

            # Flatten the data after preprocessing in-place
            original_X_train_shape = X_train.shape
            original_X_test_shape = X_test.shape
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            if verbose:
                log(f"Standardizing data...", priority=1, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            gc.collect()  # Collect after standardization

            if verbose:
                log(f"Training model...", priority=1, indent=1)

            # Train logistic regression
            if classifier_type == 'linear':
                clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
            elif classifier_type == 'cnn':
                X_train = X_train.reshape(original_X_train_shape)
                X_test = X_test.reshape(original_X_test_shape)
                clf = CNNClassifier(random_state=seed)
            elif classifier_type == 'transformer':
                X_train = X_train.reshape(original_X_train_shape)
                X_test = X_test.reshape(original_X_test_shape)
                clf = TransformerClassifier(random_state=seed)
            clf.fit(X_train, y_train)

            torch.cuda.empty_cache()
            gc.collect()

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()  # Collect after predictions

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

            regression_run_time = time.time() - start_time
            if verbose:
                log(f"Regression run in {regression_run_time:.2f} seconds", priority=0)

            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc),
                "timing": {
                    "features_processing_time": features_processing_time,
                    "regression_run_time": regression_run_time,
                }
            }
            bin_results["folds"].append(fold_result)
            
            # Clean up variables no longer needed
            del X_train, y_train, X_test, y_test, train_probs, test_probs
            del y_test_filtered, test_probs_filtered, y_test_onehot, y_train_onehot
            del clf, scaler
            gc.collect()  # Collect after cleanup

            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results
    

    results = {
        "model_name": config['model']['name'],
        "author": "Andrii Zahorodnii",
        "description": f"Simple {config['model']['name']} using all electrodes ({feature_type}).",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },

        "config": {
            "model_dir": model_dir,
            "model_epoch": model_epoch,
            "model_config": convert_dtypes(config),
            "feature_type": feature_type,
            "only_1second": only_1second,
            "seed": seed,
            "subject_id": subject_id,
            "trial_id": trial_id,
            "splits_type": splits_type,
            "classifier_type": classifier_type,
        },

        "timing": {
            "subject_load_time": subject_load_time,
        }
    }

    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)

    # Clean up at end of each eval_name loop
    del train_datasets, test_datasets
    gc.collect()