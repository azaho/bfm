import gc
from collections import defaultdict

import numpy as np
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import gc

import torch
import torch.cuda
from torch.utils.data import DataLoader

from training_setup.training_config import log
from evaluation.neuroprobe.train_test_splits import generate_splits_SS_SM
import evaluation.neuroprobe.config as neuroprobe_config
from evaluation.neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset


########################## FEATURE EXTRACTION ##########################


def apply_feature_aggregation(features, aggregation_method):
    if 'meanT' in aggregation_method:
        features = features.mean(dim=2, keepdim=True) # shape: (batch_size, n_electrodes + 1, 1, d_model)
    if 'meanE' in aggregation_method:
        features = features.mean(dim=1, keepdim=True) # shape: (batch_size, 1, n_timebins, d_model)
    if 'cls' in aggregation_method:
        features = features[:, 0:1, :, :] # shape: (batch_size, 1, n_timebins, d_model) -- take just the cls token
    return features

class NeuroprobeFrozenFeaturesExtractor():
    def __init__(self,
                 # Model preprocess and evaluation function
                 training_setup, all_subjects,
                 device, dtype,
                 # Benchmark parameters. if None, will use all subjects and trials and eval_names in Neuroprobe
                 eval_names=None, subject_trials=None,
                 feature_aggregation_method=None,
                 # misc
                 log_priority=3,
                 batch_size=100):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.log_priority = log_priority
        self.eval_names = eval_names if eval_names is not None else neuroprobe_config.NEUROPROBE_TASKS
        self.subject_trials = subject_trials if subject_trials is not None else neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
        assert all(f"btbank{subject_id}" in all_subjects for subject_id, trial_id in self.subject_trials), "All subjects must be in all_subjects."
        self.feature_aggregation_method = feature_aggregation_method

        log(f"Generating Neuroprobe Eval indices for all subjects and trials", priority=self.log_priority, indent=0)
        self.subject_indices = {} # subject_identifier -> list of indices for all eval_names
        self.subject_trial_task_labels = defaultdict(lambda: defaultdict(dict)) # subject_identifier_trial_id -> eval_name -> index -> label
        for subject_id, trial_id in self.subject_trials:
            subject = all_subjects["btbank" + str(subject_id)]

            indices = []
            for eval_name in self.eval_names:
                dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, torch.float32, eval_name, output_indices=True)
                for i, ((index_start, index_end), label) in enumerate(dataset):
                    indices.append(index_start)
                    self.subject_trial_task_labels[subject.subject_identifier + "_" + str(trial_id)][eval_name][index_start] = label
            indices = sorted(list(set(indices)))
            self.subject_indices[subject.subject_identifier] = torch.tensor(indices)
            log(f"Generated {len(indices)} indices for btbank{subject_id}_{trial_id}", priority=self.log_priority, indent=1)
        
    def generate_frozen_features_for_subject_trial(self, subject_identifier, trial_id):
        """
        Args:
            subject_identifier (str): Subject identifier.
            trial_id (int): Trial ID.
        Returns:
            torch.Tensor: Frozen features of shape: (n_indices, n_electrodes, n_timebins, dim_feature).
        """
        frozen_features = None

        subject = self.all_subjects[subject_identifier]
        electrode_subset_indices = [subject.electrode_labels.index(e) for e in neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject_identifier]]

        indices = self.subject_indices[subject_identifier]
        for i_start in range(0, len(indices), self.batch_size):
            log(f"Generating features for batch {i_start} of {len(indices)}", priority=self.log_priority+1, indent=1)
            i_end = min(i_start + self.batch_size, len(indices))
            indices_batch = indices[i_start:i_end]

            model_input = torch.zeros((len(indices_batch), len(electrode_subset_indices), window_size), device=self.device, dtype=self.dtype)
            for i_index, index in enumerate(indices_batch):
                model_input[i_index, :, :] = subject.get_all_electrode_data(trial_id, window_from=index, window_to=index+window_size)[electrode_subset_indices, :]

            batch = {
                'data': model_input, # shape (batch_size, n_electrodes, n_samples),
                'electrode_labels': [neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject_identifier]],
                'metadata': {
                    'subject_identifier': subject_identifier,
                    'trial_id': trial_id,
                    'sampling_rate': subject.get_sampling_rate(trial_id),
                },
            }

            with torch.no_grad():
                for preprocess_function in training_setup.get_preprocess_functions(pretraining=False):
                    batch = preprocess_function(batch)
                features = training_setup.generate_frozen_features(batch)
                if self.feature_aggregation_method is not None:
                    features = apply_feature_aggregation(features, self.feature_aggregation_method)

            if frozen_features is None: frozen_features = torch.zeros((len(indices), *features.shape[1:]), device=self.device, dtype=self.dtype)
            frozen_features[i_start:i_end, :] = features
        return frozen_features
    
    def generate_frozen_features(self, subject_trials=None, save_path=None):
        if subject_trials is None: subject_trials = self.subject_trials
        frozen_features = {} # subject_identifier_trial_id -> features: frozen_features, indices: indices, labels: eval: labels
        for subject_id, trial_id in subject_trials:
            subject_identifier = "btbank" + subject_id
            subject_trial_frozen_features = self.generate_frozen_features_for_subject_trial(subject_identifier, trial_id)
            frozen_features[subject_identifier + "_" + str(trial_id)] = {
                "features": subject_trial_frozen_features,
                "indices": self.subject_indices[subject_identifier],
                "labels": self.subject_trial_task_labels[subject_identifier + "_" + str(trial_id)],
            }

        if save_path is not None:
            torch.save(frozen_features, save_path)
        return frozen_features

if __name__ == "__main__":
    all_subjects = {}
    from subject.braintreebank import BrainTreebankSubject
    for subject_id, trial_id in neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS:
        if subject_id not in all_subjects:
            all_subjects["btbank" + str(subject_id)] = BrainTreebankSubject(subject_id, cache=False)

    feature_extractor = NeuroprobeFrozenFeaturesExtractor(
        training_setup=None,
        all_subjects=all_subjects,
        device='cpu',
        dtype=torch.float32,
        feature_aggregation_method=None,
        log_priority=0,
        batch_size=100,
    )


############## REGION AVERAGING (FOR DS/DM SPLITS) ###############


def get_region_labels(subject, electrode_labels):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    electrode_indices = [subject.electrode_labels.index(e) for e in electrode_labels]
    return subject.get_all_electrode_metadata()['DesikanKilliany'].to_numpy()[electrode_indices]

def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)
    
    d_model_dimension_unsqueezed = False
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        d_model_dimension_unsqueezed = True

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)

    if d_model_dimension_unsqueezed: # remove the dummy dimension
        X_train_regions = X_train_regions[:, :, :, 0]
        X_test_regions = X_test_regions[:, :, :, 0]
    
    return X_train_regions, X_test_regions, common_regions


########################## EVALUATION ##########################


def prepare_frozen_features_fold_data(frozen_features, eval_name, train_subject, train_trial_id, test_subject, test_trial_id, feature_aggregation_method=None):
    """
    Args:
        frozen_features: torch.Tensor of shape (n_indices, n_electrodes, n_timebins, dim_feature)
        eval_name: str, one of neuroprobe_config.NEUROPROBE_TASKS
        train_subject: BrainTreebankSubject
        train_trial_id: int
        test_subject: BrainTreebankSubject
        test_trial_id: int
        feature_aggregation_method: str, one of ['meanT', 'meanE', 'cls']. if None, it is 'keepall' (i.e. no aggregation)
    """
    # Check that required keys exist in frozen_features
    train_key = f"{train_subject.subject_identifier}_{train_trial_id}"
    test_key = f"{test_subject.subject_identifier}_{test_trial_id}"
    assert train_key in frozen_features, f"Training data not found for subject {train_subject.subject_identifier}, trial {train_trial_id} (key: {train_key} absent from frozen_features)"
    assert test_key in frozen_features, f"Test data not found for subject {test_subject.subject_identifier}, trial {test_trial_id} (key: {test_key} absent from frozen_features)"
    # Check that eval_name exists in labels
    assert eval_name in frozen_features[train_key]["labels"], f"Evaluation task {eval_name} not found in training data (key: {eval_name} absent from frozen_features[{train_key}]['labels'])"
    assert eval_name in frozen_features[test_key]["labels"], f"Evaluation task {eval_name} not found in test data (key: {eval_name} absent from frozen_features[{test_key}]['labels'])"

    # Prepare fold data. Note that only if the train and test subject-trial are the same, we use cross-validation.
    fold_data = []
    if (train_subject.subject_identifier, train_trial_id) == (test_subject.subject_identifier, test_trial_id): # within session
        indices = frozen_features[train_key]["indices"] # shape: (n_indices,)
        task_indices = [i for i in indices if i in frozen_features[train_key]["labels"][eval_name]] # shape: (n_task_indices,)
        task_labels = [frozen_features[train_key]["labels"][eval_name][i] for i in task_indices] # shape: (n_task_indices,)

        X = frozen_features[train_key]["features"][task_indices] # shape: (n_task_indices, n_electrodes (+1), n_timebins, dim_feature)
        y = torch.tensor(task_labels) # shape: (n_task_indices,)

        if feature_aggregation_method is not None:
            X = apply_feature_aggregation(X, feature_aggregation_method)

        # Split into folds for cross-validation
        kf = KFold(n_splits=neuroprobe_config.NEUROPROBE_LITE_N_FOLDS, shuffle=False)
        fold_data = []

        for train_idx, test_idx in kf.split(X):
            X_train_fold = X[train_idx]  # shape: (n_train_indices, n_electrodes (+1), n_timebins, dim_feature) 
            X_test_fold = X[test_idx]    # shape: (n_test_indices, n_electrodes (+1), n_timebins, dim_feature)
            y_train_fold = y[train_idx]  # shape: (n_train_indices,)
            y_test_fold = y[test_idx]    # shape: (n_test_indices,)

            fold_data.append({
                "X_train": X_train_fold,
                "X_test": X_test_fold, 
                "y_train": y_train_fold,
                "y_test": y_test_fold
            })
    else:
        train_indices = frozen_features[train_key]["indices"] # shape: (n_indices,)
        test_indices = frozen_features[test_key]["indices"] # shape: (n_indices,)
        train_labels = [frozen_features[train_key]["labels"][eval_name][i] for i in train_indices] # shape: (n_indices_task,)
        test_labels = [frozen_features[test_key]["labels"][eval_name][i] for i in test_indices] # shape: (n_indices_task,)
        
        X_train = frozen_features[train_key]["features"][train_indices] # shape: (n_indices_task_train, n_electrodes_train (+1), n_timebins, dim_feature)
        X_test = frozen_features[test_key]["features"][test_indices] # shape: (n_indices_task_test, n_electrodes_test (+1), n_timebins, dim_feature)
        y_train = torch.tensor(train_labels) # shape: (n_indices_task_train,)
        y_test = torch.tensor(test_labels) # shape: (n_indices_task_test,)

        if feature_aggregation_method is not None:
            X_train = apply_feature_aggregation(X_train, feature_aggregation_method)
            X_test = apply_feature_aggregation(X_test, feature_aggregation_method)

        if train_subject.subject_identifier != test_subject.subject_identifier: # are we in the cross subject split?
            if X_train.shape[1] > 1: # do we have more than just the CLS token?
                regions_train = get_region_labels(train_subject, neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier])
                regions_test = get_region_labels(test_subject, neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[test_subject.subject_identifier])

                if X_train.shape[1] > neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier]: # do we have more electrodes than the train subject? i.e. do we have the CLS token?
                    regions_train = ['cls'] + regions_train
                    regions_test = ['cls'] + regions_test
                
                X_train, X_test, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)

        fold_data.append({
            "X_train": X_train,
            "X_test": X_test, 
            "y_train": y_train,
            "y_test": y_test
        })

    return fold_data


# Credit: Bhadra Rupesh, MIT; Andrii Zahorodnii, MIT
def eval_frozen_features_fold_data(fold_data, max_iter=10000, tol=1e-3, seed=42, log_priority=4):
    fold_results = []
    for fold in fold_data:
        start_time = time.time()

        X_train, X_test, y_train, y_test = fold["X_train"], fold["X_test"], fold["y_train"], fold["y_test"]

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        scaler = StandardScaler(copy=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        gc.collect()  # Collect after standardization

        clf = LogisticRegression(random_state=seed, max_iter=max_iter, tol=tol)
        clf.fit(X_train, y_train)
        gc.collect()

        # Evaluate model
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)

        # Get predictions - for multiclass classification
        train_probs = clf.predict_proba(X_train)
        test_probs = clf.predict_proba(X_test)
        gc.collect()  # Collect after predictions

        # Convert y_test to one-hot encoding
        y_test_onehot = np.zeros((len(y_test), len(clf.classes_)))
        for i, label in enumerate(y_test):
            class_idx = np.where(clf.classes_ == label)[0][0]
            y_test_onehot[i, class_idx] = 1

        y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
        for i, label in enumerate(y_train):
            class_idx = np.where(clf.classes_ == label)[0][0]
            y_train_onehot[i, class_idx] = 1

        train_roc = roc_auc_score(y_train_onehot, train_probs)
        test_roc = roc_auc_score(y_test_onehot, test_probs)

        regression_run_time = time.time() - start_time
        log(f"Regression run in {regression_run_time:.2f} seconds", priority=log_priority, indent=1)

        fold_result = {
            "train_accuracy": float(train_accuracy),
            "train_roc_auc": float(train_roc),
            "test_accuracy": float(test_accuracy),
            "test_roc_auc": float(test_roc),
            "timing": {
                "regression_run_time": regression_run_time,
            }
        }
        fold_results.append(fold_result)
    return fold_results