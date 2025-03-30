from torch.utils.data import DataLoader
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from btbench.btbench_train_test_splits import generate_splits_SS_SM
from train_utils import log

# Evaluation class for Same Subject Same Movie (SS-SM), on btbench evals
class FrozenModelEvaluation_SS_SM():
    def __init__(self, eval_names, subject_trials, dtype, batch_size,
                 num_workers_eval=4, prefetch_factor=2,
                 feature_aggregation_method='concat', # 'mean', 'concat'
                 # regression parameters
                 regression_random_state=42,  regression_solver='lbfgs', 
                 regression_tol=1e-3,
                 regression_max_iter=10000):
        """
        Args:
            eval_names (list): List of evaluation metric names to use (e.g. ["volume", "word_gap"])
            subject_trials (list): List of tuples where each tuple contains (subject, trial_id).
                                 subject is a BrainTreebankSubject object and trial_id is an integer.
            dtype (torch.dtype, optional): Data type for tensors.
        """
        self.eval_names = eval_names
        self.subject_trials = subject_trials
        self.all_subjects = set([subject for subject, trial_id in self.subject_trials])
        self.all_subject_identifiers = set([subject.subject_identifier for subject in self.all_subjects])
        self.dtype = dtype
        self.batch_size = batch_size
        
        self.feature_aggregation_method = feature_aggregation_method

        self.regression_max_iter = regression_max_iter
        self.regression_random_state = regression_random_state
        self.regression_solver = regression_solver
        self.regression_tol = regression_tol
        self.num_workers_eval = num_workers_eval
        self.prefetch_factor = prefetch_factor

        evaluation_datasets = {}
        for eval_name in self.eval_names:
            for subject, trial_id in self.subject_trials:
                splits = generate_splits_SS_SM(subject, trial_id, eval_name, dtype=self.dtype)
                evaluation_datasets[(eval_name, subject.subject_identifier, trial_id)] = splits
        self.evaluation_datasets = evaluation_datasets

    def _evaluate_on_dataset(self, model, electrode_embedding_class, subject, train_dataset, test_dataset, log_priority=0):
        subject_identifier = subject.subject_identifier
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_eval, prefetch_factor=self.prefetch_factor, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_eval, prefetch_factor=self.prefetch_factor, pin_memory=True)
        device, dtype = model.device, model.dtype
        X_train, y_train = [], []
        log('generating frozen train features', priority=log_priority, indent=2)
        for i, (batch_input, batch_label) in enumerate(train_dataloader):
            log(f'generating frozen features for batch {i} of {len(train_dataloader)}', priority=log_priority, indent=3)
            batch_input = batch_input.to(device, dtype=dtype, non_blocking=True)
            electrode_embedded_data = electrode_embedding_class.forward(subject_identifier, subject.get_electrode_indices(0), batch_input)
            features = model.generate_frozen_evaluation_features(electrode_embedded_data, feature_aggregation_method=self.feature_aggregation_method).detach().cpu().float().numpy()
            log(f'done generating frozen features for batch {i} of {len(train_dataloader)}', priority=log_priority, indent=3)
            X_train.append(features)
            y_train.append(batch_label.numpy())

        X_test, y_test = [], []
        log('generating frozen test features', priority=log_priority, indent=2)
        for i, (batch_input, batch_label) in enumerate(test_dataloader):
            batch_input = batch_input.to(device, dtype=dtype)
            log(f'generating frozen features for batch {i} of {len(test_dataloader)}', priority=log_priority, indent=3)
            electrode_embedded_data = electrode_embedding_class.forward(subject_identifier, subject.get_electrode_indices(0), batch_input)
            features = model.generate_frozen_evaluation_features(electrode_embedded_data, feature_aggregation_method=self.feature_aggregation_method).detach().cpu().float().numpy()
            log(f'done generating frozen features for batch {i} of {len(test_dataloader)}', priority=log_priority, indent=3)
            X_test.append(features)
            y_test.append(batch_label.numpy())
        log('done generating frozen features', priority=log_priority, indent=2)

        log("creating numpy arrays", priority=log_priority, indent=2)
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        log("done creating numpy arrays", priority=log_priority, indent=2)

        regressor = LogisticRegression(
            random_state=self.regression_random_state, 
            max_iter=self.regression_max_iter, 
            n_jobs=self.num_workers_eval, 
            solver=self.regression_solver, 
            tol=self.regression_tol
        )

        log('fitting regressor', priority=log_priority, indent=2)
        regressor.fit(X_train, y_train)
        log('done fitting regressor', priority=log_priority, indent=2)

        # Get predictions for multiclass classification
        train_probs = regressor.predict_proba(X_train)
        test_probs = regressor.predict_proba(X_test)

        # Filter test samples to only include classes that were in training
        valid_class_mask = np.isin(y_test, regressor.classes_)
        y_test_filtered = y_test[valid_class_mask]
        test_probs_filtered = test_probs[valid_class_mask]

        # Convert to one-hot encoding
        y_test_onehot = np.zeros((len(y_test_filtered), len(regressor.classes_)))
        for i, label in enumerate(y_test_filtered):
            class_idx = np.where(regressor.classes_ == label)[0][0]
            y_test_onehot[i, class_idx] = 1

        y_train_onehot = np.zeros((len(y_train), len(regressor.classes_)))
        for i, label in enumerate(y_train):
            class_idx = np.where(regressor.classes_ == label)[0][0]
            y_train_onehot[i, class_idx] = 1

        # Calculate ROC AUC based on number of classes
        n_classes = len(regressor.classes_)
        if n_classes > 2:
            auroc = sklearn.metrics.roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
        else:
            auroc = sklearn.metrics.roc_auc_score(y_test_onehot, test_probs_filtered)

        accuracy = regressor.score(X_test, y_test)
        log('done evaluating', priority=log_priority, indent=2)
        return auroc, accuracy
    
    def _evaluate_on_metric_cv(self, model, electrode_embedding_class, subject, train_datasets, test_datasets, log_priority=0, quick_eval=False):
        auroc_list, accuracy_list = [], []
        for train_dataset, test_dataset in zip(train_datasets, test_datasets):
            auroc, accuracy = self._evaluate_on_dataset(model, electrode_embedding_class, subject, train_dataset, test_dataset, log_priority=log_priority)
            auroc_list.append(auroc)
            accuracy_list.append(accuracy)
            if quick_eval: break
        return np.mean(auroc_list), np.mean(accuracy_list)
    
    def evaluate_on_all_metrics(self, model, electrode_embedding_class, log_priority=0, quick_eval=False):
        log('evaluating on all metrics', priority=log_priority, indent=1)
        evaluation_results = {}
        for subject in self.all_subjects:
            for eval_name in self.eval_names:
                trial_ids = [trial_id for _subject, trial_id in self.subject_trials if _subject.subject_identifier == subject.subject_identifier]
                for trial_id in trial_ids:
                    splits = self.evaluation_datasets[(eval_name, subject.subject_identifier, trial_id)]
                    auroc, accuracy = self._evaluate_on_metric_cv(model, electrode_embedding_class, subject, splits[0], splits[1], log_priority=log_priority+1, quick_eval=quick_eval)
                    evaluation_results[(eval_name, subject.subject_identifier, trial_id)] = (auroc, accuracy)
        
        evaluation_results_strings = self._format_evaluation_results_strings(evaluation_results)
        log('done evaluating on all metrics', priority=log_priority, indent=1)
        return evaluation_results_strings

    def _format_evaluation_results_strings(self, evaluation_results):
        evaluation_results_strings = {}
        for eval_name in self.eval_names:
            auroc_values = []
            acc_values = []
            subject_aurocs = {}
            subject_accs = {}
            for (metric, subject_identifier, trial_id) in [key for key in evaluation_results.keys() if key[0] == eval_name]:
                if subject_identifier not in subject_aurocs:
                    subject_aurocs[subject_identifier] = []
                    subject_accs[subject_identifier] = []
                auroc, accuracy = evaluation_results[(eval_name, subject_identifier, trial_id)]
                auroc, accuracy = auroc.item(), accuracy.item()

                subject_aurocs[subject_identifier].append(auroc)
                subject_accs[subject_identifier].append(accuracy)
                evaluation_results_strings[f"eval_auroc/{subject_identifier}_{trial_id}_{eval_name}"] = auroc
                evaluation_results_strings[f"eval_acc/{subject_identifier}_{trial_id}_{eval_name}"] = accuracy
            for subject_identifier in subject_aurocs:
                auroc_values.append(np.mean(subject_aurocs[subject_identifier]).item())
                acc_values.append(np.mean(subject_accs[subject_identifier]).item())
            if len(auroc_values) > 0:
                evaluation_results_strings[f"eval_auroc/average_{eval_name}"] = np.mean(auroc_values).item()
                evaluation_results_strings[f"eval_acc/average_{eval_name}"] = np.mean(acc_values).item()
        return evaluation_results_strings