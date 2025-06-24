from subject.mgh2024 import MGH2024Subject
import numpy as np
import torch


class MGHSeizureDataset(torch.utils.data.Dataset):
    def __init__(self, subject, session_id, 
                 WINDOW_SIZE=3, SEIZURE_REGION_SIZE=20, ANNOTATION_PADDING=60*10,
                 output_index=False):
        self.subject = subject
        self.subject_id = subject.subject_id
        self.session_id = session_id
        self.output_index = output_index

        subject.load_neural_data(session_id)

        self.WINDOW_SIZE = WINDOW_SIZE

        annotations_clean = self.subject.get_annotations(session_id=session_id, window_from=0, window_to=None, remove_persyst=True)
        annotations_all = self.subject.get_annotations(session_id=session_id, window_from=0, window_to=None, remove_persyst=False, remove_cashlab=False, remove_software_changes=False)

        annotation_onsets = annotations_all[0] # array of seconds
        session_start = 0 # seconds
        session_end = self.subject.session_metadata[session_id]['session_length'] # seconds

        seizure_keywords = ["seizure", "ictal", "onset"]#"interictal", "onset", "start", "end", "offset"]
        seizure_onsets = np.array([annotation_onset for annotation_onset, annotation_description in zip(*annotations_clean) if any(x in annotation_description.lower() for x in seizure_keywords)])

        # For each window, check if it's far enough from all annotations
        far_from_all_annotations = []
        seizure_windows = []
        for window_start in np.arange(session_start, session_end, WINDOW_SIZE):
            window_end = window_start + WINDOW_SIZE
            # Check distance to all annotations
            min_distance_any_annotation = min(min(abs(annotation_onsets - window_start)), min(abs(annotation_onsets - window_end)))
            min_distance_seizure_annotations = min(min(abs(seizure_onsets - window_start)), min(abs(seizure_onsets - window_end)))
            
            # If window is far enough from all annotations, add it to far_from_all_annotations
            if min_distance_any_annotation >= ANNOTATION_PADDING:
                far_from_all_annotations.append(window_start)
            # If window is within seizure region size, add it to seizure_windows
            if min_distance_seizure_annotations <= SEIZURE_REGION_SIZE:
                seizure_windows.append(window_start)

        self.far_from_all_annotations = np.array(far_from_all_annotations)
        self.seizure_windows = np.array(seizure_windows)
        
        # Rebalance classes by randomly sampling from far_from_all_annotations to match seizure_windows length
        if len(self.far_from_all_annotations) > len(self.seizure_windows):
            self.far_from_all_annotations = np.random.choice(self.far_from_all_annotations, size=len(self.seizure_windows), replace=False)
    
    def __getitem__(self, idx):
        label = idx % 2
        if label == 0:
            index = self.far_from_all_annotations[idx // 2]
        else:
            index = self.seizure_windows[idx // 2]

        window_start = int(index * self.subject.get_sampling_rate(self.session_id))
        window_end = window_start + int(self.WINDOW_SIZE * self.subject.get_sampling_rate(self.session_id))
        if self.output_index:
            return (window_start, window_end), label

        neural_data = self.subject.get_all_electrode_data(session_id=self.session_id, window_from=window_start, window_to=window_end)
        return neural_data, label

    def __len__(self):
        return len(self.seizure_windows) + len(self.far_from_all_annotations)
    

from torch.utils.data import DataLoader
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import gc
import torch.cuda

class FrozenModelSeizureEvaluation:
    def __init__(self, subjects, session_ids, dtype, batch_size,
                 num_workers_eval=4, prefetch_factor=2,
                 # regression parameters
                 regression_random_state=42, regression_solver='lbfgs',
                 regression_tol=1e-3, regression_max_iter=10000,
                 window_size=3, seizure_region_size=20, annotation_padding=60*10,
                 max_float_precision=3):
        """
        Args:
            subjects (list): List of MGHSubject objects
            session_ids (list): List of session IDs to evaluate on
            dtype (torch.dtype): Data type for tensors
            batch_size (int): Batch size for evaluation
        """
        self.subjects = subjects
        self.session_ids = session_ids
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_float_precision = max_float_precision
        
        # Dataset parameters
        self.window_size = window_size
        self.seizure_region_size = seizure_region_size
        self.annotation_padding = annotation_padding

        # Evaluation parameters
        self.regression_max_iter = regression_max_iter
        self.regression_random_state = regression_random_state
        self.regression_solver = regression_solver
        self.regression_tol = regression_tol
        self.num_workers_eval = num_workers_eval
        self.prefetch_factor = prefetch_factor

        # Initialize evaluation datasets
        self.evaluation_datasets = {}
        from evaluation.mgh2024_seizure import MGHSeizureDataset
        for subject in self.subjects:
            for session_id in self.session_ids:
                try:
                    dataset = MGHSeizureDataset(
                        subject, session_id,
                        WINDOW_SIZE=self.window_size,
                        SEIZURE_REGION_SIZE=self.seizure_region_size,
                        ANNOTATION_PADDING=self.annotation_padding
                    )
                    self.evaluation_datasets[(subject.subject_id, session_id)] = dataset
                except Exception as e:
                    print(f"Could not create dataset for subject {subject.subject_id}, session {session_id}: {e}")

    def _generate_frozen_features(self, dataloader, model, log_priority=0):
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

        X, y = [], []
        for i, (batch_input, batch_label) in enumerate(dataloader):
            print(f'Generating frozen features for batch {i} of {len(dataloader)}')
            batch_input = batch_input.to(device, dtype=dtype, non_blocking=True)
            
            # Get features from the model
            with torch.no_grad():
                features = model(batch_input)
                
            X.append(features.detach().cpu().float().numpy())
            y.append(batch_label.numpy())
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del features, batch_input

        return np.concatenate(X), np.concatenate(y)

    def _evaluate_on_dataset(self, model, dataset, train_ratio=0.8, log_priority=0):
        # Split dataset into train and test
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers_eval, prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers_eval, prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )

        # Generate features
        X_train, y_train = self._generate_frozen_features(train_dataloader, model, log_priority)
        X_test, y_test = self._generate_frozen_features(test_dataloader, model, log_priority)

        # Clear dataloaders
        del train_dataloader, test_dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train logistic regression
        regressor = LogisticRegression(
            random_state=self.regression_random_state,
            max_iter=self.regression_max_iter,
            n_jobs=self.num_workers_eval,
            solver=self.regression_solver,
            tol=self.regression_tol
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        regressor.fit(X_train, y_train)
        
        # Get predictions and metrics
        test_probs = regressor.predict_proba(X_test)
        y_test_onehot = np.zeros((len(y_test), len(regressor.classes_)))
        for i, label in enumerate(y_test):
            class_idx = np.where(regressor.classes_ == label)[0][0]
            y_test_onehot[i, class_idx] = 1

        auroc = sklearn.metrics.roc_auc_score(y_test_onehot, test_probs)
        accuracy = regressor.score(X_test, y_test)
        
        return auroc, accuracy

    def evaluate_model(self, model, log_priority=4):
        evaluation_results = {}
        
        for (subject_id, session_id), dataset in self.evaluation_datasets.items():
            print(f"Evaluating subject {subject_id}, session {session_id}")
            try:
                auroc, accuracy = self._evaluate_on_dataset(model, dataset, log_priority=log_priority)
                evaluation_results[(subject_id, session_id)] = (auroc, accuracy)
            except Exception as e:
                print(f"Error evaluating subject {subject_id}, session {session_id}: {e}")
                continue

        # Format results
        formatted_results = {}
        all_aurocs = []
        all_accuracies = []

        for (subject_id, session_id), (auroc, accuracy) in evaluation_results.items():
            key_base = f"subject_{subject_id}_session_{session_id}"
            formatted_results[f"eval_auroc/{key_base}"] = round(auroc, self.max_float_precision)
            formatted_results[f"eval_acc/{key_base}"] = round(accuracy, self.max_float_precision)
            all_aurocs.append(auroc)
            all_accuracies.append(accuracy)

        if all_aurocs:
            formatted_results["eval_auroc/average_overall"] = round(np.mean(all_aurocs), self.max_float_precision)
            formatted_results["eval_acc/average_overall"] = round(np.mean(all_accuracies), self.max_float_precision)

        return formatted_results


if __name__ == "__main__":
    subject_id, session_id = 14, 0
    subject = MGHSubject(subject_id=subject_id, cache=False)
    dataset = MGHSeizureDataset(subject, session_id)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1])