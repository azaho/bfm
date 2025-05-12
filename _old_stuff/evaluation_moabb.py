from torch.utils.data import DataLoader
import sklearn 
import numpy as np, torch
import logging
import moabb
from moabb.datasets import BNCI2014_001, Weibo2014, Cho2017, GrosseWentrup2009, Lee2019_MI, Liu2024
from moabb.evaluations import WithinSessionEvaluation
from moabb_paradigm.motor_imagery import LeftRightImagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
import os, mne, warnings
from sklearn.model_selection import KFold
from train_utils import log
from sklearn.metrics import roc_auc_score

moabb_default_datasets = [Weibo2014]#[BNCI2014_001, Weibo2014]

# Cache directory setup
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moabb_datasets/")
mne.set_config("MNE_DATA", cache_dir)
moabb.utils.set_download_dir(cache_dir)

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# Evaluation class for frozen model evaluation on MOABB datasets
class FrozenModelEvaluation_MOABB():
    def __init__(self, eeg_electrode_names, 
                 sampling_rate, # XXX change to infer this from the model / change the resampling strategy
                 max_subjects=3, 
                 moabb_datasets=moabb_default_datasets):
        """
        Args:
            eeg_electrode_names: List of strings, each corresponding to each recorded EEG channel name.
            sampling_rate: Sampling rate of the EEG data.
            max_subjects: Maximum number of subjects to use for evaluation.
            moabb_datasets: List of MOABB dataset classes to use for evaluation.
        """
        # Store configs
        self.max_subjects = max_subjects
        self.moabb_datasets = moabb_datasets
        self.eeg_electrode_names = eeg_electrode_names
        self.sampling_rate = sampling_rate
        # Set up datasets
        self.datasets = []
        for dataset_class in moabb_datasets:
            dataset = dataset_class()
            if max_subjects:
                dataset.subject_list = dataset.subject_list[:max_subjects]
            self.datasets.append(dataset)

    class ModelTransform:
        def __init__(self, model, electrode_embedding_class, sampling_rate, feature_aggregation_method='concat'):
            self.is_fitted_ = False
            self.model = model
            self.electrode_embedding_class = electrode_embedding_class
            self.feature_aggregation_method = feature_aggregation_method
            self.device = model.device
            self.dtype = model.dtype
            self.sampling_rate = sampling_rate

            self.normalization_means = None
            self.normalization_stds = None

        def transform(self, X):
            if not self.is_fitted_:
                raise ValueError("ModelTransform has not been fitted yet.")
            
            # Convert to torch tensor and move to device
            X = torch.tensor(X, device=self.device, dtype=self.dtype)

            batch_size, n_electrodes, n_samples = X.shape
            timebin_samples = int(self.electrode_embedding_class.sample_timebin_size * self.sampling_rate)
            X = X[:, :, :-(n_samples % timebin_samples)] # trim the last timebin if it's not a full timebin
            
            # Get electrode embeddings and features
            with torch.no_grad():
                electrode_embedded_data = self.electrode_embedding_class.forward('moabb', range(X.shape[1]), X, 
                                            normalization_means=self.normalization_means, normalization_stds=self.normalization_stds)
                features = self.model.generate_frozen_evaluation_features(
                    electrode_embedded_data, 
                    feature_aggregation_method=self.feature_aggregation_method
                )
            
            return features.float().cpu().numpy()

        def fit(self, X, y):
            X = torch.tensor(X, device=self.device, dtype=self.dtype) # Shape: (batch_size, n_electrodes, n_samples)

            batch_size, n_electrodes, n_samples = X.shape
            timebin_samples = int(self.electrode_embedding_class.sample_timebin_size * self.sampling_rate)
            X = X[:, :, :-(n_samples % timebin_samples)] # trim the last timebin if it's not a full timebin

            normalization_means, normalization_stds = self.electrode_embedding_class.calculate_electrode_normalization(X, self.sampling_rate)
            self.normalization_means = normalization_means
            self.normalization_stds = normalization_stds
            self.is_fitted_ = True
            return self

        def fit_transform(self, X, y):
            return self.fit(X, y).transform(X)

        def get_params(self, deep=True):
            return {
                'model': self.model,
                'electrode_embedding_class': self.electrode_embedding_class,
                'feature_aggregation_method': self.feature_aggregation_method,
                'normalization_means': self.normalization_means,
                'normalization_stds': self.normalization_stds
            }

        def set_params(self, **params):
            for param, value in params.items():
                setattr(self, param, value)
            return self

    def evaluate_on_all_metrics(self, model, electrode_embedding_class, log_priority=1, quick_eval=True): # XXX log priority and quick eval are not used
        # Set up paradigm
        paradigm = LeftRightImagery(
            resample=self.sampling_rate,
            fmin=0, 
            fmax=99 # XXX hardcoded
        )
        paradigm.set_channel_subset(self.eeg_electrode_names)

        # Create pipelines
        pipelines = {
            "model": make_pipeline(
                self.ModelTransform(model, electrode_embedding_class, self.sampling_rate, 'concat'),
                LogisticRegression()
            ),
            "raw": make_pipeline(
                FunctionTransformer(lambda x: x.reshape(x.shape[0], -1)),
                LogisticRegression()
            ),
            "csp": make_pipeline(
                CSP(n_components=8, reg=0.1),
                LDA()
            )
        }

        # Run evaluation manually
        results = []
        for dataset in self.datasets:
            # For each subject
            for subject in dataset.subject_list:
                # Get data for this subject
                X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

                # For each session
                for session in np.unique(metadata.session):
                    session_mask = metadata.session == session
                    X_session = X[session_mask]
                    y_session = y[session_mask]
                    
                    # 5-fold cross validation with shuffling
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    # Process results for each pipeline
                    for pipeline_name, pipeline in pipelines.items():
                        scores = []
                        for train_indices, test_indices in cv.split(X_session):
                            # Split data using sklearn CV indices
                            X_train = X_session[train_indices]
                            X_test = X_session[test_indices]
                            y_train = y_session[train_indices]
                            y_test = y_session[test_indices]
                            
                            # Fit and score the pipeline
                            pipeline.fit(X_train, y_train)
                            score = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
                            scores.append(score)
                            
                            # Store results
                            results.append({
                                'dataset': dataset.code,
                                'subject': subject,
                                'session': session,
                                'pipeline': pipeline_name,
                                'score': score
                            })
                            
                        # Log average score for this pipeline/subject/session
                        avg_score = np.mean(scores)
                    
                        log(f"Dataset: {dataset.code}, Subject: {subject}, Session: {session}", priority=log_priority)
                        log(f"Pipeline: {pipeline_name}, Average Score: {avg_score:.3f}", priority=log_priority)
        return self._convert_results_to_evaluation_results_strings(results)


    def _convert_results_to_evaluation_results_strings(self, results):
        # Process results into evaluation_results_strings format
        evaluation_results_strings = {}
        
        # Group by dataset and pipeline
        # Group results by dataset
        datasets = set(r['dataset'] for r in results)
        for dataset in datasets:
            # Group by pipeline
            pipelines = set(r['pipeline'] for r in results if r['dataset'] == dataset)
            for pipeline in pipelines:
                # Get all results for this dataset/pipeline combination
                dataset_pipeline_results = [r for r in results 
                                         if r['dataset'] == dataset and r['pipeline'] == pipeline]
                
                # Calculate per subject-session scores
                subjects = set(r['subject'] for r in dataset_pipeline_results)
                for subject in subjects:
                    subject_results = [r for r in dataset_pipeline_results 
                                     if r['subject'] == subject]
                    
                    for result in subject_results:
                        session_key = str(result['session'])
                        result_key = f"eval_auroc/{dataset}_{subject}_{session_key}_{pipeline}"
                        evaluation_results_strings[result_key] = result['score']

                # Calculate average score for this dataset/pipeline
                scores = [r['score'] for r in dataset_pipeline_results]
                avg_score = sum(scores) / len(scores)
                evaluation_results_strings[f"eval_auroc/average_{dataset}_{pipeline}"] = avg_score

        # Calculate average scores across all datasets per pipeline
        all_pipelines = set(r['pipeline'] for r in results)
        for pipeline in all_pipelines:
            pipeline_results = [r for r in results if r['pipeline'] == pipeline]
            scores = [r['score'] for r in pipeline_results]
            avg_score = sum(scores) / len(scores)
            evaluation_results_strings[f"eval_auroc/average_all_datasets_{pipeline}"] = avg_score

        return evaluation_results_strings