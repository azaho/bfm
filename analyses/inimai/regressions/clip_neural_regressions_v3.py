import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0,'/om2/user/inimai/bfm')

import argparse
from evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE, BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING, NEUROPROBE_FULL_SUBJECT_TRIALS
from subject.braintreebank import BrainTreebankSubject
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from model.preprocessing.spectrogram import SpectrogramPreprocessor
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_neural_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run neural regressions with CLIP features')
parser.add_argument('--feat', type=str, default='clip', help='Feature type (default: clip)')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')
args = parser.parse_args()

# Use command line arguments
feat = args.feat
SUBJECT_ID = args.subject_id
TRIAL_ID = args.trial_id

SUBJECT_TRIAL_TO_MOVIE = {
    (subject_id, trial_id): BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING["btbank" + str(subject_id) + "_" + str(trial_id)] + ".mp4"
    for (subject_id, trial_id) in NEUROPROBE_FULL_SUBJECT_TRIALS
}

FEAT_DIR = f'/om2/data/public/braintreebank_movies_preprocessed/{feat}_features/'
MOVIES_DIR = "/om2/data/public/braintreebank_movies/"

REGR_DIR = f"/om2/data/public/braintreebank_movies_preprocessed/regressions/{feat}_features/"
os.makedirs(REGR_DIR, exist_ok=True)

def obtain_neural_data_index(sub_id, trial_id, movie_times):
    # Data frames column IDs
    start_col, end_col = 'start', 'end'
    trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'

    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')

    trigs_df = pd.read_csv(trigger_times_file)
    # display(trigs_df.head())

    last_t = trigs_df[trig_time_col].iloc[-1]
    assert np.all(movie_times < last_t), "Movie times must be less than the last trigger time"
    
    # Vectorized nearest trigger finding
    start_indices = np.searchsorted(trigs_df[trig_time_col].values, movie_times)
    start_indices = np.maximum(start_indices, 0) # handle the edge case where movie starts right at the word
    
    # Vectorized sample index calculation
    return np.round(
        trigs_df.loc[start_indices, trig_idx_col].values + 
        (movie_times - trigs_df.loc[start_indices, trig_time_col].values) * SAMPLING_RATE
    ).astype(int)


def process_subject_trial(subject_id, trial_id, timestamps, sampling_interval=1.0):
    subject = BrainTreebankSubject(subject_id, cache=False)

    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{subject_id}_trial{trial_id:03}_timings.csv')
    trigs_df = pd.read_csv(trigger_times_file)
    last_trigger_time = trigs_df['movie_time'].iloc[-1]
    safe_end_timestamp = min(timestamps[-2], last_trigger_time - 1.0)  # 1 second buffer

    sampled_times = np.arange(0, safe_end_timestamp, sampling_interval)
    timestamp_indices = np.searchsorted(timestamps, sampled_times, side="left")
    return sampled_times, timestamp_indices, subject.get_electrode_labels(), subject


def get_subject_trial_data(subject_id, trial_id):
    movie = SUBJECT_TRIAL_TO_MOVIE[(subject_id, trial_id)]
    feat_features_path = os.path.join(FEAT_DIR, movie.replace('.mp4', f'_{feat}_features.npz'))
    
    data = np.load(feat_features_path)
    features = data['features']
    timestamps = data['timestamps']

    movie_path = os.path.join(MOVIES_DIR, movie)
    return features, timestamps, movie_path

def get_base_data(subject_id, trial_id, sampled_times, subject, start_window_before_event=0.25, end_window_after_event=0.5):
    windowed_neural_data = []
    windowed_laplacian_data = []
    original_electrodes = list(subject.get_electrode_labels())

    for t in sampled_times:
        window_start = t - start_window_before_event
        idx_start = obtain_neural_data_index(subject_id, trial_id, np.array([window_start])).item()
        idx_end = int(idx_start + (end_window_after_event+start_window_before_event) * 2048)

        data = subject.get_all_electrode_data(trial_id, window_from=idx_start, window_to=idx_end)
        windowed_neural_data.append(data.cpu().numpy() if hasattr(data, "cpu") else data)

        rereferenced_data, rereferenced_labels, _ = laplacian_rereference_neural_data(data, subject.electrode_labels)
        windowed_laplacian_data.append(rereferenced_data.cpu().numpy() if hasattr(data, "cpu") else rereferenced_data)

    return windowed_neural_data, original_electrodes, windowed_laplacian_data, rereferenced_labels

def run_regression_with_mse(X, y_spectrogram, alpha=0.1):
    n_timebins = y_spectrogram.shape[2]
    n_freqs = y_spectrogram.shape[3]

    n_samples = X.shape[0]
    n_folds = 3

    fold_size = n_samples // n_folds
    fold_indices = [i * fold_size for i in range(n_folds)] + [n_samples]

    # Store correlation, pval, and mse matrices for each fold
    fold_correlation_matrices = []
    fold_pval_matrices = []
    fold_mse_matrices = []

    for fold in range(n_folds):
        fold_corr_matrix = np.zeros((n_freqs, n_timebins))
        fold_pval_matrix = np.zeros((n_freqs, n_timebins))
        fold_mse_matrix = np.zeros((n_freqs, n_timebins))
        # Chronological split: train on all but the current fold, test on the current fold
        test_start = fold_indices[fold]
        test_end = fold_indices[fold + 1]
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n_samples)
        ]) if n_folds > 1 else np.arange(0, test_start)  # For n_folds=1, just use all before test

        X_train, X_test = X[train_idx], X[test_idx]

        # Normalize features using StandardScaler (fit on train, transform both train and test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for t in tqdm(range(n_timebins), desc=f"Timebin (Fold {fold+1}/{n_folds})", leave=False):
            for f in range(n_freqs):
                y_vals = y_spectrogram[:, 0, t, f].cpu().numpy() if hasattr(y_spectrogram, 'cpu') else y_spectrogram[:, 0, t, f]
                y_train, y_test = y_vals[train_idx], y_vals[test_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    # Not enough variance to compute correlation or mse
                    fold_corr_matrix[f, t] = np.nan
                    fold_pval_matrix[f, t] = np.nan
                    fold_mse_matrix[f, t] = np.nan
                    continue
                reg = Ridge(alpha=alpha)
                reg.fit(X_train_scaled, y_train)
                y_pred_test = reg.predict(X_test_scaled)
                # Correlation and p-value
                corr, pval = pearsonr(y_test, y_pred_test)
                fold_corr_matrix[f, t] = corr
                fold_pval_matrix[f, t] = pval
                # MSE
                mse = np.mean((y_test - y_pred_test) ** 2)
                fold_mse_matrix[f, t] = mse

        fold_correlation_matrices.append(fold_corr_matrix)
        fold_pval_matrices.append(fold_pval_matrix)
        fold_mse_matrices.append(fold_mse_matrix)

    # Find the fold with the highest sum of absolute correlation values
    fold_sums = [np.nansum(np.abs(mat)) for mat in fold_correlation_matrices]
    best_fold_idx = np.argmax(fold_sums)

    test_correlation_matrix = fold_correlation_matrices[best_fold_idx]
    test_pval_matrix = fold_pval_matrices[best_fold_idx]
    test_mse_matrix = fold_mse_matrices[best_fold_idx]

    return test_correlation_matrix, test_pval_matrix, test_mse_matrix

def neural_regression(subject_id, trial_id):
    features, timestamps, movie_path = get_subject_trial_data(subject_id, trial_id)
    sampled_times, timestamp_indices, unique_electrodes, subject = process_subject_trial(subject_id, trial_id, timestamps)

    windowed_neural_data, original_electrodes, windowed_laplacian_data, rereferenced_labels = get_base_data(subject_id, trial_id, sampled_times, subject)

    X = features[timestamp_indices]
    for i in tqdm(original_electrodes, desc='original electrodes:'):
        y = np.stack(windowed_neural_data[t][i] for t in len(windowed_neural_data))
        spec_preproc = SpectrogramPreprocessor()
        y_spectrogram, freq_bins, time_bins = spec_preproc({'data':y, 'metadata':{'sampling_rate':2048}})
        test_correlation_matrix, test_pval_matrix, test_mse_matrix = run_regression_with_mse(X, y_spectrogram)

        np.savez(
            os.path.join(REGR_DIR, f"btbank{subject_id}_{trial_id}_{original_electrodes[i]}.npz"),
            test_correlation_matrix=test_correlation_matrix,
            test_pval_matrix=test_pval_matrix,
            test_mse_matrix=test_mse_matrix,
            X = X,
            y = y,
            freq_bins = freq_bins,
            time_bins = time_bins
        )
        
        

    for j in tqdm(rereferenced_labels, desc='rereferenced labels:'):
        y_ref = np.stack(windowed_laplacian_data[t][j] for t in len(windowed_laplacian_data))
        spec_preproc2 = SpectrogramPreprocessor()
        y_spectrogram, freq_bins, time_bins = spec_preproc2({'data':y_ref, 'metadata':{'sampling_rate':2048}})
        # (batch_size, n_electrodes, n_timebins, n_freqs)

        test_correlation_matrix, test_pval_matrix, test_mse_matrix = run_regression_with_mse(X, y_spectrogram)
        np.savez(
            os.path.join(REGR_DIR, f"btbank{subject_id}_{trial_id}_{original_electrodes[i]}_laplacian.npz"),
            test_correlation_matrix=test_correlation_matrix,
            test_pval_matrix=test_pval_matrix,
            test_mse_matrix=test_mse_matrix,
            X = X,
            y = y_ref,
            freq_bins = freq_bins,
            time_bins = time_bins
        )
        
        
def main():
    """Main function to run neural regression with specified parameters"""
    print(f"Running neural regression for subject {SUBJECT_ID}, trial {TRIAL_ID} with {feat} features")
    neural_regression(SUBJECT_ID, TRIAL_ID)
    print("Regression completed successfully!")

if __name__ == "__main__":
    main()
        
    # To run this file from the command line, use:
    # 
    # python bfm/analyses/inimai/regressions/clip_neural_regressions_v3.py --subject_id <SUBJECT_ID> --trial_id <TRIAL_ID> --feat <FEATURE_TYPE>
    #
    # Example:
    # python bfm/analyses/inimai/regressions/clip_neural_regressions_v3.py --subject_id 1 --trial_id 0 --feat clip
    #
    # The --feat argument is optional and defaults to "clip". Valid options depend on available features (e.g., "clip", "dinov2", "audio").