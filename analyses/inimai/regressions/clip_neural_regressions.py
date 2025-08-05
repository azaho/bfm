import numpy as np
import matplotlib.pyplot as plt
import os
from bfm.evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE
from bfm.subject.braintreebank import BrainTreebankSubject
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import torch
from bfm.model.BFModule import BFModule
import torch.nn as nn
from bfm.training_setup.training_config import get_default_config
from sklearn.preprocessing import StandardScaler

SUBJECT_TRIAL_TO_MOVIE = {
    (1,0): 'fantastic-mr-fox.mp4',
    (1,1): 'the-martian.mp4',
    (1,2): 'thor-ragnarok.mp4',
    (2,0): 'venom.mp4',
    (2,1): 'spider-man-3-homecoming.mp4',
    (2,2): 'guardians-of-the-galaxy.mp4',
    (2,3): 'guardians-of-the-galaxy-2.mp4',
    (2,4): 'avengers-infinity-war.mp4',
    (2,5): 'black-panther.mp4',
    (2,6): 'aquaman.mp4',
    (3,0): 'cars-2.mp4',
    (3,1): 'lotr-1.mp4',
    (3,2): 'lotr-2.mp4',
    (4,0): 'shrek-the-third.mp4',
    (4,1): 'megamind.mp4',
    (4,2): 'incredibles.mp4',
    (5,0): 'fantastic-mr-fox.mp4',
    (6,0): 'megamind.mp4',
    (6,1): 'toy-story.mp4',
    (6,4): 'coraline.mp4',
    (7,0): 'cars-2.mp4',
    (7,1): 'megamind.mp4',
    (8,0): 'sesame-street-episode-3990.mp4',
    (9,0): 'ant-man.mp4',
    (10,0): 'cars-2.mp4',
    (10,1): 'spider-man-far-from-home.mp4',
}

CLIP_DIR = "/om2/data/public/braintreebank_movies_clip_preprocessed_2/"
MOVIES_DIR = "/om2/data/public/braintreebank_movies/"
REGR_DIR = "/om2/data/public/braintreebank_movies_clip_preprocessed_2/regr_results_2/"


class SpectrogramPreprocessor(BFModule):
    def __init__(self, spectrogram_parameters=None, output_dim=-1):
        """
        spectrogram_parameters is a dictionary with the following keys:
        spectrogram_parameters = {
            'max_frequency': int, 'Maximum frequency for spectrogram'
            'tperseg': float, 'Time of each spectrogram segment in seconds'
            'poverlap': float, 'Proportion of overlap between segments for spectrogram'
            'window': str, 'Window function for spectrogram', # Allowed values: 'hann', 'boxcar'
        }
        """
        if spectrogram_parameters is None: # Load default spectrogram parameters from training config
            spectrogram_parameters = get_default_config(random_string="TEMP", wandb_project="")['model']['signal_preprocessing']['spectrogram_parameters']

        super(SpectrogramPreprocessor, self).__init__()
        self.output_dim = output_dim
        self.spectrogram_parameters = spectrogram_parameters
        
        # from https://docs.pytorch.org/docs/stable/generated/torch.fft.rfftfreq.html
        # if n is nperseg, and d is 1/sampling_rate, then f = torch.arange((n + 1) // 2) / (d * n)
        # note: nperseg is always going to be even, so it simplifies to torch.arange(n/2) / n * sampling_rate
        # note: n = sampling_rate * tperseg, so it simplifies to torch.arange(sampling_rate * tperseg / 2) / tperseg
        #    which is a list that goes from 0 to sampling_rate / 2 in increments of sampling_rate / nperseg = 1 / tperseg
        # so max frequency bin is max_frequency * tperseg + 1 (adding one to make the endpoint inclusive)
        self.max_frequency_bin = round(self.spectrogram_parameters['max_frequency'] * self.spectrogram_parameters['tperseg'] + 1)

        # Transform FFT output to match expected output dimension
        self.output_transform = nn.Identity() if self.output_dim == -1 else nn.Linear(self.max_frequency_bin, self.output_dim)
    
    def forward(self, batch):
        # batch['data'] is of shape (batch_size, n_electrodes, n_samples)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        batch_size, n_electrodes = batch['data'].shape[:2]
        
        # Reshape for STFT
        x = batch['data'].reshape(batch_size * n_electrodes, -1)
        x = x.to(dtype=torch.float32)  # Convert to float32 for STFT
        
        # STFT parameters
        sampling_rate = batch['metadata']['sampling_rate']
        nperseg = round(self.spectrogram_parameters['tperseg'] * sampling_rate)
        noverlap = round(self.spectrogram_parameters['poverlap'] * nperseg)
        hop_length = nperseg - noverlap
        
        window = {
            'hann': torch.hann_window,
            'boxcar': torch.ones,
        }[self.spectrogram_parameters['window']](nperseg, device=x.device)
        
        # Compute STFT
        x = torch.stft(x,
                      n_fft=nperseg, 
                      hop_length=hop_length,
                      win_length=nperseg,
                      window=window,
                      return_complex=True,
                      normalized=False,
                      center=True)
        
        # Take magnitude
        x = torch.abs(x)

        # Trim to max frequency (using a pre-calculated max frequency bin)
        x = x[:, :self.max_frequency_bin, :]
            
        # Reshape back
        _, n_freqs, n_times = x.shape
        x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
        x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)
        
        # Z-score normalization
        x = x - x.mean(dim=[0, 2], keepdim=True)
        x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

        freq_bins = torch.fft.rfftfreq(nperseg, d=1/sampling_rate)
        freq_bins = freq_bins[:self.max_frequency_bin]

        n_timebins = x.shape[2]
        time_bins = torch.arange(n_timebins, device=x.device) * hop_length / sampling_rate


        return x.to(dtype=batch['data'].dtype), freq_bins, time_bins


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

def get_neural_data_at_index(subject_id, trial_id, electrode, start, end):
    subject = BrainTreebankSubject(subject_id, cache=False)
    neural_data = subject.get_electrode_data(electrode,trial_id,window_from=start, window_to=end)
    return neural_data

def get_movie_data(movie):
    clip_features_path = os.path.join(CLIP_DIR, movie.replace('.mp4', '_clip_features.npy'))
    timestamps_path = os.path.join(CLIP_DIR, movie.replace('.mp4', '_timestamps.npy'))
    movie_path = os.path.join(MOVIES_DIR, movie)

    clip_features = np.load(clip_features_path)  # shape: (num_samples, feature_dim)
    timestamps = np.load(timestamps_path)
    return clip_features, timestamps, movie_path

def get_subject_trial_from_movie(movie):
    subject_trial_list = []
    for subject_trial, movie_name in SUBJECT_TRIAL_TO_MOVIE.items():
        if movie_name == movie:
            subject_trial_list.append(subject_trial)
    return subject_trial_list

def process_subject_trial(subject_id, trial_id, timestamps, sampling_interval=10.0):
    subject = BrainTreebankSubject(subject_id, cache=False)
    electrode_locations = {electrode_label: subject.get_electrode_metadata(electrode_label)['DesikanKilliany'] for electrode_label in subject.get_electrode_labels()}
    unique_electrodes = list(electrode_locations.keys())

    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{subject_id}_trial{trial_id:03}_timings.csv')
    trigs_df = pd.read_csv(trigger_times_file)
    last_trigger_time = trigs_df['movie_time'].iloc[-1]
    safe_end_timestamp = min(timestamps[-2], last_trigger_time - 1.0)  # 1 second buffer

    sampled_times = np.arange(0, safe_end_timestamp, sampling_interval)
    timestamp_indices = np.searchsorted(timestamps, sampled_times, side="left")
    return sampled_times, timestamp_indices, unique_electrodes, subject

def get_base_data(subject_id, trial_id, electrode_label, sampled_times, subject):
    window_ms = 250
    windowed_neural_data = []

    for t in sampled_times:
        window_start = t - window_ms / 1000
        idx_start = obtain_neural_data_index(subject_id, trial_id, np.array([window_start])).item()
        # idx_end = obtain_neural_data_index(subject_id, trial_id, np.array([window_end])).item() + 1  # +1 for inclusive window
        idx_end = int(idx_start + 0.75 * 2048)

        data = subject.get_electrode_data(electrode_label, trial_id, window_from=idx_start, window_to=idx_end)
        windowed_neural_data.append(data.cpu().numpy() if hasattr(data, "cpu") else data)
    return windowed_neural_data


def run_regression(X, y_spectrogram, freq_bins, time_bins, subject_id, trial_id, electrode_label):
    n_timebins = y_spectrogram.shape[2]
    n_freqs = y_spectrogram.shape[3]

    n_samples = X.shape[0]
    n_folds = 3

    fold_size = n_samples // n_folds
    fold_indices = [i * fold_size for i in range(n_folds)] + [n_samples]

    # Store correlation and pval matrices for each fold
    fold_correlation_matrices = []
    fold_pval_matrices = []

    for fold in range(n_folds):
        fold_corr_matrix = np.zeros((n_freqs, n_timebins))
        fold_pval_matrix = np.zeros((n_freqs, n_timebins))
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
                    # Not enough variance to compute correlation
                    fold_corr_matrix[f, t] = np.nan
                    fold_pval_matrix[f, t] = np.nan
                    continue
                reg = Ridge(alpha=0.1)
                reg.fit(X_train_scaled, y_train)
                y_pred_test = reg.predict(X_test_scaled)
                corr, pval = pearsonr(y_test, y_pred_test)
                fold_corr_matrix[f, t] = corr
                fold_pval_matrix[f, t] = pval

        fold_correlation_matrices.append(fold_corr_matrix)
        fold_pval_matrices.append(fold_pval_matrix)

    # Find the fold with the highest sum of absolute correlation values
    fold_sums = [np.nansum(np.abs(mat)) for mat in fold_correlation_matrices]
    best_fold_idx = np.argmax(fold_sums)

    test_correlation_matrix = fold_correlation_matrices[best_fold_idx]
    test_pval_matrix = fold_pval_matrices[best_fold_idx]

    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        test_correlation_matrix,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[time_bins[0],time_bins[-1],freq_bins[0],freq_bins[-1]]
    )

    star_y, star_x = np.where(test_pval_matrix < 0.05)

    time_edges = np.linspace(
        time_bins[0].item() if hasattr(time_bins[0], 'item') else time_bins[0],
        time_bins[-1].item() if hasattr(time_bins[-1], 'item') else time_bins[-1],
        test_correlation_matrix.shape[1] + 1
    )
    freq_edges = np.linspace(
        freq_bins[0].item() if hasattr(freq_bins[0], 'item') else freq_bins[0],
        freq_bins[-1].item() if hasattr(freq_bins[-1], 'item') else freq_bins[-1],
        test_correlation_matrix.shape[0] + 1
    )
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2
    freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2 - 0.3

    for y, x in zip(star_y, star_x):
        plt.text(
            time_centers[x], freq_centers[y], '*',
            color='white', fontsize=14, ha='center', va='center', fontweight='bold'
        )

    event_time = 0.25
    event_line = plt.axvline(event_time, color='black', linestyle='--', linewidth=1, label='Event (t=0.25)')
    asterisk_handle = Line2D([0], [0], marker='*', color='w', linestyle='None', markersize=14, markerfacecolor='w', label='p < 0.05')

    plt.colorbar(im, label='Test Correlation')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'btbank{subject_id}_{trial_id}_{electrode_label}: Test Correlation (Ridge Regression) between Raw Clip Features and Neural Response')
    plt.legend(handles=[event_line, asterisk_handle], loc='upper left', frameon=True)
    plt.tight_layout()

    save_dir = os.path.join(REGR_DIR, f"btbank{subject_id}", f"trial_{trial_id}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{electrode_label}_test_correlation.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    np.save(os.path.join(save_dir, f"{electrode_label}_test_correlation.npy"), test_correlation_matrix)
    np.save(os.path.join(save_dir, f"{electrode_label}_test_pval.npy"), test_pval_matrix)


def process_movie(movie):
    clip_features, timestamps, movie_path = get_movie_data(movie)
    subject_trial_list = get_subject_trial_from_movie(movie)

    for subject_trial in tqdm(subject_trial_list, desc='Processing subject trials'):
        subject_id, trial_id = subject_trial
        sampled_times, timestamp_indices, unique_electrodes, subject = process_subject_trial(subject_id, trial_id, timestamps)

        for electrode_label in unique_electrodes:
            windowed_neural_data = get_base_data(subject_id, trial_id, electrode_label, sampled_times, subject)
            y = np.stack(windowed_neural_data)
            X = clip_features[timestamp_indices]

            y_copy = y.copy()
            y_tensor = torch.tensor(y_copy)
            y_tensor = y_tensor.reshape(y_copy.shape[0], 1, y_copy.shape[1])

            nperseg = 400
            noverlap = 350
            tperseg = nperseg / 2048
            poverlap = noverlap / nperseg
            spec_preproc = SpectrogramPreprocessor({'max_frequency': 200, 'window': 'hann', 'tperseg': tperseg, 'poverlap': poverlap})
            y_spectrogram, freq_bins, time_bins = spec_preproc({'data': y_tensor, 'metadata': {'sampling_rate': 2048}})  # shape: (batch_size, n_electrodes, n_timebins, n_freqs)

            run_regression(X, y_spectrogram, freq_bins, time_bins, subject_id, trial_id, electrode_label)


movies_list = [f for f in os.listdir(MOVIES_DIR) if os.path.isfile(os.path.join(MOVIES_DIR, f)) and f >= 'coraline.mp4']
for movie in tqdm(movies_list, desc='Processing movies'):
    process_movie(movie)