import mne
import numpy as np
import os
import re
import json
import pandas as pd

from .braintreebank import BrainTreebankSubject

def preprocess(raw, l_freq = 1, h_freq = 256):
    # apply notch filter
    raw.notch_filter(freqs = np.arange(60, h_freq+61, 60))
    # apply band pass filter
    raw.filter(l_freq = l_freq, h_freq = h_freq)
    return raw
    
class MNEBraintreebankSubject:
    """
    A subject-specific class to handle BrainTreeBank data and convert it to MNE Raw objects.
    """
    def __init__(self, subject_id: str, allow_corrupted: bool = False, cache: bool = True):
        """
        Initializes the subject, loads non-trial-specific data like channel names and montage.

        Parameters:
        - subject_id: The identifier for the subject (e.g., 'P001S01').
        - allow_corrupted: Whether to include electrodes marked as corrupted.
        - cache: Whether to cache neural data in memory for faster access.
        """
        self.subject = BrainTreebankSubject(
            subject_id=subject_id,
            allow_corrupted=allow_corrupted,
            cache=cache
        )
        
        # Get data that is not trial-specific
        self.ch_names = self.subject.get_electrode_labels()
        self.sfreq = self.subject.get_sampling_rate()
        self.ch_types = ['seeg'] * len(self.ch_names)
        
        # Create MNE Info object that will be shared across trials
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
        
        # Get electrode coordinates and set montage
        coords = self.subject.get_electrode_coordinates().numpy()  # in mm, (L, I, P)
        coords_m = coords / 1000.0  # Convert coordinates from mm to meters

        # Convert from LIP to RAS for MNE (x=Right, y=Anterior, z=Superior)
        # R = -L, A = -P, S = -I
        # Our coordinates are L, I, P. So we map:
        # x (Right) = -L -> index 0
        # y (Anterior) = -P -> index 2
        # z (Superior) = -I -> index 1
        ras_coords = np.c_[-coords_m[:, 0], -coords_m[:, 2], -coords_m[:, 1]]
        self.ch_pos = dict(zip(self.ch_names, ras_coords))
        
        self.montage = mne.channels.make_dig_montage(ch_pos=self.ch_pos, coord_frame='head')
        self.info.set_montage(self.montage)

    def get_trial_raw(self, trial_id: int, t_to = None, t_from = None, ref_channels = 'average', t0 = 0) -> mne.io.RawArray:
        """
        Loads the electrode data for a specific trial and returns it as an MNE RawArray object.
        Re-references data if ref_channels == 'average'

        Parameters:
        - trial_id: The trial number to load.
        - t_to/t_from: Start and end time (in seconds)
        - t0: Start index (e.g., based on triggers)

        Returns:
        - An MNE RawArray object containing the data for the specified trial.
        """
        # Data is returned as (n_electrodes, n_samples)
        window_from = t0 + self.sfreq * t_from
        window_to = t0 + self.sfreq * t_to
        electrode_data = self.subject.get_all_electrode_data(trial_id, 
                                                            window_from = window_from,
                                                            window_to = window_to).numpy()
        
        # Create a new RawArray object for the trial data with the pre-configured info
        raw = mne.io.RawArray(electrode_data, self.info)#, first_samp=0, copy='auto')

        # Re-reference using ref_channels (default average referencing)
        raw.set_eeg_reference(ref_channels)
        
        return raw

    def get_trial_info(self, trial_id: int):
        return self.subject._load_trial_metadata(trial_id)

def extract_movie_metadata(dir_path):
    data = []
    pattern = re.compile(r"sub_(\d+)_trial(\d+)_metadata\.json")

    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if match:
            subject = int(match.group(1))
            trial = int(match.group(2))
            filepath = os.path.join(dir_path, filename)

            with open(filepath, 'r') as f:
                metadata = json.load(f)
                title = metadata.get("title", "UNKNOWN")
                data.append({
                    "subject": subject,
                    "trial": trial,
                    "title": title
                })

    return data

def words_to_events(words_df, event_column='is_onset', onset_column='est_idx'):
    """
    Converts words_df into MNE events array.
    
    Parameters:
    - words_df: pd.DataFrame containing the annotated words
    - event_column: column to use for labeling events (e.g., 'pos', 'text')
    - onset_column: column with estimated sample index (e.g., 'est_idx')

    Returns:
    - events: np.ndarray of shape (n_events, 3)
    - event_id_dict: dict mapping label names to integer IDs
    """
    df = words_df.dropna(subset=[onset_column]).copy()
    df[onset_column] = df[onset_column].astype(int)

    # Create event_id mapping
    unique_labels = sorted(df[event_column].unique())
    event_id_dict = {str(int(label)) if isinstance(label, (np.integer, np.int64)) else str(label): idx + 1 for idx, label in enumerate(unique_labels)}  # ID must be > 0

    # Build the events array
    events = np.array([
        [int(row[onset_column]), 0, event_id_dict[str(row[event_column])]]
        for _, row in df.iterrows()
    ])

    return events, event_id_dict

def get_subject_data(subject_id, trial_id, t_from, t_to, sfreq):
    subject = MNEBraintreebankSubject(subject_id=subject_id, allow_corrupted=False, cache=True)
    meta_dict, words_df, triggers = subject.get_trial_info(trial_id)
    events, events_dict = words_to_events(words_df, event_column = 'is_onset')

    raw = subject.get_trial_raw(trial_id, t_from = t_from, t_to = t_to)
    raw = preprocess(raw)
    raw, events = raw.resample(sfreq=sfreq, events = events)
    raw = raw.filter(l_freq=1.0, h_freq=None)
    epochs = mne.Epochs(raw, events, event_id=events_dict, 
                    tmin = -0.2, tmax = 0.6, baseline = (None, 0), 
                    event_repeated = 'drop')

    X = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    # Compute per-channel stats on the training set
    means = X.mean(axis=(0, 2), keepdims=True)   # shape (1, n_ch, 1)
    stds  = X.std(axis=(0, 2), keepdims=True)

    # Z-score
    X = (X - means) / (stds + 1e-6)
    
    Y = epochs.events[:, -1]  # the event codes as target labels
    return X, Y