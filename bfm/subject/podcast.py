import os
import numpy as np
import torch
import mne
from mne_bids import BIDSPath
from subject.subject import Subject
import json

PODCAST_ROOT_DIR = "/om2/data/public/fietelab/ecog-the-podcast"

class PodcastSubject(Subject):
    """
    This class is used to load the neural data for a given podcast subject.
    It follows the same interface as BrainTreebankSubject for compatibility.
    """
    
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32):
        self.subject_id = subject_id
        self.subject_identifier = f'podcast{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype
        
        # Load the raw data and extract information
        self.raw = self._load_raw_data()
        self.electrode_labels = self._get_electrode_labels()
        self.electrode_ids = {e: i for i, e in enumerate(self.electrode_labels)}
        
        # Cache for neural data
        self.neural_data_cache = {}
        self.electrode_data_length = {}
        
    def _load_raw_data(self):
        """Load the raw ECoG data for this subject"""
        file_path = BIDSPath(root="derivatives/ecogprep",
                             subject=f"{self.subject_id:02d}",
                             task="podcast",
                             datatype="ieeg",
                             description="highgamma",
                             suffix="ieeg",
                             extension="fif")
        
        full_path = os.path.join(PODCAST_ROOT_DIR, str(file_path))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Could not find data file: {full_path}")
        
        return mne.io.read_raw_fif(full_path, verbose=False)
    
    def _get_electrode_labels(self):
        """Get electrode labels from the raw data"""
        return self.raw.info.ch_names
    
    def get_n_electrodes(self):
        return len(self.electrode_labels)
    
    def set_electrode_subset(self, electrode_subset):
        self.electrode_labels = electrode_subset
        self.electrode_ids = {e: i for i, e in enumerate(self.electrode_labels)}
    
    def get_electrode_labels(self, session_id=None):
        return self.electrode_labels
    
    def get_electrode_indices(self, session_id=None):
        return np.arange(self.get_n_electrodes())
    
    def get_sampling_rate(self, session_id=None):
        return self.raw.info['sfreq']
    
    def get_electrode_coordinates(self):
        """
        Get the coordinates of the electrodes for this subject
        Returns:
            coordinates: (n_electrodes, 3) tensor of coordinates (L, I, P)
        """
        # Extract coordinates from the raw data
        ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in self.raw.info['chs']}
        coords = np.vstack([ch2loc[ch] for ch in self.electrode_labels])
        coords *= 1000  # Convert to mm to match BrainTreebank format
        
        return torch.tensor(coords, dtype=self.dtype)
    
    def get_electrode_metadata(self, electrode_label):
        """
        Get the metadata for a given electrode.
        """
        # Find the channel info for this electrode
        for ch in self.raw.info['chs']:
            if ch['ch_name'] == electrode_label:
                return {
                    'L': ch['loc'][0] * 1000,  # Convert to mm
                    'I': ch['loc'][1] * 1000,
                    'P': ch['loc'][2] * 1000,
                    'name': ch['ch_name']
                }
        raise ValueError(f"Electrode {electrode_label} not found")
    
    def get_all_electrode_metadata(self):
        """Get metadata for all electrodes"""
        metadata = []
        for label in self.electrode_labels:
            metadata.append(self.get_electrode_metadata(label))
        return metadata
    
    def get_electrode_data(self, electrode_label, trial_id=None, window_from=None, window_to=None):
        """
        Get data for a specific electrode
        Note: Podcast data doesn't have trials, so trial_id is ignored
        """
        if self.cache:
            if 'data' not in self.neural_data_cache:
                self._cache_data()
            
            electrode_id = self.electrode_ids[electrode_label]
            if window_from is None:
                window_from = 0
            if window_to is None:
                window_to = self.electrode_data_length['data']
            
            return self.neural_data_cache['data'][electrode_id, window_from:window_to]
        else:
            # Get data directly from raw
            data = self.raw.get_data(picks=electrode_label)
            if window_from is None:
                window_from = 0
            if window_to is None:
                window_to = data.shape[1]
            
            return torch.tensor(data[0, window_from:window_to], dtype=self.dtype)
    
    def get_all_electrode_data(self, trial_id=None, window_from=None, window_to=None):
        """
        Get data for all electrodes
        Note: Podcast data doesn't have trials, so trial_id is ignored
        """
        if self.cache:
            if 'data' not in self.neural_data_cache:
                self._cache_data()
            
            if window_from is None:
                window_from = 0
            if window_to is None:
                window_to = self.electrode_data_length['data']
            
            return self.neural_data_cache['data'][:, window_from:window_to]
        else:
            # Get data directly from raw
            data = self.raw.get_data()
            if window_from is None:
                window_from = 0
            if window_to is None:
                window_to = data.shape[1]
            
            return torch.tensor(data[:, window_from:window_to], dtype=self.dtype)
    
    def _cache_data(self):
        """Cache all the neural data in memory"""
        if self.cache and 'data' not in self.neural_data_cache:
            data = self.raw.get_data()
            self.neural_data_cache['data'] = torch.tensor(data, dtype=self.dtype)
            self.electrode_data_length['data'] = data.shape[1]
    
    def clear_neural_data_cache(self, trial_id=None):
        """Clear the neural data cache"""
        self.neural_data_cache = {}
        self.electrode_data_length = {}
    
    def load_neural_data(self, trial_id=None, cache_window_from=None, cache_window_to=None):
        """Load neural data (for compatibility with BrainTreebankSubject)"""
        if self.cache:
            self._cache_data()
        else:
            # When cache=False, we still need to set the data length for compatibility
            if 'data' not in self.electrode_data_length:
                data = self.raw.get_data()
                self.electrode_data_length['data'] = data.shape[1]
        
        # Make sure electrode_data_length has the trial_id key for compatibility
        if trial_id is not None and trial_id not in self.electrode_data_length:
            # For podcast data, all trial_ids map to the same data
            if 'data' in self.electrode_data_length:
                self.electrode_data_length[trial_id] = self.electrode_data_length['data']