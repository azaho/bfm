import h5py
import os
import json
import pandas as pd
import numpy as np
import torch
from subject.subject import Subject

BRAINTREEBANK_ROOT_DIR = "/cluster/scratch/bradya/braintreebank" # Root directory for the braintreebank data
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable file locking for HDF5 files. This is helpful for parallel processing.

class BrainTreebankSubject(Subject):
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32):
        self.subject_id = subject_id
        self.subject_identifier = f'btbank{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable

        self.localization_data = self._load_localization_data()
        self.electrode_labels = self._get_all_electrode_names()

        self.h5_neural_data_keys = {e:"electrode_"+str(i) for i, e in enumerate(self.electrode_labels)} # only used for accessing the neural data in h5 files
        self.electrode_labels = self._filter_electrode_labels()
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.neural_data_cache = {} # structure: {trial_id: torch.Tensor of shape (n_electrodes, n_samples)}
        self.h5_files = {} # structure: {trial_id: h5py.File}

        self.neural_data_cache_window_from = 0
        self.neural_data_cache_window_to = None

    def get_n_electrodes(self):
        return len(self.electrode_labels)
    
    def set_electrode_subset(self, electrode_subset):
        self.electrode_labels = electrode_subset
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

    def get_electrode_labels(self, session_id=None):
        return self.electrode_labels
    def get_electrode_indices(self, session_id=None):
        return np.arange(self.get_n_electrodes())
    def get_sampling_rate(self, session_id=None):
        return 2048 # in Hz

    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = os.path.join(BRAINTREEBANK_ROOT_DIR, f'localization/sub_{self.subject_id}/depth-wm.csv')
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self._clean_electrode_label)
        return df
    def _get_all_electrode_names(self):
        electrode_labels_file = os.path.join(BRAINTREEBANK_ROOT_DIR, f'electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        return electrode_labels
    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')
    def _get_corrupted_electrodes(self):
        corrupted_electrodes_file = os.path.join(BRAINTREEBANK_ROOT_DIR, "corrupted_elec.json")
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'sub_{self.subject_id}']]
        return corrupted_electrodes
    def _filter_electrode_labels(self):
        """
            Filter the electrode labels to remove corrupted electrodes and electrodes that don't have brain signal
        """
        filtered_electrode_labels = self.electrode_labels
        # Step 1. Remove corrupted electrodes
        if not self.allow_corrupted:
            corrupted_electrodes = self._get_corrupted_electrodes()
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [e for e in self.electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels
    
    def cache_neural_data(self, trial_id, cache_window_from=None, cache_window_to=None, force_cache=False):
        assert self.cache, "Cache is not enabled; not able to cache neural data."
        if trial_id in self.neural_data_cache and not force_cache: return  # no need to cache again
        
        # Open file with context manager to ensure proper closing
        neural_data_file = os.path.join(BRAINTREEBANK_ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        with h5py.File(neural_data_file, 'r', locking=False) as f:
            # Get data length first
            self.electrode_data_length[trial_id] = f['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]

            if cache_window_from is None: cache_window_from = 0
            if cache_window_to is None: cache_window_to = self.electrode_data_length[trial_id]
            
            self.cache_neural_data_window_from = cache_window_from
            self.cache_neural_data_window_to = cache_window_to
            
            # Pre-allocate tensor with specific dtype
            self.neural_data_cache[trial_id] = torch.zeros((len(self.electrode_labels), 
                                                        cache_window_to-cache_window_from), 
                                                        dtype=self.dtype)
            # Load data
            for electrode_label, electrode_id in self.electrode_ids.items():
                neural_data_key = self.h5_neural_data_keys[electrode_label]
                self.neural_data_cache[trial_id][electrode_id] = torch.from_numpy(f['data'][neural_data_key][cache_window_from:cache_window_to]).to(self.dtype)
    def clear_neural_data_cache(self, trial_id=None):
        if trial_id is None:
            self.neural_data_cache = {}
            self.h5_files = {}
        else:
            if trial_id in self.neural_data_cache: del self.neural_data_cache[trial_id]
            if trial_id in self.h5_files: del self.h5_files[trial_id]
        self.electrode_data_length = {}
        self.neural_data_cache_window_from = 0
        self.neural_data_cache_window_to = None
    def open_neural_data_file(self, trial_id):
        assert not self.cache, "Cache is enabled; Use cache_neural_data() instead."
        if trial_id in self.h5_files: return
        neural_data_file = os.path.join(BRAINTREEBANK_ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        self.h5_files[trial_id] = h5py.File(neural_data_file, 'r', locking=False)
        self.electrode_data_length[trial_id] = self.h5_files[trial_id]['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]
    def load_neural_data(self, trial_id, cache_window_from=None, cache_window_to=None):
        if self.cache: self.cache_neural_data(trial_id, cache_window_from=cache_window_from, cache_window_to=cache_window_to)
        else: self.open_neural_data_file(trial_id)
    
    def get_electrode_coordinates(self):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) tensor of coordinates (L, I, P) without any preprocessing of the coordinates
                All coordinates are in between 50mm and 200mm for this dataset (check braintreebank_utils.ipynb for statistics)
        """
        # Create tensor of coordinates in same order as electrode_labels
        coordinates = torch.zeros((len(self.electrode_labels), 3), dtype=self.dtype)
        for i, label in enumerate(self.electrode_labels):
            row = self.get_electrode_metadata(label)
            coordinates[i] = torch.tensor([row['L'], row['I'], row['P']], dtype=self.dtype)
        return coordinates
    def get_electrode_metadata(self, electrode_label):
        """
            Get the metadata for a given electrode.
        """
        return self.localization_data[self.localization_data['Electrode'] == electrode_label].iloc[0]
    def get_all_electrode_metadata(self):
        filtered_df = self.localization_data[self.localization_data['Electrode'].isin(self.electrode_labels)]
        ordered_df = pd.DataFrame([filtered_df[filtered_df['Electrode'] == label].iloc[0] for label in self.electrode_labels])
        return ordered_df.reset_index(drop=True)

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache:
            if trial_id not in self.neural_data_cache: self.cache_neural_data(trial_id)

            # in case we cached a subset of the data, we need to force a re-cache if we want to access data outside of that range
            if ((self.cache_neural_data_window_from is not None) and (window_from < self.cache_neural_data_window_from)) or \
                ((self.cache_neural_data_window_to is not None) and (window_to > self.cache_neural_data_window_to)):
                self.cache_neural_data(trial_id, force_cache=True)

            electrode_id = self.electrode_ids[electrode_label]
            data = self.neural_data_cache[trial_id][electrode_id][window_from-self.cache_neural_data_window_from:window_to-self.cache_neural_data_window_from]
            return data
        else:
            if trial_id not in self.h5_files: self.open_neural_data_file(trial_id)
            neural_data_key = self.h5_neural_data_keys[electrode_label]
            data = torch.from_numpy(self.h5_files[trial_id]['data'][neural_data_key][window_from:window_to]).to(self.dtype)
            return data

    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache: 
            # in case we cached a subset of the data, we need to force a re-cache if we want to access data outside of that range
            if ((self.cache_neural_data_window_from is not None) and (window_from < self.cache_neural_data_window_from)) or \
                ((self.cache_neural_data_window_to is not None) and (window_to > self.cache_neural_data_window_to)):
                self.cache_neural_data(trial_id, force_cache=True)

            return self.neural_data_cache[trial_id][:, window_from-self.cache_neural_data_window_from:window_to-self.cache_neural_data_window_from]
        else:
            all_electrode_data = torch.zeros((len(self.electrode_labels), window_to-window_from), dtype=self.dtype)
            for electrode_label, electrode_id in self.electrode_ids.items():
                all_electrode_data[electrode_id] = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)
            return all_electrode_data