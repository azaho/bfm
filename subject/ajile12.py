import os
import numpy as np
import torch
from pynwb import NWBHDF5IO
import warnings
from subject.subject_interface import SubjectInterface

# Suppress specific HDMF namespace warnings
warnings.filterwarnings("ignore", message="Ignoring cached namespace .* because version .* is already loaded")

AJILE_ROOT_DIR = "/om2/user/hmor/ajile12/000055"
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable file locking for HDF5 files. This is helpful for parallel processing.

class AjileSubject(SubjectInterface):
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32, replace_nan_with=0):
        self.subject_id = subject_id
        self.subject_identifier = f'ajile{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable
        self.replace_nan_with = replace_nan_with # TODO: Figure out why the dataset contains NaN in the first place. Conjecture: this is when the epoch is Blocked, and it is NaN even though the paper said 0.

        self.channel_metadata = self._load_channel_metadata()
        self.electrode_labels = self._generate_electrode_labels()
        self.nwb_electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.neural_data_cache = {} # structure: {trial_id: torch.Tensor of shape (n_electrodes, n_samples)}
        self.nwb_files = {} # structure: {trial_id: h5py.File}

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
        return 512 # in Hz
    
    def _get_subject_session_path(self, session_id=None):
        if session_id is None: session_id = 3 # all subjects have session 3, so it can be used as a default to extract metadata
        return os.path.join(AJILE_ROOT_DIR, f'sub-{self.subject_id:02d}', f'sub-{self.subject_id:02d}_ses-{session_id}_behavior+ecephys.nwb')

    def _load_channel_metadata(self):
        """Load channel metadata for this electrode's subject from depth-wm.csv"""
        nwb_file_path = self._get_subject_session_path()
        io = NWBHDF5IO(nwb_file_path, 'r', load_namespaces=True)
        nwb = io.read()
        df = nwb.electrodes[:]
        io.close()
        return df
    
    def _generate_electrode_labels(self):
        """Generate electrode labels for this electrode's subject from channel metadata"""
        n_channels = len(self.channel_metadata)
        electrode_labels = []
        for channel_i in range(n_channels):
            channel_metadata = self.channel_metadata.iloc[channel_i]
            channel_name = str(channel_i+1) + channel_metadata['group_name']
            electrode_labels.append(channel_name)
        self.channel_metadata['label'] = electrode_labels
        return electrode_labels
    
    def cache_neural_data(self, trial_id):
        assert self.cache, "Cache is not enabled; not able to cache neural data."
        if trial_id in self.neural_data_cache: return  # no need to cache again

        nwb_file_path = self._get_subject_session_path(trial_id)
        io = NWBHDF5IO(nwb_file_path, 'r', load_namespaces=True)
        nwb = io.read()
        self.electrode_data_length[trial_id] = nwb.acquisition['ElectricalSeries'].data.shape[0]
        electrode_ids = [self.nwb_electrode_ids[label] for label in self.electrode_labels]
        neural_data = nwb.acquisition['ElectricalSeries'].data[:, electrode_ids].T
        self.neural_data_cache[trial_id] = torch.from_numpy(neural_data).to(self.dtype)
        self.neural_data_cache[trial_id][torch.isnan(self.neural_data_cache[trial_id])] = self.replace_nan_with
        io.close()

    def clear_neural_data_cache(self, trial_id=None):
        if trial_id is None:
            self.neural_data_cache = {}
            self.nwb_files = {}
        else:
            if trial_id in self.neural_data_cache: del self.neural_data_cache[trial_id]
            if trial_id in self.nwb_files: 
                self.nwb_files[trial_id].close()
                del self.nwb_files[trial_id]
    def open_neural_data_file(self, trial_id, ignore_assert=False):
        assert ignore_assert or not self.cache, "Cache is enabled; Use cache_neural_data() instead."
        if trial_id in self.nwb_files: return
        neural_data_file = self._get_subject_session_path(trial_id)
        io = NWBHDF5IO(neural_data_file, 'r', load_namespaces=True)
        nwb = io.read()
        self.nwb_files[trial_id] = nwb
        self.electrode_data_length[trial_id] = nwb.acquisition['ElectricalSeries'].data.shape[0]
    def load_neural_data(self, trial_id):
        if self.cache: self.cache_neural_data(trial_id)
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
            row = self.channel_metadata[self.channel_metadata['label'] == label]
            coordinates[i] = torch.tensor([row['x'], row['y'], row['z']], dtype=self.dtype)
        return coordinates

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache:
            if trial_id not in self.neural_data_cache: self.cache_neural_data(trial_id)
            electrode_id = self.electrode_ids[electrode_label]
            data = self.neural_data_cache[trial_id][electrode_id][window_from:window_to]
            return data
        else:
            if trial_id not in self.nwb_files: self.open_neural_data_file(trial_id)
            neural_data_key = self.nwb_electrode_ids[electrode_label]
            data = torch.from_numpy(self.nwb_files[trial_id].acquisition['ElectricalSeries'].data[window_from:window_to, neural_data_key]).to(self.dtype).T
            data[torch.isnan(data)] = self.replace_nan_with
            return data

    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache: 
            return self.neural_data_cache[trial_id][:, window_from:window_to]
        else:
            if trial_id not in self.nwb_files: self.open_neural_data_file(trial_id)
            electrode_ids = [self.nwb_electrode_ids[label] for label in self.electrode_labels]
            all_electrode_data = torch.from_numpy(self.nwb_files[trial_id].acquisition['ElectricalSeries'].data[window_from:window_to, electrode_ids]).to(self.dtype).T
            all_electrode_data[torch.isnan(all_electrode_data)] = self.replace_nan_with
            return all_electrode_data
    
    def get_electrode_data_length(self, trial_id):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        return self.electrode_data_length[trial_id]

    def get_reach_events(self, trial_id):
        if trial_id not in self.nwb_files: self.open_neural_data_file(trial_id, ignore_assert=True)
        return self.nwb_files[trial_id].intervals['reaches']
    
    def get_epochs(self, trial_id):
        if trial_id not in self.nwb_files: self.open_neural_data_file(trial_id, ignore_assert=True)
        return self.nwb_files[trial_id].intervals['epochs']

if __name__ == "__main__":
    subject = AjileSubject(1, cache=True, dtype=torch.bfloat16)
    subject.load_neural_data(4)
    data = subject.get_all_electrode_data(4, 0, 100)
    
    print(f"Data shape: {data.shape}")