import os
import json
import numpy as np
import torch
import mne
import h5py

MGH_ROOT_DIR = "/om2/user/zaho/mgh_2024_11_04/Infolab-SEEG-Data-20241022/NatusXLTEK" # Root directory for the data

class MGHSubject:
    """ 
        This class is used to load the neural data for a given subject and session.
        It also contains methods to get the data for a given electrode and session.
    """
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32, use_h5_file=False):
        self.subject_id = subject_id
        self.subject_identifier = f'mgh{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable

        # Load patient map to get session filenames
        with open(os.path.join(MGH_ROOT_DIR, '../patient_map.json'), 'r') as f:
            self.patient_info = json.load(f)[self.subject_id-1]
        self.sessions = self.patient_info['sessions']

        with open(os.path.join(MGH_ROOT_DIR, "../name_map.json"), "r") as f:
            self.name_map = json.load(f)

        self.localization_data = self._load_localization_data()
        self.electrode_labels = self._get_all_electrode_names()
        self.electrode_labels = self._filter_electrode_labels(self.electrode_labels)
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.electrode_index_subset = {}
        self.electrode_labels_subset = {}
        self.session_annotations = {}
        
        self.use_h5_file = use_h5_file
        self.h5_files = {} # structure: {session_filename: h5py.File}
        self.edf_files = {} # structure: {session_filename: mne.io.Raw}
        self.neural_data_cache = {} # structure: {session_filename: torch.Tensor of shape (n_electrodes, n_samples)}

        # Load electrode labels for each session
        for session_id in range(len(self.sessions)):
            session_electrode_labels = self.sessions[session_id]['channel_names']
            session_electrode_labels = self._filter_electrode_labels(session_electrode_labels)
            self.electrode_index_subset[session_id] = np.array([self.electrode_ids[e] for e in session_electrode_labels])
            self.electrode_labels_subset[session_id] = session_electrode_labels

        # For downstream Laplacian rereferencing
        self.laplacian_electrodes, self.electrode_neighbors = self._get_all_laplacian_electrodes()

    def get_n_electrodes(self, session_id=None):
        if session_id is None: return len(self.electrode_labels)
        else: return len(self.electrode_index_subset[session_id])

    def get_electrode_indices(self, session_id=None):
        if session_id is None: return np.arange(self.get_n_electrodes())
        else: return self.electrode_index_subset[session_id]
    def get_electrode_labels(self, session_id=None):
        if session_id is None: return self.electrode_labels
        else: return self.electrode_labels_subset[session_id]

    def get_sampling_rate(self, session_id):
        return self.sessions[session_id]['sampling_rate']

    def _load_localization_data(self):
        """Load localization data for this electrode's subject"""
        return None # TODO: Implement this

    def _get_all_electrode_names(self):
        """Get electrode names from all sessions"""
        return self.patient_info['all_channel_names']

    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')

    def _get_corrupted_electrodes(self):
        corrupted_electrodes_file = os.path.join(MGH_ROOT_DIR, "../corrupted_elec.json")
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[self.subject_identifier]]
        return corrupted_electrodes

    def _get_all_laplacian_electrodes(self, verbose=False):
        """Get all laplacian electrodes for a given subject"""
        return [], {} # TODO: Implement this

    def _filter_electrode_labels(self, electrode_labels):
        """Filter out corrupted and non-neural electrodes"""
        filtered_electrode_labels = electrode_labels
        if not self.allow_corrupted:
            corrupted_electrodes = self._get_corrupted_electrodes()
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Remove non-neural channels
        non_neural_channels = ["X", "DC", "TRIG"]
        non_neural_channels += ["EMG", "LEMG", "REMG"]
        non_neural_channels += ["EKG", "ECG"]
        non_neural_channels += ["LOC", "ROC"]
        non_neural_channels += ["OSAT", "PR", "PLETH", "SP"]
        non_neural_channels += ["CI", "C."] # not sure what those are, names like "CII" and "C. II"
    
        filtered_electrode_labels = [e for e in filtered_electrode_labels if not any(e.upper().startswith(x) for x in non_neural_channels)]
        return filtered_electrode_labels

    def cache_neural_data(self, session_id):
        assert self.cache, "Cache is not enabled"
        if session_id in self.neural_data_cache: return
        
        if self.use_h5_file:
            h5_path = os.path.join(MGH_ROOT_DIR, 'h5', self.sessions[session_id]['filename'] + '.h5')
            original_h5_electrode_labels = self.sessions[session_id]['channel_names']
            original_h5_electrode_ids = [original_h5_electrode_labels.index(e) for e in self.get_electrode_labels(session_id)]
            original_h5_electrode_keys = ["channel_"+str(i) for i in original_h5_electrode_ids]

            with h5py.File(h5_path, 'r', locking=False) as f:
                self.electrode_data_length[session_id] = f['data'][original_h5_electrode_keys[0]].shape[0]

                self.neural_data_cache[session_id] = torch.zeros((len(original_h5_electrode_keys), self.electrode_data_length[session_id]), dtype=self.dtype)
                for i, key in enumerate(original_h5_electrode_keys):
                    self.neural_data_cache[session_id][i] = torch.from_numpy(f['data'][key][:]).to(self.dtype)
        else: 
            edf_path = os.path.join(MGH_ROOT_DIR, self.sessions[session_id]['filename'] + '.edf')
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            data = raw.get_data(picks=self.get_electrode_labels(session_id))
            self.neural_data_cache[session_id] = torch.from_numpy(data).to(self.dtype)
            self.electrode_data_length[session_id] = data.shape[1]

    def clear_neural_data_cache(self, session_id=None):
        if session_id is None:
            self.neural_data_cache = {}
            self.edf_files = {}
            self.h5_files = {}
            self.session_annotations = {}
            self.electrode_data_length = {}
        else:
            if session_id in self.neural_data_cache: del self.neural_data_cache[session_id]
            if session_id in self.edf_files: del self.edf_files[session_id]
            if session_id in self.h5_files: del self.h5_files[session_id]
            if session_id in self.session_annotations: del self.session_annotations[session_id]
            if session_id in self.electrode_data_length: del self.electrode_data_length[session_id]

    def open_neural_data_file(self, session_id):
        assert not self.cache, "Cache is enabled; Use cache_neural_data() instead"
        if session_id in self.edf_files: return
        
        if self.use_h5_file:
            h5_path = os.path.join(MGH_ROOT_DIR, 'h5', self.sessions[session_id]['filename'] + '.h5')
            self.h5_files[session_id] = h5py.File(h5_path, 'r', locking=False)
            self.electrode_data_length[session_id] = self.h5_files[session_id]['data']['channel_0'].shape[0]
        else: 
            edf_path = os.path.join(MGH_ROOT_DIR, self.sessions[session_id]['filename'] + '.edf')
            self.edf_files[session_id] = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            self.electrode_data_length[session_id] = len(self.edf_files[session_id].times)
    
    def load_annotations(self, session_id):
        annotation_file = os.path.join(MGH_ROOT_DIR, "../annotations", self.name_map[self.sessions[session_id]['filename']] + '.json')
        with open(annotation_file, "r") as f:
            self.session_annotations[session_id] = json.load(f)

    def load_neural_data(self, session_id):
        self.load_annotations(session_id)
        if self.cache: self.cache_neural_data(session_id)
        else: self.open_neural_data_file(session_id)

    def get_annotations(self, session_id, window_from=None, window_to=None):
        if session_id not in self.session_annotations: self.load_annotations(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id] / self.get_sampling_rate(session_id)
        
        annotation_onsets = np.array([a['onset'] for a in self.session_annotations[session_id]])
        annotation_descriptions = np.array([a['description'] for a in self.session_annotations[session_id]])
        mask = (annotation_onsets >= window_from) & (annotation_onsets <= window_to)
        return annotation_onsets[mask], annotation_descriptions[mask]
    

    def get_electrode_data(self, electrode_label, session_id, window_from=None, window_to=None):
        if session_id not in self.electrode_data_length: self.load_neural_data(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id]

        if self.cache:
            electrode_id = self.electrode_ids[electrode_label]
            return self.neural_data_cache[session_id][electrode_id][window_from:window_to]
        else:
            if self.use_h5_file:
                original_h5_electrode_labels = self.sessions[session_id]['channel_names']
                electrode_id = original_h5_electrode_labels.index(electrode_label)
                h5_electrode_key = "channel_" + str(electrode_id)
                return torch.from_numpy(self.h5_files[session_id]['data'][h5_electrode_key][window_from:window_to]).to(self.dtype)
            else:
                data = self.edf_files[session_id].get_data(picks=[electrode_label], start=window_from, stop=window_to)
                return torch.from_numpy(data[0]).to(self.dtype)

    def get_all_electrode_data(self, session_id, window_from=None, window_to=None):
        if session_id not in self.electrode_data_length: self.load_neural_data(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id]

        if self.cache:
            return self.neural_data_cache[session_id][:, window_from:window_to]
        else:
            if self.use_h5_file:
                h5_path = os.path.join(MGH_ROOT_DIR, 'h5', self.sessions[session_id]['filename'] + '.h5')
                original_h5_electrode_labels = self.sessions[session_id]['channel_names']
                original_h5_electrode_ids = [original_h5_electrode_labels.index(e) for e in self.get_electrode_labels(session_id)]
                original_h5_electrode_keys = ["channel_"+str(i) for i in original_h5_electrode_ids]

                with h5py.File(h5_path, 'r', locking=False) as f:
                    data = torch.zeros((len(original_h5_electrode_keys), window_to-window_from), dtype=self.dtype)
                    for i, key in enumerate(original_h5_electrode_keys):
                        data[i] = torch.from_numpy(f['data'][key][window_from:window_to]).to(self.dtype)
                return data
            else: 
                data = self.edf_files[session_id].get_data(picks=self.get_electrode_labels(session_id), start=window_from, stop=window_to)
                return torch.from_numpy(data).to(self.dtype)

Subject = MGHSubject # alias for convenience

if __name__ == "__main__":
    subject = MGHSubject(5)
    print(subject.electrode_labels)
    print(subject.get_n_electrodes())
    print(subject.get_all_electrode_data(0).shape)