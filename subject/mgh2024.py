import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import torch

from subject.subject import Subject


MGH_ROOT_DIR = os.environ["MGH_ROOT_DIR"]  # Root directory for the MGH data

class MGH2024Subject(Subject):
    """ 
        This class is used to load the neural data for a given subject and session.
        It also contains methods to get the data for a given electrode and session.
    """
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32):
        assert subject_id >= 0 and subject_id <= 62, f"Subject ID must be between 0 and 62, got {subject_id}"


        self.subject_id = subject_id
        self.subject_identifier = f'mgh{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable

        # Load patient map to get session filenames
        with open(os.path.join(MGH_ROOT_DIR, 'patient_sessions_map.json'), 'r') as f:
            self.patient_info = json.load(f)[self.subject_id-1]
            assert self.patient_info["patient_identifier"] == self.subject_identifier, f"Patient identifier {self.patient_info['patient_identifier']} does not match subject identifier {self.subject_identifier}. It is probably because the file patient_sessions_map.json is our of order. You can fix this"
        self.sessions = self.patient_info['sessions']

        self.localization_data = self._load_localization_data()
        self.electrode_labels = self._get_all_electrode_names()
        self.electrode_labels = self._filter_electrode_labels(self.electrode_labels, keep_corrupted=True)
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.electrode_index_subset = {}
        self.electrode_labels_subset = {}
        self.session_metadata = {}
        self.session_annotations = {}
        
        self.h5_files = {} # structure: {session_filename: h5py.File}
        self.neural_data_cache = {} # structure: {session_filename: torch.Tensor of shape (n_electrodes, n_samples)}

        # Load electrode labels for each session
        for session_id in range(len(self.sessions)):
            session_hash = self.sessions[session_id]
            session_metadata_filename = os.path.join(MGH_ROOT_DIR, 'json', session_hash + '.json')
            with open(session_metadata_filename, 'r') as f:
                self.session_metadata[session_id] = json.load(f)

            session_electrode_labels = self.session_metadata[session_id]['channel_names']
            session_electrode_labels = self._filter_electrode_labels(session_electrode_labels, session_id=session_id)
            self.electrode_index_subset[session_id] = np.array([self.electrode_ids[e] for e in session_electrode_labels])
            self.electrode_labels_subset[session_id] = session_electrode_labels

            session_length = self.session_metadata[session_id]['session_length'] * self.session_metadata[session_id]['sampling_rate']
            self.electrode_data_length[session_id] = round(session_length)


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
        return self.session_metadata[session_id]['sampling_rate']

    def _load_localization_data(self):
        """Load localization data for this electrode's subject"""
        return None # TODO: Implement this

    def _get_all_electrode_names(self):
        """Get electrode names from all sessions"""
        return self.patient_info['all_channel_names']

    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')

    def _get_corrupted_electrodes(self, session_id):
        corrupted_electrodes_file = os.path.join(MGH_ROOT_DIR, "corrupted_elec.json")
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[self.subject_identifier + "_" + str(session_id)]]
        return corrupted_electrodes

    # Process group and mni_name columns to handle special cases with hyphens
    def _process_channel_name(self, name):
        if '-' not in name:
            return name
        base, suffix = name.split('-', 1)
        # Handle range cases
        if 'one-to-eight' in name:
            num = int(''.join(c for c in suffix if c.isdigit()))
            return base + str(num)
        elif 'nine-to-sixteen' in name:
            num = int(''.join(c for c in suffix if c.isdigit()))
            return base + str(num + 8)
        
        # Handle numeric suffix case
        if any(c.isdigit() for c in suffix):
            num = int(''.join(c for c in suffix if c.isdigit()))
            return base + str(num)
        return name
    def _filter_electrode_labels(self, electrode_labels, session_id=None, keep_corrupted=False):
        """Filter out corrupted and non-neural electrodes"""
        filtered_electrode_labels = electrode_labels
        if not self.allow_corrupted and not keep_corrupted and session_id is not None:
            corrupted_electrodes = self._get_corrupted_electrodes(session_id)
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Remove non-neural channels
        non_neural_channels = ["X", "DC", "TRIG"]
        non_neural_channels += ["EMG", "LEMG", "REMG"]
        non_neural_channels += ["EKG", "ECG"]
        non_neural_channels += ["LOC", "ROC"]
        non_neural_channels += ["OSAT", "PR", "PLETH", "SP"]
        non_neural_channels += ["CI", "C."] # not sure what those are, names like "CII" and "C. II"
        non_neural_channels += ["C"] # Those are the "default" electrode names, and we do not know what they are; for now, just removed
    
        filtered_electrode_labels = [e for e in filtered_electrode_labels if not any(e.upper().startswith(x) for x in non_neural_channels)]

        # replace substrings like "CM" and "ANT" in the electrode labels with ""
        filtered_electrode_labels = [e.replace("CM", "").replace("ANT", "") for e in filtered_electrode_labels]
        # replace the dashed names with clean names
        filtered_electrode_labels = [self._process_channel_name(e) for e in filtered_electrode_labels]

        # remove electrodes that are not of the format [A-Za-z]+\d+ (where \d+ is any number)
        filtered_electrode_labels = [e for e in filtered_electrode_labels if re.match(r'[A-Za-z]+\d+$', e)]

        return filtered_electrode_labels

    def get_electrode_coordinates(self, session_id=None):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) tensor of coordinates (MNI) without any preprocessing of the coordinates
                if coordinates are not available, returns nan for that electrode
        """
        electrode_labels = self.get_electrode_labels(session_id)

        patient_coordinate_map = os.path.join(MGH_ROOT_DIR, 'patient_coordinate_map.csv')
        patient_coordinate_map = pd.read_csv(patient_coordinate_map)

        coordinates = torch.full((len(electrode_labels), 3), float('nan'), dtype=self.dtype)
        for i, label in enumerate(electrode_labels):
            # Find matching row in coordinate map for this electrode and subject
            matching_row = patient_coordinate_map[
                (patient_coordinate_map['patient_id'] == self.subject_identifier) & 
                (patient_coordinate_map['mni_name'] == label)
            ]
            if len(matching_row) > 0:
                # Take first matching row if multiple exist
                row = matching_row.iloc[0]
                coordinates[i] = torch.tensor([
                    row['mni_x'],
                    row['mni_y'], 
                    row['mni_z']
                ], dtype=self.dtype)
        return coordinates

    def cache_neural_data(self, session_id):
        assert self.cache, "Cache is not enabled"
        if session_id in self.neural_data_cache: return
        
        session_hash = self.sessions[session_id]
        h5_path = os.path.join(MGH_ROOT_DIR, 'h5', session_hash + '.h5')
        original_h5_electrode_labels = self.session_metadata[session_id]['channel_names']
        original_h5_electrode_ids = [original_h5_electrode_labels.index(e) for e in self.get_electrode_labels(session_id)]
        original_h5_electrode_keys = np.array(original_h5_electrode_ids)#["channel_"+str(i) for i in original_h5_electrode_ids]

        with h5py.File(h5_path, 'r', locking=False) as f:
            self.neural_data_cache[session_id] = torch.from_numpy(f['data'][original_h5_electrode_keys, :]).to(self.dtype)

    def clear_neural_data_cache(self, session_id=None):
        if session_id is None:
            self.neural_data_cache = {}
            self.h5_files = {}
            self.session_annotations = {}
            self.electrode_data_length = {}
        else:
            if session_id in self.neural_data_cache: del self.neural_data_cache[session_id]
            if session_id in self.h5_files: del self.h5_files[session_id]
            if session_id in self.session_annotations: del self.session_annotations[session_id]

    def open_neural_data_file(self, session_id):
        assert not self.cache, "Cache is enabled; Use cache_neural_data() instead"
        if session_id in self.h5_files: return
        
        session_hash = self.sessions[session_id]
        h5_path = os.path.join(MGH_ROOT_DIR, 'h5', session_hash + '.h5')
        self.h5_files[session_id] = h5py.File(h5_path, 'r', locking=False)

    def load_annotations(self, session_id=None):
        sessions = range(len(self.sessions)) if session_id is None else [session_id]
        for session_id in sessions:
            session_hash = self.sessions[session_id]
            annotation_file = os.path.join(MGH_ROOT_DIR, "annotations", session_hash + '.json')
            with open(annotation_file, "r") as f:
                self.session_annotations[session_id] = json.load(f)

    def load_neural_data(self, session_id):
        self.load_annotations(session_id)
        if self.cache: self.cache_neural_data(session_id)
        else: self.open_neural_data_file(session_id)

    def get_annotations(self, session_id, window_from=None, window_to=None, 
                        remove_persyst=True, remove_cashlab=True, remove_software_changes=True):
        if session_id not in self.session_annotations: self.load_annotations(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id] / self.get_sampling_rate(session_id)
        
        annotation_onsets = np.array([a['onset'] for a in self.session_annotations[session_id]])
        annotation_descriptions = np.array([a['description'] for a in self.session_annotations[session_id]])
        mask = (annotation_onsets >= window_from) & (annotation_onsets <= window_to)
        if remove_persyst:
            mask = (~np.array(['Persyst' in a['description'] for a in self.session_annotations[session_id]])) & mask
        if remove_cashlab:
            mask = (~np.array(['Cashlab' in a['description'] for a in self.session_annotations[session_id]])) & mask
        if remove_software_changes:
            mask = (~np.array([' Change' in a['description'] for a in self.session_annotations[session_id]])) & mask
        return annotation_onsets[mask], annotation_descriptions[mask]
    

    def get_electrode_data(self, electrode_label, session_id, window_from=None, window_to=None):
        if session_id not in self.electrode_data_length: self.load_neural_data(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id]

        if self.cache:
            electrode_id = self.electrode_ids[electrode_label]
            return self.neural_data_cache[session_id][electrode_id][window_from:window_to]
        else:
            original_h5_electrode_labels = self.session_metadata[session_id]['channel_names']
            electrode_id = original_h5_electrode_labels.index(electrode_label)
            h5_electrode_key = "channel_" + str(electrode_id)
            return torch.from_numpy(self.h5_files[session_id]['data'][h5_electrode_key][window_from:window_to]).to(self.dtype)

    def get_all_electrode_data(self, session_id, window_from=None, window_to=None):
        if session_id not in self.electrode_data_length: self.load_neural_data(session_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[session_id]

        if self.cache:
            return self.neural_data_cache[session_id][:, window_from:window_to]
        else:
            session_hash = self.sessions[session_id]
            h5_path = os.path.join(MGH_ROOT_DIR, 'h5', session_hash + '.h5')
            original_h5_electrode_labels = self.session_metadata[session_id]['channel_names']
            original_h5_electrode_ids = [original_h5_electrode_labels.index(e) for e in self.get_electrode_labels(session_id)]
            original_h5_electrode_keys = np.array(original_h5_electrode_ids)#["channel_"+str(i) for i in original_h5_electrode_ids]

            with h5py.File(h5_path, 'r', locking=False) as f:
                data = torch.from_numpy(f['data'][original_h5_electrode_keys, window_from:window_to]).to(self.dtype)
            return data

if __name__ == "__main__":
    subject = MGH2024Subject(22, cache=False)
    print(subject.electrode_labels)
    print(subject.get_n_electrodes())
    print(subject.get_all_electrode_data(0, window_from=0, window_to=1000).shape)


# if __name__ == "__main__":
#     for subject_id in range(1, 62):
#         subject = MGHSubject(subject_id, cache=False)
#         sessions = subject.sessions

#         print(f"{subject.subject_identifier} (n_electrodes={subject.get_n_electrodes()}, n_sessions={len(sessions)})")

#         for session_id in range(len(sessions)):
#             session_hash = sessions[session_id]
#             metadata = subject.session_metadata[session_id]

#             electrode_labels = subject.get_electrode_labels(session_id)
            
#             EEG_CHANNEL_NAME_MAPPING = {
#                 'T3': 'T7',
#                 'T4': 'T8',
#                 'T5': 'P7',
#                 'T6': 'P8'
#             }
#             EEG_channels = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6'] + list(EEG_CHANNEL_NAME_MAPPING.keys()) + list(EEG_CHANNEL_NAME_MAPPING.values())
#             upper_EEG_channels = [e.upper() for e in EEG_channels]
#             n_EEG_channels = len([e for e in electrode_labels if e.upper() in upper_EEG_channels])
#             n_non_EEG_channels = len(electrode_labels) - n_EEG_channels

#             electrode_coordinates = subject.get_electrode_coordinates(session_id)
#             n_non_nan_coordinates = torch.sum(~torch.isnan(electrode_coordinates[:, 0]))

#             session_length = metadata['session_length']
#             hours = int(session_length // 3600)
#             minutes = int((session_length % 3600) // 60)
#             seconds = int(session_length % 60)
#             session_length_str = f"{hours}h{minutes}m{seconds}s"
            
#             print(f"\t{subject.subject_identifier}_{session_id}:\t{session_length_str},\tsampling_rate={metadata['sampling_rate']},\tn_electrodes={len(electrode_labels)},\tn_EEG_electrodes={n_EEG_channels},\tn_non_EEG_electrodes={n_non_EEG_channels},\tn_coordinates={n_non_nan_coordinates}")

