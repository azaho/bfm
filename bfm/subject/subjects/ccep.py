import json
import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mne
from mne_bids import BIDSPath, read_raw_bids

from bfm.subject.base import Subject
from bfm.subject.registry import subjects

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

CCEP_ROOT_DIR = os.getenv("CCEP_ROOT_DIR")  # Root directory for the CCEP OpenNeuro data

@subjects.register("ccep")
class CCEPSubject(Subject):
    """ 
    This class is used to load the neural data for a given CCEP subject from the OpenNeuro BIDS dataset.
    Each BIDS run is treated as a separate session for compatibility with the Subject interface.
    """
    def __init__(self, subject_id, allow_corrupted=False, cache=False, dtype=torch.float32, suppress_warnings=True):
        # CCEP subjects range from 1-75 (ccepAgeUMCU01 to ccepAgeUMCU75)
        assert subject_id >= 1 and subject_id <= 75, f"Subject ID must be between 1 and 75, got {subject_id}"

        self.subject_id = subject_id
        self.subject_identifier = f'ccepAgeUMCU{subject_id:02d}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype
        self.verbose = not suppress_warnings

        if CCEP_ROOT_DIR is None:
            raise ValueError("CCEP_ROOT_DIR environment variable not set. Please set it to the path of the OpenNeuro CCEP dataset.")

        self.root_dir = Path(CCEP_ROOT_DIR)
        self.subject_dir = self.root_dir / f"sub-{self.subject_identifier}"
        
        if not self.subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")

        # Discover available sessions (treating each BIDS run as a session)
        self._discover_sessions()
        
        # Load electrode information and metadata for each session
        self.electrode_labels = {}
        self.electrode_ids = {}
        self.electrode_coordinates = {}
        self.sampling_rates = {}
        self.session_metadata = {}
        self.raw_objects = {}  # Cache for MNE raw objects
        self.neural_data_cache = {}
        self.electrode_data_length = {}

        # Load metadata for each session
        for session_id in self.sessions:
            self._load_session_metadata(session_id)

    def _discover_sessions(self):
        """Discover available sessions (BIDS runs) for this subject"""
        self.sessions = []  # List of session IDs (run IDs)
        self.bids_session = None  # The BIDS session name (e.g., "1" or "1b")
        
        # Find the session directory (should be only one)
        session_dirs = list(self.subject_dir.glob("ses-*"))
        
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found for subject {self.subject_identifier}")
        
        # Use the first (and typically only) session directory
        session_dir = session_dirs[0]
        self.bids_session = session_dir.name.split('-')[1]  # e.g., "1" from "ses-1"
        
        ieeg_dir = session_dir / "ieeg"
        if not ieeg_dir.exists():
            raise FileNotFoundError(f"No ieeg directory found in {session_dir}")
            
        # Find all runs for this session - each run becomes a session in our interface
        run_files = list(ieeg_dir.glob("*_task-SPESclin_run-*_ieeg.vhdr"))
        
        for run_file in run_files:
            # Extract run ID from filename
            # e.g., sub-ccepAgeUMCU03_ses-1_task-SPESclin_run-031411_ieeg.vhdr
            run_match = re.search(r'run-(\w+)_ieeg\.vhdr', run_file.name)
            if run_match:
                self.sessions.append(run_match.group(1))
        
        self.sessions = sorted(self.sessions)

    def _load_session_metadata(self, session_id):
        """Load metadata for a specific session (BIDS run)"""
        # Load electrode information from BIDS files
        session_dir = self.subject_dir / f"ses-{self.bids_session}" / "ieeg"
        
        # Load electrodes.tsv file (same for all runs in a BIDS session)
        electrodes_file = session_dir / f"sub-{self.subject_identifier}_ses-{self.bids_session}_electrodes.tsv"
        if electrodes_file.exists():
            electrodes_df = pd.read_csv(electrodes_file, sep='\t')

            # Drop any rows that contain NaN values -- those are usually non-neural channels TODO: check if this is correct
            electrodes_df = electrodes_df.dropna()
            
            # Filter out bad channels if not allowing corrupted
            if not self.allow_corrupted and 'status' in electrodes_df.columns:
                good_electrodes = electrodes_df[electrodes_df['status'] == 'good']
            else:
                good_electrodes = electrodes_df
                
            self.electrode_labels[session_id] = good_electrodes['name'].tolist()
            self.electrode_ids[session_id] = {e: i for i, e in enumerate(self.electrode_labels[session_id])}
            
            # Load coordinates if available
            if all(col in electrodes_df.columns for col in ['x', 'y', 'z']):
                coords = good_electrodes[['x', 'y', 'z']].values
                # Handle NaN values
                coords = np.where(np.isnan(coords), float('nan'), coords)
                self.electrode_coordinates[session_id] = torch.tensor(coords, dtype=self.dtype)
            else:
                # Create NaN coordinates if not available
                n_electrodes = len(self.electrode_labels[session_id])
                self.electrode_coordinates[session_id] = torch.full((n_electrodes, 3), float('nan'), dtype=self.dtype)
        
        # Load sampling rate from this run's JSON file
        json_file = session_dir / f"sub-{self.subject_identifier}_ses-{self.bids_session}_task-SPESclin_run-{session_id}_ieeg.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                self.sampling_rates[session_id] = metadata.get('SamplingFrequency', 1000.0)
                self.session_metadata[session_id] = metadata

    def get_n_electrodes(self, session_id=None):
        if session_id is None:
            # Return total unique electrodes across all sessions
            all_electrodes = set()
            for sess_electrodes in self.electrode_labels.values():
                all_electrodes.update(sess_electrodes)
            return len(all_electrodes)
        else:
            return len(self.electrode_labels.get(session_id, []))

    def get_electrode_indices(self, session_id=None):
        if session_id is None:
            return np.arange(self.get_n_electrodes())
        else:
            return np.arange(len(self.electrode_labels.get(session_id, [])))

    def get_electrode_labels(self, session_id=None):
        if session_id is None:
            # Return unique electrodes across all sessions
            all_electrodes = set()
            for sess_electrodes in self.electrode_labels.values():
                all_electrodes.update(sess_electrodes)
            return sorted(list(all_electrodes))
        else:
            return self.electrode_labels.get(session_id, [])

    def get_sampling_rate(self, session_id):
        return self.sampling_rates.get(session_id, 1000.0)

    def get_electrode_coordinates(self, session_id=None):
        """
        Get the coordinates of the electrodes for this subject
        Returns:
            coordinates: (n_electrodes, 3) tensor of coordinates (MNI space)
            if coordinates are not available, returns nan for that electrode
        """
        if session_id is None:
            # For now, just return coordinates from the first available session
            if self.electrode_coordinates:
                first_session = next(iter(self.electrode_coordinates.keys()))
                return self.electrode_coordinates[first_session]
            else:
                return torch.full((0, 3), float('nan'), dtype=self.dtype)
        else:
            return self.electrode_coordinates.get(session_id, 
                torch.full((0, 3), float('nan'), dtype=self.dtype))

    def get_available_sessions(self):
        """Get list of available session IDs (BIDS run IDs)"""
        return self.sessions.copy()

    def _load_raw_data(self, session_id):
        """Load raw data using MNE-BIDS for a specific session (BIDS run)"""
        if session_id in self.raw_objects:
            return self.raw_objects[session_id]
        
        try:
            bids_path = BIDSPath(
                subject=self.subject_identifier,
                session=self.bids_session,
                task='SPESclin',
                run=session_id,
                datatype='ieeg',
                root=self.root_dir
            )
            
            # Suppress common BIDS->MNE mapping warnings if not in verbose mode
            if not self.verbose:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No BIDS -> MNE mapping found")
                    warnings.filterwarnings("ignore", message="Unable to map the following column")
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne_bids")
                    raw = read_raw_bids(bids_path, verbose=False)
                    
                    # Pick only the electrodes that exist in this session and are in electrode_labels
                    electrode_labels = self.get_electrode_labels(session_id)
                    available_channels = [ch for ch in electrode_labels if ch in raw.ch_names]
                    raw = raw.pick(available_channels)
            else:
                raw = read_raw_bids(bids_path, verbose=True)
                
                # Pick only the electrodes that exist in this session and are in electrode_labels
                electrode_labels = self.get_electrode_labels(session_id)
                available_channels = [ch for ch in electrode_labels if ch in raw.ch_names]
                raw = raw.pick(available_channels)
            
            # Cache the raw object if caching is enabled
            if self.cache:
                self.raw_objects[session_id] = raw
                
            return raw
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load data for session {session_id}: {e}")

    def cache_neural_data(self, session_id):
        """Cache neural data for a specific session"""
        assert self.cache, "Cache is not enabled"
        
        if session_id in self.neural_data_cache:
            return
        
        raw = self._load_raw_data(session_id)
        
        # Get data for electrodes in this session
        electrode_labels = self.get_electrode_labels(session_id)
        if electrode_labels:
            # Pick only the electrodes that exist in this session
            available_channels = [ch for ch in electrode_labels if ch in raw.ch_names]
            if available_channels:
                data = raw.get_data(picks=available_channels)
                self.neural_data_cache[session_id] = torch.tensor(data, dtype=self.dtype)
                self.electrode_data_length[session_id] = data.shape[1]

    def clear_neural_data_cache(self, session_id=None):
        if session_id is None:
            self.neural_data_cache = {}
            self.raw_objects = {}
            self.electrode_data_length = {}
        else:
            if session_id in self.neural_data_cache:
                del self.neural_data_cache[session_id]
            if session_id in self.raw_objects:
                del self.raw_objects[session_id]
            if session_id in self.electrode_data_length:
                del self.electrode_data_length[session_id]

    def load_neural_data(self, session_id):
        """Load neural data for a specific session"""
        if self.cache:
            self.cache_neural_data(session_id)
        else:
            # Just load the raw object without caching data
            raw = self._load_raw_data(session_id)
            self.electrode_data_length[session_id] = raw.n_times

    def get_events(self, session_id, window_from=None, window_to=None):
        """Get CCEP stimulation events from the events.tsv file"""
        session_dir = self.subject_dir / f"ses-{self.bids_session}" / "ieeg"
        events_file = session_dir / f"sub-{self.subject_identifier}_ses-{self.bids_session}_task-SPESclin_run-{session_id}_events.tsv"
        
        if not events_file.exists():
            return np.array([]), np.array([])
        
        events_df = pd.read_csv(events_file, sep='\t')
        
        # Filter events by time window if specified
        if window_from is not None:
            events_df = events_df[events_df['onset'] >= window_from]
        if window_to is not None:
            events_df = events_df[events_df['onset'] <= window_to]
        
        return events_df['onset'].values, events_df['trial_type'].values

    def get_electrode_data(self, electrode_label, session_id, window_from=None, window_to=None):
        """Get data for a specific electrode in a specific session"""
        if session_id not in self.electrode_data_length:
            self.load_neural_data(session_id)
        
        if window_from is None:
            window_from = 0
        if window_to is None:
            window_to = self.electrode_data_length[session_id]
        
        if self.cache and session_id in self.neural_data_cache:
            # Get from cached data
            electrode_labels = self.get_electrode_labels(session_id)
            if electrode_label not in electrode_labels:
                raise ValueError(f"Electrode {electrode_label} not found in session {session_id}")
            
            electrode_idx = electrode_labels.index(electrode_label)
            return self.neural_data_cache[session_id][electrode_idx, window_from:window_to]
        else:
            # Load directly from raw data
            raw = self._load_raw_data(session_id)
            if electrode_label not in raw.ch_names:
                raise ValueError(f"Electrode {electrode_label} not found in session {session_id}")
            
            data = raw.get_data(picks=[electrode_label], start=window_from, stop=window_to)
            return torch.tensor(data[0], dtype=self.dtype)

    def get_all_electrode_data(self, session_id, window_from=None, window_to=None):
        """Get data for all electrodes in a specific session"""
        if session_id not in self.electrode_data_length:
            self.load_neural_data(session_id)
        
        if window_from is None:
            window_from = 0
        if window_to is None:
            window_to = self.electrode_data_length[session_id]
        
        if self.cache and session_id in self.neural_data_cache:
            return self.neural_data_cache[session_id][:, window_from:window_to]
        else:
            # Load directly from raw data
            raw = self._load_raw_data(session_id)
            electrode_labels = self.get_electrode_labels(session_id)
            
            # Pick only available electrodes
            available_channels = [ch for ch in electrode_labels if ch in raw.ch_names]
            if not available_channels:
                return torch.empty((0, window_to - window_from), dtype=self.dtype)
            
            data = raw.get_data(picks=available_channels, start=window_from, stop=window_to)
            return torch.tensor(data, dtype=self.dtype)

if __name__ == "__main__":
    # Test with subject 3 (ccepAgeUMCU03)
    subject = CCEPSubject(3, cache=False, verbose=False)  # Set verbose=True to see warnings
    print(f"Subject: {subject.subject_identifier}")
    print(f"Available sessions: {subject.get_available_sessions()}")
    
    if subject.get_available_sessions():
        session_id = subject.get_available_sessions()[0]
        print(f"Testing session: {session_id}")
        print(f"Number of electrodes: {subject.get_n_electrodes(session_id)}")
        print(f"Electrode labels: {subject.get_electrode_labels(session_id)[:5]}...")  # Show first 5
        print(f"Sampling rate: {subject.get_sampling_rate(session_id)} Hz")
        
        try:
            # Test loading data
            data = subject.get_all_electrode_data(session_id, window_from=0, window_to=1000)
            print(f"Data shape: {data.shape}")
            
            # Test loading events
            events_onset, events_type = subject.get_events(session_id)
            print(f"Number of events: {len(events_onset)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")

