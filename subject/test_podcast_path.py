#!/usr/bin/env python3
"""
Test script to verify podcast file path construction.
"""

import os
import mne
from mne_bids import BIDSPath

# Use the same path as in the podcast.py file
PODCAST_ROOT_DIR = "/om2/data/public/fietelab/ecog-the-podcast"

def test_path_construction():
    """Test the path construction for podcast data."""
    print("Testing podcast file path construction...")
    
    # Test with subject 02 (as in the notebook)
    subject_id = 2
    
    # Construct the file path using BIDSPath (relative path)
    file_path = BIDSPath(
        root="derivatives/ecogprep",
        subject=f"{subject_id:02d}",
        task="podcast",
        datatype="ieeg",
        description="highgamma",
        suffix="ieeg",
        extension="fif"
    )
    
    # Construct the full path
    full_path = os.path.join(PODCAST_ROOT_DIR, str(file_path))
    
    print(f"BIDSPath: {file_path}")
    print(f"Full path: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")
    
    # Check if the directory exists
    dir_path = os.path.dirname(full_path)
    print(f"Directory exists: {os.path.exists(dir_path)}")
    
    # List files in the directory if it exists
    if os.path.exists(dir_path):
        print(f"Files in directory: {os.listdir(dir_path)}")
    
    # Try to load the file if it exists
    if os.path.exists(full_path):
        try:
            raw = mne.io.read_raw_fif(full_path, verbose=False)
            print(f"Successfully loaded file!")
            print(f"Number of channels: {len(raw.ch_names)}")
            print(f"Number of samples: {raw.n_times}")
            print(f"Sampling rate: {raw.info['sfreq']} Hz")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    else:
        print("File not found!")
        return False

if __name__ == "__main__":
    success = test_path_construction()
    if success:
        print("\n✅ Path construction test passed!")
    else:
        print("\n❌ Path construction test failed!") 