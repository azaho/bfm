#!/usr/bin/env python3
"""
Debug script to check podcast temporal alignment
"""
import torch
from subject.podcast import PodcastSubject
from subject.podcast_pair import PodcastTrialPairDataset
import numpy as np
import matplotlib.pyplot as plt

def debug_podcast_alignment():
    print("Loading podcast subjects...")
    
    # Load two subjects
    subject_a = PodcastSubject(1, cache=False)
    subject_b = PodcastSubject(2, cache=False)
    
    # Load neural data first
    subject_a.load_neural_data(0)
    subject_b.load_neural_data(0)
    
    print(f"Subject A: {subject_a.subject_identifier}")
    print(f"Subject B: {subject_b.subject_identifier}")
    print(f"Subject A data length: {subject_a.electrode_data_length[0]}")
    print(f"Subject B data length: {subject_b.electrode_data_length[0]}")
    
    # Create paired dataset
    window_size = 1536  # 3 seconds at 512 Hz
    dataset = PodcastTrialPairDataset(
        subject_a, subject_b, window_size,
        dtype=torch.float32, output_metadata=True, output_electrode_labels=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Check a few samples
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Time point A: {sample['metadata']['time_point']:.2f}s")
        print(f"  Time point B: {sample['metadata_b']['time_point']:.2f}s")
        print(f"  Data shape A: {sample['data'].shape}")
        print(f"  Data shape B: {sample['data_b'].shape}")
        
        # Check if data is actually different
        data_diff = torch.abs(sample['data'] - sample['data_b']).mean()
        print(f"  Mean absolute difference: {data_diff:.6f}")
        
        # Check correlation
        corr = torch.corrcoef(sample['data'].flatten(), sample['data_b'].flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")
    
    # Check if the data is actually the same (which would be bad)
    print(f"\nChecking for data duplication...")
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    # Check if consecutive samples are different
    diff_consecutive = torch.abs(sample1['data'] - sample2['data']).mean()
    print(f"Difference between consecutive samples: {diff_consecutive:.6f}")
    
    # Check if subjects have different data
    diff_subjects = torch.abs(sample1['data'] - sample1['data_b']).mean()
    print(f"Difference between subjects: {diff_subjects:.6f}")
    
    if diff_subjects < 1e-6:
        print("WARNING: Subject data appears to be identical!")
        print("This suggests the temporal alignment is not working correctly.")
    else:
        print("Subject data appears to be different (good!)")

if __name__ == "__main__":
    debug_podcast_alignment() 