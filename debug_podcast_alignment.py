#!/usr/bin/env python3
"""
Debug script to check podcast electrode counts
"""
from subject.podcast import PodcastSubject

def debug_podcast_alignment():
    print("=== PODCAST ELECTRODE COUNTS ===\n")
    
    # Check all subjects
    for subject_id in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        try:
            print(f"Loading subject {subject_id}...")
            subject = PodcastSubject(subject_id, cache=False)
            subject.load_neural_data(0)
            
            # Get electrode count
            if hasattr(subject, 'electrode_labels'):
                n_electrodes = subject.get_n_electrodes()
                print(f"  Subject {subject_id}: {n_electrodes} electrodes")
            else:
                print(f"  Subject {subject_id}: electrode_names not found")
                
        except Exception as e:
            print(f"  Subject {subject_id}: Error - {e}")

if __name__ == "__main__":
    debug_podcast_alignment() 