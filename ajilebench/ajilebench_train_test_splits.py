import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from subject_ajile12 import AjileSubject
from ajilebench_datasets import AjileSubjectTrialBenchmarkDataset
import numpy as np
from ajilebench_config import *

_all_subject_trials = [
    (1, 3), (1, 4), (1, 5), (1, 7),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 3), (5, 4), (5, 7),
    (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
    (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
    (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
    (9, 3), (9, 4), (9, 5), (9, 6), (9, 7),
    (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
    (11, 3), (11, 4), (11, 5), (11, 6), (11, 7),
    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7),
]

def generate_splits_DS_DM(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_reach_onset=int(START_NEURAL_DATA_BEFORE_REACH_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_reach_onset=int(END_NEURAL_DATA_AFTER_REACH_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Different Subject Different Movie (DS-DM) evaluation.
    
    This function creates train/test splits by using one subject and movie as the test set,
    and using all other subjects and movies (except the test movie) as the training set.
    This evaluates generalization across both subjects and movie content.

    Args:
        all_subjects (dict): Dictionary mapping subject IDs to Subject objects
        test_subject_id (int): ID of the subject to use as test set
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_dataset (ConcatDataset): Combined dataset of all training subjects and trials
            - test_dataset (Dataset): Dataset for the test subject and trial
    """
    test_dataset = AjileSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
    train_subject_trials = [(subject_id, trial_id) for subject_id, trial_id in _all_subject_trials if subject_id != test_subject_id and trial_id != test_trial_id]

    train_datasets = []
    for train_subject_id, train_trial_id in train_subject_trials:
        train_dataset = AjileSubjectTrialBenchmarkDataset(all_subjects[train_subject_id], train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                  output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
        train_datasets.append(train_dataset)

    train_dataset = ConcatDataset(train_datasets)
    return train_dataset, test_dataset


def generate_splits_DS_SM(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_reach_onset=int(START_NEURAL_DATA_BEFORE_REACH_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_reach_onset=int(END_NEURAL_DATA_AFTER_REACH_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Different Subject Same Movie (DS-SM) evaluation.
    
    This function creates train/test splits by using one subject and movie as the test set,
    and using the same movie from all other subjects as the training set. This evaluates
    generalization across subjects while controlling for the movie content.

    Args:
        all_subjects (dict): Dictionary mapping subject IDs to Subject objects
        test_subject_id (int): ID of the subject to use as test set
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_dataset (ConcatDataset): Combined dataset of all training subjects for the test trial
            - test_dataset (Dataset): Dataset for the test subject and trial
    """
    trial_subject_mapping = {}
    for subject_id, trial_id in _all_subject_trials:
        if trial_id not in trial_subject_mapping:
            trial_subject_mapping[trial_id] = []
        trial_subject_mapping[trial_id].append(subject_id)
    other_subject_id_list = [subject_id for subject_id in trial_subject_mapping[test_trial_id] if subject_id != test_subject_id]
    if len(other_subject_id_list) == 0: raise ValueError(f"Trial {test_subject_id} has no other subjects to train on.")

    test_dataset = AjileSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
    train_datasets = []
    for other_subject_id in other_subject_id_list:
        train_datasets.append(AjileSubjectTrialBenchmarkDataset(all_subjects[other_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                        output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset))
    train_dataset = ConcatDataset(train_datasets)
    return train_dataset, test_dataset


def generate_subject_trials_for_SS_DM():
    """Generate list of all subject-trial pairs in the dataset that are valid for SS-DM evaluation.
    (i.e. subjects that contain at least two trials)
    
    This function creates a list of tuples containing all valid subject-trial pairs
    based on the available recordings in the BrainTreebank dataset. Each tuple contains
    a subject ID and trial ID.

    Returns:
        list: List of tuples, where each tuple contains:
            - subject_id (int): ID of the subject
            - trial_id (int): ID of the trial/movie for that subject
    """
    _subject_trials = {
        1: [0, 1, 2],
        2: [0, 1, 2, 3, 4, 5, 6],
        3: [0, 1, 2],
        4: [0, 1, 2],
        5: [0],
        6: [0, 1, 4],
        7: [0, 1],
        8: [0],
        9: [0],
        10: [0, 1],
    }
    for key in _subject_trials:
        if len(_subject_trials[key]) == 0:
            del _subject_trials[key]
    subject_trials = []
    for key in _subject_trials:
        for trial in _subject_trials[key]:
            subject_trials.append((key, trial))
    return subject_trials
    

def generate_splits_SS_DM(test_subject, test_trial_id, eval_name, dtype=torch.float32, max_other_trials=3,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_reach_onset=int(START_NEURAL_DATA_BEFORE_REACH_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_reach_onset=int(END_NEURAL_DATA_AFTER_REACH_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Single Subject Different Movies (SS-DM) evaluation.
    
    This function creates train/test splits by using one movie as the test set and all other
    movies from the same subject as the training set (trimmed at max_other_trials movies). 
    Unlike SS-SM, this does not perform k-fold cross validation since movies are already naturally separated.

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        max_other_trials (int, optional): Maximum number of other trials to include in the training set. Defaults to 2.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_reach_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_reach_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_dataset (ConcatDataset): Combined dataset of all training trials
            - test_dataset (Dataset): Dataset for the test trial
    """
    # Define subject-to-trials mapping
    _subject_trials = {
        1: [0, 1, 2],
        2: [0, 1, 2, 3, 4, 5, 6],
        3: [0, 1, 2],
        4: [0, 1, 2],
        5: [0],
        6: [0, 1, 4],
        7: [0, 1],
        8: [0],
        9: [0],
        10: [0, 1],
    }
    # Choose subject
    test_subject_id = test_subject.subject_id  # Change this to test different subjects
    if test_subject_id not in _subject_trials: raise ValueError(f"Subject {test_subject_id} not found in dataset.")

    train_trials = [t for t in _subject_trials[test_subject_id] if t != test_trial_id][:max_other_trials]
    if len(train_trials) == 0: raise ValueError(f"Subject {test_subject_id} has no training trials.")

    test_dataset = AjileSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
    # Load training datasets dynamically
    train_datasets = []
    for train_trial_id in train_trials:
        train_dataset = AjileSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                  output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
        train_datasets.append(train_dataset)
    train_dataset = ConcatDataset(train_datasets)
    return train_dataset, test_dataset


def generate_splits_SS_SM(test_subject, test_trial_id, eval_name, add_other_trials=False, k_folds=5, dtype=torch.float32, gap_length=None,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_reach_onset=int(START_NEURAL_DATA_BEFORE_REACH_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_reach_onset=int(END_NEURAL_DATA_AFTER_REACH_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Single Subject Single Movie (SS-SM) evaluation.
    
    This function performs k-fold cross validation on data from a single subject and movie.
    If gap_length is specified and not None, it ensures temporal gaps between train and test sets to avoid
    temporal correlation in the data. For example, if gap_length=300, there will be at least
    300 seconds between any training and test samples. If gap_length is None, no temporal gap
    is enforced between train and test sets.

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use
        eval_name (str): Name of the evaluation metric to use (e.g. "rms", "word_gap", "pitch", "delta_volume")
        add_other_trials (bool, optional): Whether to add other movies from the same subject to the training set. Defaults to False.
        k_folds (int, optional): Number of folds for cross validation. Defaults to 5.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        gap_length (int, optional): Minimum temporal gap in seconds between train and test sets. If None, no gap is enforced. Defaults to None.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_reach_onset (int, optional): Number of seconds before the reach onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_REACH_ONSET.
        end_neural_data_after_reach_onset (int, optional): Number of seconds after the reach onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_REACH_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of k training dataset splits
            - test_datasets (list): List of k test dataset splits, which correspond to the train datasets in the array above
    """
    assert gap_length is None, "gap_length is not fully implemented yet (doesn't work for some tasks, for example speech, because there is no direct correspondence between words and samples)"

    train_datasets = []
    test_datasets = []

    dataset = AjileSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                        output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset)
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!

    if add_other_trials:
        other_trials = [_trial_id for _subject_id, _trial_id in _all_subject_trials if _subject_id == test_subject.subject_id and _trial_id != test_trial_id]
        other_trials_dataset = ConcatDataset([AjileSubjectTrialBenchmarkDataset(test_subject, t, dtype=dtype, eval_name=eval_name, 
                                                                                        output_indices=output_indices, start_neural_data_before_reach_onset=start_neural_data_before_reach_onset, end_neural_data_after_reach_onset=end_neural_data_after_reach_onset) for t in other_trials])

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        
        train_dataset = Subset(dataset, train_idx)
        if add_other_trials:
            train_dataset = ConcatDataset([other_trials_dataset, train_dataset])
        train_datasets.append(train_dataset)
        test_datasets.append(Subset(dataset, test_idx))

    return train_datasets, test_datasets


if __name__ == "__main__":
    eval_name = "rms"
    test_subject_id, test_trial_id = 3, 0
    all_subjects = {subject_id: AjileSubject(subject_id, cache=False) for subject_id in range(1, 11)}

    print("= LOADING DATASETS = DS-DM (Different Subject Different Movie)")
    train_datasets, test_datasets = generate_splits_DS_DM(all_subjects, test_subject_id, test_trial_id, eval_name)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = DS-SM (Different Subject Same Movie)")
    train_datasets, test_datasets = generate_splits_DS_SM(all_subjects, test_subject_id, test_trial_id, eval_name)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = SS-DM (Single Subject Different Movie)")
    train_datasets, test_datasets = generate_splits_SS_DM(all_subjects[test_subject_id], test_trial_id, eval_name)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = SS-SM (Single Subject Same Movie)")
    train_datasets, test_datasets = generate_splits_SS_SM(all_subjects[test_subject_id], test_trial_id, eval_name)
    print(train_datasets)
    print(test_datasets)