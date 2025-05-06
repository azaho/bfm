import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from .braintreebank_subject import BrainTreebankSubject
from .btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import numpy as np
from .btbench_config import *


def generate_splits_DS_DM(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          lite=False, allow_partial_cache=True):
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
        lite (bool): if True, the eval is BTBench-Lite (the default), otherwise it is BTBench-Full.
        allow_partial_cache (bool): if True, the dataset will allow partial caching of the neural data. Defaults to True.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of training datasets
            - test_dataset (Dataset): Dataset for the test subject and trial
    """
    assert test_subject_id != DS_DM_TRAIN_SUBJECT_ID, "Test subject cannot be the same as the training subject."

    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, allow_partial_cache=allow_partial_cache)
    
    train_subject_id, train_trial_id = DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID
    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[train_subject_id], train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite, allow_partial_cache=allow_partial_cache)

    return train_dataset, test_dataset


def generate_splits_DS_SM(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=END_NEURAL_DATA_AFTER_WORD_ONSET,
                          lite=False, allow_partial_cache=True):
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
        lite (bool): if True, the eval is BTBench-Lite (the default), otherwise it is BTBench-Full.
        allow_partial_cache (bool): if True, the dataset will allow partial caching of the neural data. Defaults to True.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of training datasets
            - test_dataset (Dataset): Dataset for the test subject and trial
    """
    test_movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{test_subject_id}_{test_trial_id}"]
    other_subject_trials_list = []
    for subject_id, trial_id in BTBENCH_LITE_SUBJECT_TRIALS if lite else BTBENCH_FULL_SUBJECT_TRIALS:
        if BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{subject_id}_{trial_id}"] == test_movie_name and subject_id != test_subject_id:
            other_subject_trials_list.append((subject_id, trial_id))

    if len(other_subject_trials_list) == 0: raise ValueError(f"Trial {test_trial_id} of the test subject {test_subject_id} (movie name: {test_movie_name}) has no other subjects to train on which have the same movie trials.")

    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, allow_partial_cache=allow_partial_cache)
    train_datasets = []
    for other_subject_id, other_trial_id in other_subject_trials_list:
        train_datasets.append(BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[other_subject_id], other_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                        lite=lite, allow_partial_cache=allow_partial_cache))
    #train_dataset = ConcatDataset(train_datasets)
    return train_datasets, test_dataset
    

def generate_splits_SS_DM(test_subject, test_trial_id, eval_name, dtype=torch.float32, max_other_trials=3,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          lite=False, allow_partial_cache=True):
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
        lite (bool): if True, the eval is BTBench-Lite (the default), otherwise it is BTBench-Full.
        allow_partial_cache (bool): if True, the dataset will allow partial caching of the neural data. Defaults to True.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of training datasets
            - test_dataset (Dataset): Dataset for the test trial
    """
    assert len(BTBENCH_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id]) > 1, f"Training subject must have at least two trials. But subject {test_subject.subject_id} has only {len(BTBENCH_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id])} trials."

    
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, allow_partial_cache=allow_partial_cache)
    
    if not lite:
        train_trial_id = BTBENCH_LONGEST_TRIALS_FOR_SUBJECT[0]
        if train_trial_id == test_trial_id:
            train_trial_id = BTBENCH_LONGEST_TRIALS_FOR_SUBJECT[1] # If the longest trial is the test trial, use the second longest trial for training
    else:
        train_trial_id = [trial_id for subject_id, trial_id in BTBENCH_LITE_SUBJECT_TRIALS if subject_id == test_subject.subject_id and trial_id != test_trial_id][0] # Get the first other trial for the training set (there should only be one)
    

    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite, allow_partial_cache=allow_partial_cache)
    return train_dataset, test_dataset


def generate_splits_SS_SM(test_subject, test_trial_id, eval_name, k_folds=5, dtype=torch.float32,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          lite=False, allow_partial_cache=True):
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
        k_folds (int, optional): Number of folds for cross validation. Defaults to 5.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is BTBench-Lite (the default), otherwise it is BTBench-Full.
        allow_partial_cache (bool): if True, the dataset will allow partial caching of the neural data. Defaults to True.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of k training dataset splits
            - test_datasets (list): List of k test dataset splits, which correspond to the train datasets in the array above
    """

    train_datasets = []
    test_datasets = []

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                        lite=lite, allow_partial_cache=allow_partial_cache)
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        
        train_dataset = Subset(dataset, train_idx)
        train_datasets.append(train_dataset)
        test_datasets.append(Subset(dataset, test_idx))

    return train_datasets, test_datasets


if __name__ == "__main__":
    lite = True
    eval_name = "volume"
    test_subject_id, test_trial_id = 3, 0
    all_subjects = {subject_id: BrainTreebankSubject(subject_id, cache=False) for subject_id in range(1, 11)}

    print("= LOADING DATASETS = DS-DM (Different Subject Different Movie)")
    train_datasets, test_datasets = generate_splits_DS_DM(all_subjects, test_subject_id, test_trial_id, eval_name, lite=lite)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = DS-SM (Different Subject Same Movie)")
    train_datasets, test_datasets = generate_splits_DS_SM(all_subjects, test_subject_id, test_trial_id, eval_name, lite=lite)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = SS-DM (Single Subject Different Movie)")
    train_datasets, test_datasets = generate_splits_SS_DM(all_subjects[test_subject_id], test_trial_id, eval_name, lite=lite)
    print(train_datasets)
    print(test_datasets)

    print("= LOADING DATASETS = SS-SM (Single Subject Same Movie)")
    train_datasets, test_datasets = generate_splits_SS_SM(all_subjects[test_subject_id], test_trial_id, eval_name, lite=lite)
    print(train_datasets)
    print(test_datasets)