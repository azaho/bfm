import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from .btbench_config import *

single_float_variables_name_remapping = {
    "pitch": "pitch",
    "volume": "rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "delta_pitch": "delta_pitch",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length"
}
four_way_cardinal_directions_name_remapping = {
    "global_flow_angle": "max_global_angle",
    "local_flow_angle": "max_vector_angle",
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos"
}
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys())
four_way_cardinal_direction_variables = list(four_way_cardinal_directions_name_remapping.values()) + list(four_way_cardinal_directions_name_remapping.keys())
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + four_way_cardinal_direction_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index", "speaker"] + classification_variables
        
class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, 
                 start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE),
                 end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
                Options for eval_name (from the BTBench paper):
                    frame_brightness, global_flow, local_flow, global_flow_angle, local_flow_angle, face_num, volume, pitch, delta_volume, 
                    delta_pitch, speech, onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech, speaker

            output_indices (bool): 
                if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
                if False, the dataset will output the neural data directly
            
            start_neural_data_before_word_onset (int): the number of samples to start the neural data before each word onset
            end_neural_data_after_word_onset (int): the number of samples to end the neural data after each word onset
        """
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.output_indices = output_indices
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset    

        eval_name_remapped = eval_name
        if eval_name in single_float_variables_name_remapping: eval_name_remapped = single_float_variables_name_remapping[eval_name]
        if eval_name in four_way_cardinal_directions_name_remapping: eval_name_remapped = four_way_cardinal_directions_name_remapping[eval_name]
        if eval_name in classification_variables_name_remapping: eval_name_remapped = classification_variables_name_remapping[eval_name]
        self.eval_name_remapped = eval_name_remapped

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)
        self.subject.load_neural_data(self.trial_id)

        rebalance_classes = False # setting this flag as false by default; it is only relevant for classification tasks
        if eval_name == "word_gap": # create the word gap column
            word_gaps = []
            for i in range(len(self.all_words_df)):
                if i == 0 or self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']:
                    word_gaps.append(-1) 
                else:
                    gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                    word_gaps.append(gap)
            self.all_words_df['word_gap'] = word_gaps
        
        if eval_name in single_float_variables:
            # Get indices for words in top and bottom quartiles
            all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            self.extreme_indices = np.where((label_percentiles > 0.75) | (label_percentiles < 0.25))[0]
            self.extreme_labels = torch.from_numpy((label_percentiles[self.extreme_indices] > 0.75).astype(int))
            self.n_samples = len(self.extreme_indices)
            self.__getitem__ = self._simple_float_variable__getitem__
        elif eval_name in ["onset", "speech"]:
            self.positive_indices = np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.all_words_df))
            self.negative_indices = np.arange(len(self.nonverbal_df))
            min_len = min(len(self.positive_indices), len(self.negative_indices)) # make sure we have an equal number of positive and negative samples
            self.positive_indices = np.sort(np.random.choice(self.positive_indices, size=min_len, replace=False))
            self.negative_indices = np.sort(np.random.choice(self.negative_indices, size=min_len, replace=False))
            self.n_samples = len(self.positive_indices) + len(self.negative_indices)
            self.__getitem__ = self._onset_speech__getitem__
        elif eval_name in four_way_cardinal_direction_variables:
            self.class_labels = np.zeros(len(self.all_words_df), dtype=int)
            angles = self.all_words_df[self.eval_name_remapped].to_numpy()
            cardinal_directions = np.array([0, 90, 180, 270])
            angles_expanded = angles[:, np.newaxis]
            distances = np.minimum(np.abs(angles_expanded - cardinal_directions),
                                360 - np.abs(angles_expanded - cardinal_directions))
            self.class_labels = np.argmin(distances, axis=1)
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "face_num":
            self.n_samples = len(self.all_words_df)
            self.class_labels = self.all_words_df["face_num"].to_numpy().astype(int)
            self.class_labels[self.class_labels > 1] = 2 # cap at 2
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "word_index":
            self.n_samples = len(self.all_words_df)
            self.class_labels = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            self.class_labels[self.class_labels > 3] = 3 # cap at 3
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "word_head_pos":
            self.n_samples = len(self.all_words_df)
            self.class_labels = self.all_words_df[self.eval_name_remapped].to_numpy().astype(int)
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "word_part_speech":
            self.n_samples = len(self.all_words_df)
            self.class_labels = np.ones(len(self.all_words_df)).astype(int) * 3
            for i, pos in enumerate(["NOUN", "VERB", "PRON"]):
                self.class_labels[self.all_words_df[self.eval_name_remapped] == pos] = i
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "speaker":
            self.n_samples = len(self.all_words_df)
            self.class_labels = np.ones(len(self.all_words_df)).astype(int) * 3
            most_frequent_speakers = self.all_words_df['speaker'].value_counts().index
            for i, speaker in enumerate(most_frequent_speakers[:3]):
                self.class_labels[self.all_words_df['speaker'] == speaker] = i
            self.__getitem__ = self._classification__getitem__
            rebalance_classes = True
        elif eval_name == "word_gap":
            # Get indices for words in top and bottom quartiles, ignoring -1 values
            all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()
            valid_mask = all_labels != -1
            valid_labels = all_labels[valid_mask]
            label_percentiles = np.array([np.mean(valid_labels < x) for x in all_labels[valid_mask]])
            valid_indices = np.where(valid_mask)[0]
            extreme_mask = (label_percentiles > 0.75) | (label_percentiles < 0.25)
            self.extreme_indices = valid_indices[extreme_mask]
            self.extreme_labels = torch.from_numpy((label_percentiles[extreme_mask] > 0.75).astype(int))
            self.n_samples = len(self.extreme_indices)
            self.__getitem__ = self._simple_float_variable__getitem__

        if rebalance_classes:
            # Get counts for each class
            unique_classes, class_counts = np.unique(self.class_labels, return_counts=True)
            min_count = np.min(class_counts)
            # Create balanced subset by randomly sampling min_count elements from each class
            balanced_indices = []
            for class_label in unique_classes:
                class_indices = np.where(self.class_labels == class_label)[0]
                sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
                sampled_indices = np.sort(sampled_indices)
                balanced_indices.extend(sampled_indices)
            self.balanced_indices = np.sort(np.array(balanced_indices))
            self.class_labels = self.class_labels[self.balanced_indices]
            self.n_samples = len(self.balanced_indices)

    def _get_neural_data(self, window_from, window_to):
        if not self.output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            return input.to(dtype=self.dtype)
        else:
            return window_from, window_to # just return the window indices

    def _simple_float_variable__getitem__(self, idx):
        word_index = self.extreme_indices[idx]
        row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx)
        return input, self.extreme_labels[idx].item()

    def _onset_speech__getitem__(self, idx):
        if idx % 2 == 0: # even indices are positive samples
            word_index = self.positive_indices[idx//2]
            row = self.all_words_df.iloc[word_index]
            est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
            est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
            input = self._get_neural_data(est_idx, est_end_idx)
            return input, 1
        else: # odd indices are negative samples
            item_index = self.negative_indices[idx//2]
            row = self.nonverbal_df.iloc[item_index]
            est_idx = int(row['est_idx'])
            est_end_idx = int(est_idx) + int(self.end_neural_data_after_word_onset) + int(self.start_neural_data_before_word_onset)
            input = self._get_neural_data(est_idx, est_end_idx)
            return input, 0
        
    def _classification__getitem__(self, idx):
        word_index = self.balanced_indices[idx]
        row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx)
        return input, self.class_labels[idx].item()
        
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.eval_name in single_float_variables or self.eval_name == "word_gap":
            return self._simple_float_variable__getitem__(idx)
        elif self.eval_name in four_way_cardinal_direction_variables or self.eval_name in ["face_num", "word_index", "word_head_pos", "word_part_speech", "speaker"]:
            return self._classification__getitem__(idx)
        elif self.eval_name in ["onset", "speech"]:
            return self._onset_speech__getitem__(idx)
        else:
            raise ValueError(f"Invalid eval_name: {self.eval_name}")