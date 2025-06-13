import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, json

from .config import *
from .braintreebank_subject import BrainTreebankSubject

# Defining the names of evaluations and preparing them for downstream processing
single_float_variables_name_remapping = {
    "pitch": "enhanced_pitch", #"pitch",
    "volume": "enhanced_volume", #"rms",
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
new_pitch_variables = ['enhanced_pitch', 'enhanced_volume', 'delta_enhanced_pitch', 'delta_enhanced_volume', 'raw_pitch', 'raw_volume', 'delta_raw_pitch', 'delta_raw_volume']
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys()) + new_pitch_variables
four_way_cardinal_direction_variables = list(four_way_cardinal_directions_name_remapping.values()) + list(four_way_cardinal_directions_name_remapping.keys())
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + four_way_cardinal_direction_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index", "speaker"] + classification_variables


class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, 
                 start_neural_data_before_word_onset=START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE, end_neural_data_after_word_onset=END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE,
                 lite=True, random_seed=NEUROPROBE_GLOBAL_RANDOM_SEED, allow_partial_cache=True, output_dict=False):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
                Options for eval_name (from the Neuroprobe paper):
                    frame_brightness, global_flow, local_flow, global_flow_angle, local_flow_angle, face_num, volume, pitch, delta_volume, 
                    delta_pitch, speech, onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech, speaker
            lite (bool): if True, the eval is Neuroprobe-Lite (the default), otherwise it is Neuroprobe-Full

            output_indices (bool): 
                if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
                if False, the dataset will output the neural data directly

            output_dict (bool): 
                if True, the dataset will output a dictionary with the following keys:
                    "data": the neural data -- either directly or as a tuple (index_from, index_to)
                    "label": the label
                    "electrode_labels": the labels of the electrodes
                If False, the dataset will output a tuple (input, label) or ((index_from, index_to), label) directly


            allow_partial_cache (bool): if True, the dataset will allow partial caching of the neural data 
                (if only part of the recording is needed for the dataset); this is useful for large datasets
            
            start_neural_data_before_word_onset (int): the number of samples to start the neural data before each word onset
            end_neural_data_after_word_onset (int): the number of samples to end the neural data after each word onset
            random_seed (int): seed for random operations within this dataset
        """

        # Set up a local random state with the provided seed
        self.rng = np.random.RandomState(random_seed)
        
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.output_indices = output_indices
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset
        self.lite = lite
        self.binary_tasks = True # Neuroprobe always uses binary tasks
        self.n_classes = 0
        self.output_dict = output_dict

        if self.lite:
            lite_electrodes = NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in lite_electrodes if e in subject.electrode_labels]

        eval_name_remapped = eval_name
        if eval_name in single_float_variables_name_remapping: eval_name_remapped = single_float_variables_name_remapping[eval_name]
        if eval_name in four_way_cardinal_directions_name_remapping: eval_name_remapped = four_way_cardinal_directions_name_remapping[eval_name]
        if eval_name in classification_variables_name_remapping: eval_name_remapped = classification_variables_name_remapping[eval_name]
        self.eval_name_remapped = eval_name_remapped

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)

        self.movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"{self.subject.subject_identifier}_{self.trial_id}"]
        
        # Add the original features from braintreebank to the all_words_df
        transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/features.csv')
        original_features_df = pd.read_csv(transcript_file_format.format(self.movie_name)).set_index('Unnamed: 0')
        # Add new columns from words_df using original_index mapping
        new_columns = [col for col in original_features_df.columns if col not in self.all_words_df.columns]
        for col in new_columns:
            self.all_words_df[col] = self.all_words_df['original_index'].map(original_features_df[col])

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
            # Grab the new pitch volume features if they exist
            if self.eval_name_remapped in new_pitch_variables:
                pitch_volume_features_path = os.path.join(PITCH_VOLUME_FEATURES_DIR, f"{self.movie_name}_pitch_volume_features.json")
                with open(pitch_volume_features_path, 'r') as f:
                    raw_pitch_volume_features = json.load(f)

                TARGET_DP_FOR_KEYS = 5  # Standard number of decimal places
                normalized_pvf = {}
                for k_str, v_val in raw_pitch_volume_features.items():
                    k_float = float(k_str)
                    normalized_key = f"{k_float:.{TARGET_DP_FOR_KEYS}f}"
                    normalized_pvf[normalized_key] = v_val
                pitch_volume_features = normalized_pvf

                start_times = self.all_words_df['start'].to_list()

                all_labels = []
                for start_time_val in start_times:
                    lookup_key = f"{start_time_val:.{TARGET_DP_FOR_KEYS}f}"
                    label = pitch_volume_features[lookup_key][self.eval_name_remapped]
                    all_labels.append(label)

                all_labels = np.array(all_labels)
            else:
                all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()


            # Get indices for words in top and bottom quartiles
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            self.extreme_indices = np.where((label_percentiles > 0.75) | (label_percentiles < 0.25))[0]
            self.extreme_labels = torch.from_numpy((label_percentiles[self.extreme_indices] > 0.75).astype(int))
            self.n_samples = len(self.extreme_indices)
            self.n_classes = 2
        elif eval_name in ["onset", "speech"]:
            self.positive_indices = np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.all_words_df))
            self.negative_indices = np.arange(len(self.nonverbal_df))
            min_len = min(len(self.positive_indices), len(self.negative_indices)) # make sure we have an equal number of positive and negative samples
            self.positive_indices = np.sort(self.rng.choice(self.positive_indices, size=min_len, replace=False))
            self.negative_indices = np.sort(self.rng.choice(self.negative_indices, size=min_len, replace=False))
            self.n_samples = len(self.positive_indices) + len(self.negative_indices)
            self.n_classes = 2
        elif eval_name in four_way_cardinal_direction_variables: 
            self.class_labels = np.zeros(len(self.all_words_df), dtype=int)
            angles = self.all_words_df[self.eval_name_remapped].to_numpy()
            cardinal_directions = np.array([0, 90, 180, 270]) if not self.binary_tasks else np.array([0, 180])
            angles_expanded = angles[:, np.newaxis]
            distances = np.minimum(np.abs(angles_expanded - cardinal_directions),
                                360 - np.abs(angles_expanded - cardinal_directions))
            self.class_labels = np.argmin(distances, axis=1)
            rebalance_classes = True
            self.n_classes = len(cardinal_directions)
        elif eval_name == "face_num":
            self.n_samples = len(self.all_words_df)
            self.n_classes = 3 if not self.binary_tasks else 2
            self.class_labels = self.all_words_df["face_num"].to_numpy().astype(int)
            self.class_labels[self.class_labels > self.n_classes-1] = self.n_classes-1 # cap at 2
            rebalance_classes = True
        elif eval_name == "word_index":
            self.n_samples = len(self.all_words_df)
            self.n_classes = 4 if not self.binary_tasks else 2
            self.class_labels = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            self.class_labels[self.class_labels > self.n_classes-1] = self.n_classes-1 # cap at 3
            rebalance_classes = True
        elif eval_name == "word_head_pos":
            self.n_samples = len(self.all_words_df)
            self.class_labels = self.all_words_df[self.eval_name_remapped].to_numpy().astype(int)
            rebalance_classes = True
            self.n_classes = 2
        elif eval_name == "word_part_speech":
            self.n_samples = len(self.all_words_df)
            self.n_classes = 4 if not self.binary_tasks else 2
            self.class_labels = np.ones(len(self.all_words_df)).astype(int) * (self.n_classes - 1)
            for i, pos in enumerate(["VERB", "NOUN", "PRON"][:self.n_classes-1]):
                self.class_labels[self.all_words_df[self.eval_name_remapped] == pos] = i
            rebalance_classes = True                
        elif eval_name == "speaker":
            self.n_samples = len(self.all_words_df)
            self.n_classes = 4 if not self.binary_tasks else 2
            self.class_labels = np.ones(len(self.all_words_df)).astype(int) * (self.n_classes - 1)
            most_frequent_speakers = self.all_words_df['speaker'].value_counts().index
            for i, speaker in enumerate(most_frequent_speakers[:self.n_classes-1]):
                self.class_labels[self.all_words_df['speaker'] == speaker] = i
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
            self.n_classes = 2
        else:
            raise ValueError(f"Invalid eval_name: {eval_name}")

        if rebalance_classes:
            # Get counts for each class
            unique_classes, class_counts = np.unique(self.class_labels, return_counts=True)
            min_count = np.min(class_counts)
            # Create balanced subset by randomly sampling min_count elements from each class
            balanced_indices = []
            for class_label in unique_classes:
                class_indices = np.where(self.class_labels == class_label)[0]
                sampled_indices = self.rng.choice(class_indices, size=min_count, replace=False)
                sampled_indices = np.sort(sampled_indices)
                balanced_indices.extend(sampled_indices)
            
            # Print rebalancing information
            # print(f"Subject {self.subject.subject_id}, Trial {self.trial_id}, Eval {self.eval_name}: Total datapoints before rebalancing: {len(self.class_labels)}, Classes: {', '.join([f'Class {cl}: {ct}' for cl, ct in zip(unique_classes, class_counts)])}, Total after rebalancing: {len(unique_classes) * min_count}")
            
            self.balanced_indices = np.sort(np.array(balanced_indices))
            self.class_labels = self.class_labels[self.balanced_indices]
            self.n_samples = len(self.balanced_indices)
        

        # If lite, then cache only the part of the neural data that is needed for the dataset. This is to save memory.
        # the samples will be the first NEUROPROBE_LITE_MAX_SAMPLES samples of the movie
        self.cache_window_from = None
        self.cache_window_to = None
        if self.lite:
            # if self.n_samples < NEUROPROBE_LITE_MAX_SAMPLES: print(f"WARNING: Subject {self.subject.subject_id}, Trial {self.trial_id}, Eval {self.eval_name}: Not enough samples to create a lite dataset, using all {self.n_samples} samples")
            max_samples = min(NEUROPROBE_LITE_MAX_SAMPLES, self.n_samples)
            self.n_samples = max_samples

            if allow_partial_cache:
                n_try_indices = self.n_classes # try some first and last samples to get a good estimate of the edges of the needed data in the dataset
                window_indices = []
                for i in list(range(n_try_indices))+list(range(self.n_samples-n_try_indices, self.n_samples)):
                    (window_from, window_to), _ = self.__getitem__(i, force_output_indices=True)
                    window_indices.append(window_from)
                    window_indices.append(window_to)
                self.cache_window_from = np.min(window_indices)
                self.cache_window_to = np.max(window_indices)
        

    def _get_neural_data(self, window_from, window_to, force_output_indices=False):
        self.subject.load_neural_data(self.trial_id, cache_window_from=self.cache_window_from, cache_window_to=self.cache_window_to)
        if not self.output_indices and not force_output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            if self.lite:
                input = input[self.electrode_indices_subset]
            return input.to(dtype=self.dtype)
        else:
            return window_from, window_to # just return the window indices

    def _simple_float_variable__getitem__(self, idx, force_output_indices=False):
        word_index = self.extreme_indices[idx]
        row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
        return input, self.extreme_labels[idx].item()

    def _positive_negative_getitem__(self, idx, force_output_indices=False):
        if idx % 2 == 0: # even indices are positive samples
            word_index = self.positive_indices[idx//2]
            row = self.all_words_df.iloc[word_index]
            est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
            est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
            input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
            return input, 1
        else: # odd indices are negative samples
            item_index = self.negative_indices[idx//2]
            row = self.nonverbal_df.iloc[item_index]
            est_idx = int(row['est_idx'])
            est_end_idx = est_idx + self.end_neural_data_after_word_onset + self.start_neural_data_before_word_onset
            input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
            return input, 0
        
    def _classification__getitem__(self, idx, force_output_indices=False):
        word_index = self.balanced_indices[idx]
        row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = int(row['est_idx']) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
        return input, self.class_labels[idx].item()
        
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx, force_output_indices=False):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n_samples}")
        
        if self.eval_name in single_float_variables or self.eval_name == "word_gap":
            input, label = self._simple_float_variable__getitem__(idx, force_output_indices=force_output_indices)
        elif self.eval_name in four_way_cardinal_direction_variables or self.eval_name in ["face_num", "word_index", "word_head_pos", "word_part_speech", "speaker"]:
            input, label = self._classification__getitem__(idx, force_output_indices=force_output_indices)
        elif self.eval_name in ["onset", "speech"]:
            input, label = self._positive_negative_getitem__(idx, force_output_indices=force_output_indices)
        else:
            raise ValueError(f"Invalid eval_name: {self.eval_name}")

        if self.output_dict:
            return {"data": input, "label": label, "electrode_labels": self.subject.electrode_labels}
        else:
            return input, label