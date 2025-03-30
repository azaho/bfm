import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import matplotlib.font_manager as fm
import time
font_path = 'analyses/font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

task_name_mapping = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Audio Volume', 
    'pitch': 'Audio Pitch',
    'speaker': 'Speaker Identity',
    'delta_volume': 'Delta Volume',
    'delta_pitch': 'Delta Pitch',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_length': 'Word Length',
    'word_gap': 'Inter-word Gap',
    'word_index': 'Word Position',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    'frame_brightness': 'Frame Brightness',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'global_flow_angle': 'Global Flow Angle',
    'local_flow_angle': 'Local Flow Angle',
    'face_num': 'Number of Faces',
}
task_list = list(task_name_mapping.keys())

# optional subsetting
task_list = ['onset', 'speech', 'volume', 'speaker']

models = ['Linear (raw voltage)', 'Linear (spectrogram)']

subject_trials = [(1, 2), (2, 6), (3, 0), (10, 0), (7, 0)] # the eval_subject_trials from the training script
#subject_trials += [(1, 0), (1, 1), (2, 4), (2, 5), (3, 1), (3, 2), (7, 1), (10, 1)] # the train_subject_trials from the training script
#subject_trials += [(2, 0), (2, 1), (2, 2), (2, 4)] # unused trials from subject 2

metric = 'AUROC' # 'AUROC'
assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

for model_type, model_name in zip(['voltage', 'spectrogram'], models):
    for task in task_list:
        results = {
            "model_name": model_name,
            "description": f"Linear regression on {model_type} of the neural data.",
            "author": "Andrii Zahorodnii",
            "organization": "MIT",
            "organization_url": "https://azaho.org",
            "timestamp": time.time(),
            "evaluation_results": {}
        }
        
        for subject_id, trial_id in subject_trials:
            subject_trial_key = f"btbank{subject_id}_{trial_id}"
            filename = f'../btbench/eval_results_ss_sm/linear_{model_type}_normalized_subject{subject_id}_trial{trial_id}_{task}.json'
        
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, saying None...")
                results["evaluation_results"][subject_trial_key] = {
                    "population": {
                        "one_second_after_onset": {
                                "time_bin_start": 0.0,
                                "time_bin_end": 1.0,
                                "folds": [
                                    {
                                        "test_accuracy": None,
                                        "test_roc_auc": None
                                    }
                                    for _ in range(5)
                                ]
                        }
                    }
                }
                continue
                
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                
            results["evaluation_results"][subject_trial_key] = {
                "population": {
                    "one_second_after_onset": {
                        "time_bin_start": 0.0,
                        "time_bin_end": 1.0,
                        "folds": [
                            {
                                #"train_accuracy": None,
                                #"train_roc_auc": None,
                                "test_accuracy": None if math.isnan(fold_result["accuracy"]) else fold_result["accuracy"],
                                "test_roc_auc": None if math.isnan(fold_result["auroc"]) else fold_result["auroc"]
                            }
                            for fold_result in data["fold_results"]
                        ]
                    }
                }
            }
            
        # Save combined results for this task
        output_filename = f"analyses/eval_results/linear_{model_type}/population_{task}.json"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)