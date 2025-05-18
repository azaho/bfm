import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob
import math
import matplotlib.font_manager as fm
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create performance figure for BTBench evaluation')
parser.add_argument('--split_type', type=str, default='SS_SM', 
                    help='Split type to use (SS_SM or SS_DM)')
args = parser.parse_args()
split_type = args.split_type

BTBENCH_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]

BTBENCH_LITE_SUBJECT_TRIALS = [(3, 0), (3, 1), (3, 2)]

task_list = {
    'onset': 0.5, 
    'speech': 0.5,
    'volume': 0.5,
    'pitch': 0.5,
    'speaker': 0.25,
    #'delta_volume': 0.5,
    #'delta_pitch': 0.5,
    'gpt2_surprisal': 0.5,
    'word_length': 0.5,
    'word_gap': 0.5,
    'word_index': 0.25,
    'word_head_pos': 0.5,
    'word_part_speech': 0.25,
    #'frame_brightness': 0.5,
    'global_flow': 0.5,
    'local_flow': 0.5,
    #'global_flow_angle': 0.25,
    #'local_flow_angle': 0.25,
    #'face_num': 0.33,
}
task_name_mapping = {
    'overall': 'Mean decoding AUROC (14 tasks)',
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Voice Volume', 
    'pitch': 'Voice Pitch',
    'speaker': 'Speaker Identity',
    #'delta_volume': 'Delta Volume',
    #'delta_pitch': 'Delta Pitch',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_length': 'Word Length',
    'word_gap': 'Inter-word Gap',
    'word_index': 'Word Position',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    #'frame_brightness': 'Frame Brightness',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    #'global_flow_angle': 'Global Flow Angle',
    #'local_flow_angle': 'Local Flow Angle',
    #'face_num': 'Number of Faces',
}

def get_performance_data():
    nperseg = 256
    
    models = [
    f'Linear (voltage traces)',
              'Linear (spectrogram)', 

              'BrainBERT (original)',

            #   'BrainBERT L2 (voltage traces, latent space)',
            #   'BrainBERT L2 (spectrogram, latent space)',
            # all of these are brainbert
              'BrainBERT L2 (voltage traces, raw)',
              'BrainBERT L2 (spectrogram, raw)',

              'BrainBERT contrastive (voltage traces, latent space)',
              'BrainBERT contrastive (spectrogram, latent space)',
              'BrainBERT contrastive (voltage traces, raw)',
              'BrainBERT contrastive (spectrogram, raw)',

              ]

    models = [
              'MSE loss (voltage)',
              'MSE loss (spectrogram)',
              'Contrastive (voltage, latent space)',
              'Contrastive (spectrogram, latent space)',
              'Contrastive (voltage, data space)',
              'Contrastive (spectrogram, data space)',
              ]
    
    popt_models = [model for model in models if model.startswith('PopT')]
    non_popt_models = [model for model in models if not model.startswith('PopT')]
    popt_csv_paths = [f'eval_results_popt/popt_{split_type}_results.csv']
    
    non_popt_model_dirs = [
        #"", "", "", # for the linear models
    #"M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_l2_beT_nII_pmt0.1_SU_eeL_fk256_rBBTT_6",
    #"M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_l2_beT_nII_nSP_pmt0.1_SU_eeL_fk256_rBBTT_6",
    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_l2_beT_nII_nSP_pmt0.1_eeL_fk256_rBBTT_6",
    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_l2_beT_nII_pmt0.1_eeL_fk256_rBBTT_6",

    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_beT_nII_nSP_pmt0.1_SU_eeL_fk256_rBBTT_6",
    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_beT_nII_pmt0.1_SU_eeL_fk256_rBBTT_6",
    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_beT_nII_nSP_pmt0.1_eeL_fk256_rBBTT_6",
    "M_nst1_dm192_dmb192_nh12_nl4_5_nes45_nf_beT_nII_pmt0.1_eeL_fk256_rBBTT_6",
    ]
    
    subject_trials = BTBENCH_LITE_SUBJECT_TRIALS
    metric = 'AUROC'
    
    performance_data = {}
    for task in task_list.keys():
        performance_data[task] = {}
        for model in models:
            performance_data[task][model] = {
                'mean': None,
                'sem': None,
                'all_folds': []  # New field to store all fold values
            }

    # Process non-PopT models
    for model_idx, model in enumerate(non_popt_models):
        for task in task_list.keys():
            all_folds = []  # Store all fold values
            for subject_id, trial_id in subject_trials:
                nperseg_suffix = f'_nperseg{nperseg}' if nperseg != 256 else ''
                if model.startswith('Linear (voltage traces)'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_voltage/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (spectrogram'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_abs{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (FFT)'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_realimag{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('BrainBERT (original)'):
                    granularity = -1
                    filename = f'/om2/user/zaho/BrainBERT/eval_results_lite_{split_type}/brainbert_frozen_mean_granularity_{granularity}/population_btbank{subject_id}_{trial_id}_{task}.json'
                else:
                    model_dir = non_popt_model_dirs[model_idx]
                    epoch=100
                    filename = f'/om2/user/zaho/bfm/eval_results_lite_{split_type}/{model_dir}/frozen_bin_epoch{epoch}/population_btbank{subject_id}_{trial_id}_{task}.json'
                    
                if not os.path.exists(filename):
                    print(f"Warning: File {filename} not found, skipping...")
                    continue

                with open(filename, 'r') as json_file:
                    data = json.load(json_file)
                if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']:
                    data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
                else:
                    data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window']
                values = [fold_result['test_roc_auc'] for fold_result in data['folds']]
                all_folds.extend(values)
            
            performance_data[task][model] = {
                'mean': np.mean(all_folds),
                'sem': np.std(all_folds) / np.sqrt(len(all_folds)),
                'all_folds': all_folds
            }

    # Process PopT models
    for popt_model_i, popt_model in enumerate(popt_models):
        popt_csv_path = popt_csv_paths[popt_model_i]
        if os.path.exists(popt_csv_path):
            popt_data = pd.read_csv(popt_csv_path)
            for task in task_list.keys():
                all_folds = []
                
                for subject_id, trial_id in subject_trials:
                    task_data = popt_data[(popt_data['subject_id'] == subject_id) & 
                                        (popt_data['trial_id'] == trial_id) & 
                                        ((popt_data['task_name'] == task) | (popt_data['task_name'] == task + '_frozen_True'))]
                    
                    if not task_data.empty:
                        all_folds.extend(task_data['test_roc_auc'].values)
                
                if all_folds:
                    performance_data[task][popt_model] = {
                        'mean': np.mean(all_folds),
                        'sem': np.std(all_folds) / np.sqrt(len(all_folds)),
                        'all_folds': all_folds
                    }
                else:
                    performance_data[task][popt_model] = {
                        'mean': np.nan,
                        'sem': np.nan,
                        'all_folds': []
                    }
        else:
            print(f"Warning: POPT results file {popt_csv_path} not found")
            for task in task_list.keys():
                performance_data[task][popt_model] = {
                    'mean': np.nan,
                    'sem': np.nan,
                    'all_folds': []
                }
    
    return performance_data, models, task_list


performance_data, models, task_list = get_performance_data()
print(performance_data)

# Calculate 'overall' task performance by averaging across all folds from all tasks
overall_performance = {}

for model in models:
    all_folds = []
    
    for task in task_list.keys():
        if model in performance_data[task] and performance_data[task][model]['all_folds']:
            all_folds.extend(performance_data[task][model]['all_folds'])
    
    if all_folds:
        overall_performance[model] = {
            'mean': np.mean(all_folds),
            'sem': np.std(all_folds) / np.sqrt(len(all_folds))
        }
    else:
        overall_performance[model] = {
            'mean': np.nan,
            'sem': np.nan
        }

# Add the overall performance to the performance_data dictionary
performance_data['overall'] = overall_performance

print("\nOverall performance across all tasks:")
for model in models:
    print(f"{model}: Mean = {performance_data['overall'][model]['mean']:.4f}, SEM = {performance_data['overall'][model]['sem']:.4f}")


tasks_subset = ['overall', 'speech', 'gpt2_surprisal']
tasks_subset = ['overall']#, 'speech', 'gpt2_surprisal', 'word_part_speech', 'global_flow']

# Generate LaTeX table code
print("\nLaTeX Table Code:")

# Start the LaTeX table
latex_table = "\\begin{table}[h]\n"
latex_table += "\\centering\n"
latex_table += "\\begin{tabular}{l" + "c" * len(tasks_subset) + "}\n"
latex_table += "\\toprule\n"

# Add header row
latex_table += "Training objective & " + " & ".join(task_name_mapping[task] for task in tasks_subset) + " \\\\\n"
latex_table += "\\midrule\n"

# Find the best model for each task
best_models = {}
for task in tasks_subset:
    best_mean = -float('inf')
    best_model = None
    for model in models:
        if (task in performance_data and model in performance_data[task] and 
            not np.isnan(performance_data[task][model]['mean'])):
            if performance_data[task][model]['mean'] > best_mean:
                best_mean = performance_data[task][model]['mean']
                best_model = model
    best_models[task] = best_model

# Add data rows
for i, model in enumerate(models):
    row = f"{model}"
    for task in tasks_subset:
        if task in performance_data and model in performance_data[task]:
            mean = performance_data[task][model]['mean']
            sem = performance_data[task][model]['sem']
            if not np.isnan(mean) and not np.isnan(sem):
                if model == best_models[task]:
                    row += f" & \\textbf{{{mean:.3f}}} $\\pm$ {sem:.3f}"
                else:
                    row += f" & {mean:.3f} $\\pm$ {sem:.3f}"
            else:
                row += " & -"
        else:
            row += " & -"
    row += " \\\\"
    latex_table += row + "\n"
    
    # Add horizontal lines after specific indices
    if i in [1, 3]:
        latex_table += "\\midrule\n"

# End the LaTeX table
latex_table += "\\bottomrule\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\caption{Performance comparison across different models and tasks (mean $\\pm$ SEM).}\n"
latex_table += "\\label{tab:model_performance}\n"
latex_table += "\\end{table}"

print(latex_table)
