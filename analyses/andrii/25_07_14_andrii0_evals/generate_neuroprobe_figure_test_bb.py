import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import pandas as pd
# from evaluation.neuroprobe import config as neuroprobe_config

### PARSE ARGUMENTS ###

import argparse
parser = argparse.ArgumentParser(description='Create performance figure for BTBench evaluation')
parser.add_argument('--split_type', type=str, default='SS_DM', 
                    help='Split type to use (SS_SM or SS_DM or DS_DM)')
args = parser.parse_args()
split_type = args.split_type

metric = 'AUROC' # 'AUROC'
assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

separate_overall_yscale = True # Whether to have the "Task Mean" figure panel have a 0.5-0.6 ylim instead of 0.5-0.9 (used to better see the difference between models)
overall_axis_ylim = (0.4925, 0.75) if separate_overall_yscale else (0.48, 0.95)
other_axis_ylim = (0.48, 0.95)

figure_size_multiplier = 1.8
n_fig_legend_cols = 3 #if figure_size_multiplier<1.8 else 4

### DEFINE MODELS ###

models = [
    {
        'name': 'Linear',
        'color_palette': 'viridis',
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_voltage/'
    },
    {
        'name': 'Linear (spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_stft_abs_nperseg512_poverlap0.75_maxfreq150/'
    },
] + [
#     {
#         'name': f'Andrii0 epoch {model_epoch} ({feature_type})',
#         'color_palette': 'rainbow',
#         'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_andrii0_lr0.003_wd0.001_dr0.0_rR1_t20250714_121055_epoch{model_epoch}_{feature_type}/',
#         'pad_x': 1 if model_epoch==0 else 0,
#     } for feature_type in ['keepall', 'meanE', 'cls', 'meanT', 'meanT_meanE', 'meanT_cls'] for model_epoch in [0, 1, 10, 40]
# ] + [
    {
        'name': f'Andrii0 epoch {model_epoch} ({feature_type})',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_andrii0_lr0.003_wd0.0_dr0.2_rR1_t20250714_121055_epoch{model_epoch}_{feature_type}/',
        'pad_x': 1 if model_epoch==0 else 0,
    } for feature_type in ['keepall'] for model_epoch in [0, 1, 10, 20]
] + [
    {
        'name': f'Andrii_BrainBERT epoch {model_epoch} ({feature_type})',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_{"andrii_brainbert_lr0.003_wd0.0_dr0.2_rR2_t20250716_001553"}_epoch{model_epoch}_{feature_type}/',
        'pad_x': 1 if model_epoch==0 else 0,
    } for feature_type in ['keepall'] for model_epoch in [0, 1, 10, 20]
] + [
    {
        'name': f'CZW_BrainBERT random init (keepall)',
        'color_palette': 'rainbow',
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_randomly_initialized_keepall/',
        'pad_x': 1
    },
    {
        'name': f'CZW_BrainBERT (keepall)',
        'color_palette': 'rainbow',
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_keepall/',
        'pad_x': 0
    },
] + [
    {
        'name': f'Andrii BB CZW params epoch {model_epoch} ({feature_type})',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_{"andrii_brainbert_lr0.003_wd0.0_dr0.2_rR_CZWPARAMS3_t20250719_173741"}_epoch{model_epoch}_{feature_type}/',
        'pad_x': 1 if model_epoch==0 else 0,
    } for feature_type in ['keepall'] for model_epoch in [0, 10, 15, 30]
] + [
    {
        'name': f'Andrii BB CZW SLR params epoch {model_epoch} ({feature_type})',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_{"andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_CZWPARAMS3SLR_t20250719_173743"}_epoch{model_epoch}_{feature_type}/',
        'pad_x': 1 if model_epoch==0 else 0,
    } for feature_type in ['keepall'] for model_epoch in [0, 10, 15, 30]
] + [
    {
        'name': f'Andrii BB SLR params epoch {model_epoch} ({feature_type})',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_{"andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_SLR_t20250719_173751"}_epoch{model_epoch}_{feature_type}/',
        'pad_x': 1 if model_epoch==0 else 0,
    } for feature_type in ['keepall'] for model_epoch in [0, 10, 15, 30]
] + [
    {
        'name': f'Inputs to BB CZW params',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_inputs_{"andrii_brainbert_lr0.003_wd0.0_dr0.2_rR_CZWPARAMS3_t20250719_173741"}_epoch0_keepall/',
        'pad_x': 1,
    },
    {
        'name': f'Inputs to BB SLR params',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_inputs_{"andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_SLR_t20250719_173751"}_epoch0_keepall/',
        'pad_x': 1,
    },{
        'name': f'Inputs to BB CZW params (no otf)',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_inputs_no_otf_{"andrii_brainbert_lr0.003_wd0.0_dr0.2_rR_CZWPARAMS3_t20250719_173741"}_epoch0_keepall/',
        'pad_x': 1,
    },
    {
        'name': f'Inputs to BB SLR params (no otf)',
        'color_palette': 'rainbow',
        'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_inputs_no_otf_{"andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_SLR_t20250719_173751"}_epoch0_keepall/',
        'pad_x': 1,
    }
    
]

### DEFINE TASK NAME MAPPING ###

task_name_mapping = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'pitch': 'Voice Pitch',

    'delta_volume': 'Delta Volume',
    'word_index': 'Word Position',
    'word_gap': 'Inter-word Gap',
    'word_length': 'Word Length',

    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    'speaker': 'Speaker Identity',

    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'frame_brightness': 'Frame Brightness',
    'face_num': 'Number of Faces',
    
    # 'delta_pitch': 'Delta Pitch',
    # 'global_flow_angle': 'Global Flow Angle',
    # 'local_flow_angle': 'Local Flow Angle',
}

subject_trials = [
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]
if split_type == 'DS_DM':
    subject_trials = [(s, t) for s, t in subject_trials if s != 2]

### DEFINE RESULT PARSING FUNCTIONS ###

performance_data = {}
for task in task_name_mapping.keys():
    performance_data[task] = {}
    for model in models:
        performance_data[task][model['name']] = {}

def parse_results_default(model):
    for task in task_name_mapping.keys():
        subject_trial_means = []
        for subject_id, trial_id in subject_trials:
            filename = model['eval_results_path'] + f'population_btbank{subject_id}_{trial_id}_{task}.json'
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping...")
                continue

            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            
            if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']: # XXX remove this later, have a unified interface for all models
                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
            else:
                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window'] # for BrainBERT only
            value = np.nanmean([fold_result['test_roc_auc'] for fold_result in data['folds']])
            subject_trial_means.append(value)

        performance_data[task][model['name']] = {
            'mean': np.mean(subject_trial_means),
            'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
        }
for model in models:
    model['parse_results_function'] = parse_results_default

def parse_results_hara(model):
    for task in task_name_mapping.keys():
        subject_trial_means = []
        for subject_id, trial_id in subject_trials:
            pattern = f'/om2/user/hmor/btbench/eval_results_ds_dt_lite_desikan_killiany/DS-DT-FixedTrain-Lite_{task}_test_S{subject_id}T{trial_id}_*.json'
            matching_files = glob.glob(pattern)
            if matching_files:
                filename = matching_files[0]  # Take the first matching file
            else:
                print(f"Warning: No matching file found for pattern {pattern}, skipping...")
            
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            data = data['final_auroc']
            subject_trial_means.append(data)
        performance_data[task][model['name']] = {
            'mean': np.mean(subject_trial_means),
            'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
        }
if split_type == 'DS_DM': # XXX remove this later, have a unified interface for all models
    models[0]['parse_results_function'] = parse_results_hara

def parse_results_popt(model):
    # Read the CSV file
    popt_data = pd.read_csv(model['eval_results_path'])
    # Group by subject_id, trial_id, and task_name to calculate mean across folds
    for task in task_name_mapping.keys():
        subject_trial_means = []
        
        for subject_id, trial_id in subject_trials:
            # Filter data for current subject, trial, and task
            task_data = popt_data[(popt_data['subject_id'] == subject_id) & 
                                (popt_data['trial_id'] == trial_id) & 
                                ((popt_data['task_name'] == task) | (popt_data['task_name'] == task + '_frozen_True'))]
            
            if not task_data.empty:
                # Calculate mean ROC AUC across folds
                value = task_data['test_roc_auc'].mean()
                subject_trial_means.append(value)
            else:
                print(f"Warning: No data found for subject {subject_id}, trial {trial_id}, task {task} in POPT results ({model['eval_results_path']})")
        
        if subject_trial_means:
            performance_data[task][model['name']] = {
                'mean': np.mean(subject_trial_means),
                'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
            }
        else:
            performance_data[task][model['name']] = {
                'mean': np.nan,
                'sem': np.nan
            }
for model in models:
    if 'PopT' in model['name']:
        model['parse_results_function'] = parse_results_popt

for model in models:
    model['parse_results_function'](model)
    
### CALCULATE OVERALL PERFORMANCE ###

overall_performance = {}
for model in models:
    means = [performance_data[task][model['name']]['mean'] for task in task_name_mapping.keys()]
    sems = [performance_data[task][model['name']]['sem'] for task in task_name_mapping.keys()]
    overall_performance[model['name']] = {
        'mean': np.nanmean(means),
        'sem': np.sqrt(np.sum(np.array(sems)**2)) / len(sems)  # Combined SEM
    }

### PREPARING FOR PLOTTING ###

# Add Arial font
import matplotlib.font_manager as fm
font_path = 'analyses/font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

# Assign colors to models based on color palette
color_palette_ids = {}
for model in models:
    if model['color_palette'] not in color_palette_ids: color_palette_ids[model['color_palette']] = 0
    model['color_palette_id'] = color_palette_ids[model['color_palette']]
    color_palette_ids[model['color_palette']] += 1
for model in models:
    model['color'] = sns.color_palette(model['color_palette'], color_palette_ids[model['color_palette']])[model['color_palette_id']]

# Assign model x-positions
current_x_pos = 0
for i, model in enumerate(models):
    if model.get('pad_x', 0): current_x_pos += model['pad_x']
    model['x_pos'] = current_x_pos
    current_x_pos += 1

### PLOT STUFF ###

# Create figure with modified grid layout using GridSpec
import matplotlib.gridspec as gridspec

n_cols = 4
overall_height = 1.2  # Height of overall axis
margin_height = -0.05   # Margin between overall and task plots
task_rows = math.ceil(len(task_name_mapping)/n_cols)

# Create height ratios: [overall_height, margin_height, task_row_1, task_row_2, ...]
height_ratios = [overall_height, margin_height] + [1.0] * task_rows
n_rows = len(height_ratios)

fig = plt.figure(figsize=(figure_size_multiplier*8/5*n_cols, figure_size_multiplier*6/4*n_rows+.6 * len(models) / n_fig_legend_cols/3/2))
gs = gridspec.GridSpec(n_rows, n_cols, height_ratios=height_ratios, hspace=0.3, wspace=0.2)

# Bar width
bar_width = 0.2

# Plot overall performance spanning entire first row
first_ax = fig.add_subplot(gs[0, :])
for i, model in enumerate(models):
    perf = overall_performance[model['name']]
    first_ax.bar(model['x_pos']*bar_width, perf['mean'], bar_width,
                yerr=perf['sem'],
                color=model['color'],
                capsize=6)

first_ax.set_title('Task Mean', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
first_ax.set_ylim(overall_axis_ylim)
first_ax.set_yticks(np.arange(0.5, overall_axis_ylim[1], 0.1))
first_ax.set_xticks([])
first_ax.set_ylabel(metric)
first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
first_ax.spines['top'].set_visible(False)
first_ax.spines['right'].set_visible(False)
first_ax.tick_params(axis='y')

# Plot counter - start from 0 for task plots in remaining rows
plot_idx = 0

for task, chance_level in task_name_mapping.items():
    # Calculate row and column for current task (starting after overall axis and margin)
    row = plot_idx // n_cols + 2  # Start from row 2 (0=overall, 1=margin, 2+=tasks)
    col = plot_idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    
    # Plot bars for each model
    x = np.arange(len(models))
    for i, model in enumerate(models):
        perf = performance_data[task][model['name']]
        ax.bar(model['x_pos']*bar_width, perf['mean'], bar_width,
                yerr=perf['sem'], 
                color=model['color'],
                capsize=6/(models[-1]['x_pos']+1) * 10)
    
    # Customize plot
    ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
    ax.set_ylim(other_axis_ylim)
    ax.set_yticks(np.arange(0.5, other_axis_ylim[1], 0.1))
    ax.set_xticks([])
    if col == 0:  # Left-most plots
        ax.set_ylabel("AUROC")

    # Add horizontal line at chance level
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make tick labels smaller
    ax.tick_params(axis='y')
    
    plot_idx += 1

# Create a proxy artist for the chance line with the correct style
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)

# Add legend at the bottom with custom handles
handles = [plt.Rectangle((0,0),1,1, color=model['color']) for model in models]
handles.append(chance_line)
fig.legend(handles, [model['name'] for model in models] + ["Chance"],
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.1),
            ncol=n_fig_legend_cols,
            frameon=False)

# Adjust layout with space at the bottom for legend
rect_y = (0.17 + 0.05 * (math.ceil((len(models)+1)/n_fig_legend_cols)-1)) / figure_size_multiplier
plt.subplots_adjust(bottom=rect_y)

# Save figure
save_path = f'analyses/andrii/25_07_14_andrii0_evals/figures/neuroprobe_eval_lite_{split_type}_test.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
plt.close()
