import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import pandas as pd
import evaluation.neuroprobe.config as neuroprobe_config

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
n_fig_legend_cols = 3 if figure_size_multiplier<1.8 else 4

### DEFINE MODELS ###

# assert split_type == 'SS_DM', 'Split type must be SS_DM'

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
    {
        'name': 'Linear (laplacian+spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_laplacian-stft_abs_nperseg512_poverlap0.75_maxfreq150/'
    },
    # {
    #     'name': 'Population Transformer',
    #     'color_palette': 'viridis', 
    #     'eval_results_path': f'/om2/user/zaho/PopTCameraReadyPrep/outputs/neuroprobe_popt_lite/eval_results_{split_type}/'
    # },
    {
        'name': 'BrainBERT (untrained)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_randomly_initialized_keepall/'
    },
    {
        'name': 'BrainBERT',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_keepall/'
    },
] 
# \ + [
#     {
#         'name': f'Andrii0 epoch {model_epoch} ({feature_type})',
#         'color_palette': 'rainbow',
#         'eval_results_path': f'runs/data/andrii0_lr0.003_wd0.0_dr0.2_rR2_t20250905_141853/model_epochSCUFFED_FT/results/',
#         'pad_x': 1 if model_epoch==0 else 0,
#     } for feature_type in ['cls'] for model_epoch in [10] # , 'meanE', 'cls', 'meanT', 'meanT_meanE', 'meanT_cls'
# ] + [
    # {
    #     'name': f'MSE autoreg. epoch {model_epoch} ({feature_type})',
    #     'color_palette': 'rainbow',
    #     'eval_results_path': f'runs/data/mse_ar_lr0.003_wd0.0_dr0.2_rR2_t20250905_141854/model_epochSCUFFED_FT/results/',
    #     'pad_x': 1 if model_epoch==0 else 0,
    # } for feature_type in ['cls'] for model_epoch in [10] # , 'meanE', 'cls', 'meanT', 'meanT_meanE', 'meanT_cls'
# ] + [
#     {
#         'name': f'MSE random masking epoch {model_epoch} ({feature_type})',
#         'color_palette': 'rainbow',
#         'eval_results_path': f'runs/data/mse_rm_lr0.003_wd0.0_dr0.2_rR2_t20250905_141848/model_epochSCUFFED_FT/results/',
#         'pad_x': 1 if model_epoch==0 else 0,
#     } for feature_type in ['cls'] for model_epoch in [10] # , 'meanE', 'cls', 'meanT', 'meanT_meanE', 'meanT_cls'
# ] + [
#     {
#         'name': f'MSE-MtM epoch {model_epoch} ({feature_type})',
#         'color_palette': 'rainbow',
#         'eval_results_path': f'runs/data/mse_mtm_lr0.003_wd0.0_dr0.2_rR2_t20250905_141646/model_epochSCUFFED_FT/results/',
#         'pad_x': 1 if model_epoch==0 else 0,
#     } for feature_type in ['cls'] for model_epoch in [10] # , 'meanE', 'cls', 'meanT', 'meanT_meanE', 'meanT_cls'
# ] + [
#     # {
#     #     'name': f'Andrii BB {bb_model_name} {model_epoch} ({feature_type})',
#     #     'color_palette': 'rainbow',
#     #     'eval_results_path': f'runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_{split_type}/linear_{bb_model_dir}_epoch{model_epoch}_{feature_type}/',
#     #     'pad_x': 1 if model_epoch==0 else 0,
#     # } for bb_model_name, bb_model_dir in {
#     #     "default": "andrii_brainbert_lr0.003_wd0.0_dr0.2_rR2_t20250716_001553",
#     #     "slr": "andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_SLR_t20250719_173751",
#     #     "czw": "andrii_brainbert_lr0.003_wd0.0_dr0.2_rR_CZWPARAMS3_t20250719_173741",
#     #     "czw_slr": "andrii_brainbert_lr0.0003_wd0.0_dr0.2_rR_CZWPARAMS3SLR_t20250719_173743"
#     # }.items() for feature_type in ['keepall'] for model_epoch in [0, 10, 15, 30]
# ]

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

subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
if split_type == 'DS_DM':
    subject_trials = [(s, t) for s, t in subject_trials if s != neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID]

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
rect_y = (0.11 + 0.05 * (math.ceil((len(models)+1)/n_fig_legend_cols)-1)) / figure_size_multiplier
plt.subplots_adjust(bottom=rect_y)

# Save figure
save_path = f'analyses/andrii/25_07_14_andrii0_evals/figures/neuroprobe_eval_lite_{split_type}.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')

performance_data['overall'] = overall_performance
print(performance_data)
filename = f'analyses/andrii/25_07_14_andrii0_evals/figures/neuroprobe_eval_lite_{split_type}.json' 
with open(filename, 'w') as f:
    json.dump(performance_data, f)
print(f'Saved performance data to {filename}')

plt.close()
