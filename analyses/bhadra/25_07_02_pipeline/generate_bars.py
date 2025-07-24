import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from collections import defaultdict
from evaluation.neuroprobe import config as neuroprobe_config


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--primary_title', type=str, required=True)
parser.add_argument('--primary_epochs', nargs='+', type=int, required=True)
parser.add_argument('--primary_eval_results_path', type=str, required=True)
parser.add_argument('--comparison_models_json', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--split_type', type=str, default='SS_DM', 
                    help='Split type to use (SS_SM or SS_DM or DS_DM)')
args = parser.parse_args()

# Load config values
primary_title = args.primary_title
primary_epochs = args.primary_epochs
comparison_models = json.loads(args.comparison_models_json)
save_dir = args.save_dir
subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
split_type = args.split_type

metric = 'AUROC' # 'AUROC'
assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

separate_overall_yscale = True # Whether to have the "Task Mean" figure panel have a 0.5-0.6 ylim instead of 0.5-0.9 (used to better see the difference between models)
n_fig_legend_cols = 3

### Define primary model
primary_model = {
    'name': primary_title,
    'color_palette': 'viridis',
    'eval_results_path': args.primary_eval_results_path,
}

models = comparison_models.copy()

if split_type == "DS_SM":
    print("No support for PopT DS_SM split")
    models = [m for m in models if "PopT" not in m["name"]]

if split_type == "DS_DM":
    print("No support for PopT frozen DS_DM split")
    models = [m for m in models if "PopT (frozen)" not in m["name"]]

for epoch in primary_epochs:
    new_model = primary_model.copy()
    new_model['name'] = f"{primary_title} (epoch {epoch})"
    new_model["eval_results_path"] = primary_model["eval_results_path"].replace("{epoch}", str(epoch))
    models.append(new_model)

### DEFINE TASK NAME MAPPING ###

task_name_mapping = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'pitch': 'Voice Pitch',
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

### DEFINE RESULT PARSING FUNCTIONS ###

performance_data = {}
for task in task_name_mapping.keys():
    performance_data[task] = {}
    for model in models:
        model["eval_results_path"] = model["eval_results_path"].replace("{split_type}", split_type)
        performance_data[task][model['name']] = {}

    performance_data[task][primary_title] = {}

def parse_results_default(model):
    for task in task_name_mapping.keys():
        subject_trial_means = []
        for subject_id, trial_id in subject_trials:
            filename = model['eval_results_path'] + f'population_btbank{subject_id}_{trial_id}_{task}.json'
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found for split type {split_type} (may not be needed), skipping...")
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

def parse_results_popt(model):
    # Read the CSV file
    if not os.path.exists(model['eval_results_path']):
        print(f"Warning: File {model['eval_results_path']} not found, skipping...")
        return
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
        parse_results_popt(model)
    else:
        parse_results_default(model)

print("Done parsing all model results. Ready for plotting.")

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
font_path = 'assets/font_arial.ttf'
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

### PLOT STUFF ###

# Create figure with 4x5 grid - reduced size
n_cols = 5
n_rows = math.ceil((len(task_name_mapping)+1)/n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(8/5*n_cols, 6/4*n_rows+.6 * len(models) / n_fig_legend_cols/3/2))

# Flatten axs for easier iteration
axs_flat = axs.flatten()

# Bar width
bar_width = 0.2

# Plot overall performance in first axis
first_ax = axs_flat[0]
for i, model in enumerate(models):
    perf = overall_performance[model['name']]
    first_ax.bar(i * bar_width, perf['mean'], bar_width,
                yerr=perf['sem'],
                color=model['color'],
                capsize=6)

first_ax.set_title('Task Mean', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
if metric == 'accuracy':
    first_ax.set_ylim(0.2, 1.0)
else:
    if separate_overall_yscale:
        first_ax.set_ylim(0.4925, 0.65)
        first_ax.set_yticks([0.5, 0.6])
    else:
        first_ax.set_ylim(0.48, 0.87)
        first_ax.set_yticks([0.5, 0.6, 0.7, 0.8])
first_ax.set_xticks([])
first_ax.set_ylabel(metric)
first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
first_ax.spines['top'].set_visible(False)
first_ax.spines['right'].set_visible(False)
first_ax.tick_params(axis='y')

# Plot counter - start from 1 for remaining plots
plot_idx = 1

for task, chance_level in task_name_mapping.items():
    ax = axs_flat[plot_idx]
    
    # Plot bars for each model
    x = np.arange(len(models))
    for i, model in enumerate(models):
        perf = performance_data[task][model['name']]
        ax.bar(i * bar_width, perf['mean'], bar_width,
                yerr=perf['sem'], 
                color=model['color'],
                capsize=6)
    
    # Customize plot
    ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
    if metric == 'accuracy':
        ax.set_ylim(0.2, 1.0)
    else:
        ax.set_ylim(0.48, 0.87)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xticks([])
    if (plot_idx % 5 == 0):  # Left-most plots
        ax.set_ylabel(metric)

    # Add horizontal line at chance level
    if metric == 'AUROC':
        chance_level = 0.5
    ax.axhline(y=chance_level, color='black', linestyle='--', alpha=0.5)
    
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
plt.tight_layout(rect=[0, 0.2 if len(task_name_mapping)<10 or len(models)>3 else 0.1, 1, 1], w_pad=0.4)

# Save figure
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{primary_title}_{split_type}.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
plt.close()
