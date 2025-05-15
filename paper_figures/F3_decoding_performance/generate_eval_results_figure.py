import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import matplotlib.font_manager as fm
font_path = '/om2/user/zaho/btbench/figures/font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

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

def create_performance_figure():
    task_name_mapping = {
        'onset': 'Sentence Onset',
        'speech': 'Speech',
        'volume': 'Voice Volume', 
        'delta_volume': 'Delta Volume',
        'pitch': 'Voice Pitch',
        'word_length': 'Word Length',
        'speaker': 'Speaker Identity',
        #'delta_pitch': 'Delta Pitch',
        'gpt2_surprisal': 'GPT-2 Surprisal',
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
    task_list = task_name_mapping

    # optional: filter tasks
    #tasks_filter = ['onset', 'speech', 'volume', 'pitch', 'speaker', 'global_flow', 'local_flow', 'global_flow_angle', 'local_flow_angle']
    #tasks_filter = ['onset', 'speech', 'volume', 'speaker']
    #task_list = {task: task_list[task] for task in tasks_filter}

    nperseg = 256
    ms_per_seg = int(nperseg / 2048 * 1000)

    
    models = ['Linear (raw voltage)', 'Linear (-line noise)',
               f'Linear (FFT - real+imag, {ms_per_seg}ms)', f'Linear (FFT - abs, {ms_per_seg}ms)',
              'BrainBERT', 'PopT (frozen)', 'PopT', 'NCF-1 (frozen)']
    
    
    models = [f'Linear (spectrogram)', f'Linear (FFT)', 'Linear',
               'PopT', 'BrainBERT', 'CNF-1 (frozen)']
    

    popt_models = [model for model in models if model.startswith('PopT')]  # Get the last model which is POPT
    non_popt_models = [model for model in models if not model.startswith('PopT')]
    popt_csv_paths = [f'/om2/user/zaho/btbench/eval_results_popt/popt_{split_type}_results.csv']

    # Define models
    # models = [f'Linear (spectrogram, {ms_per_seg}ms)', 
    #           'BrainBERT (granularity=1)', 'BrainBERT (granularity=4)', 'BrainBERT (granularity=16)', 'BrainBERT (granularity=-1)']
    
    subject_trials = BTBENCH_LITE_SUBJECT_TRIALS
    metric = 'AUROC' # 'AUROC'
    assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

    performance_data = {}
    for task in task_list.keys():
        performance_data[task] = {}
        for model in models:
            performance_data[task][model] = {}

    for model_idx, model in enumerate(non_popt_models): # Andrii's model format
        for task in task_list.keys():
            subject_trial_means = []
            for subject_id, trial_id in subject_trials:
                nperseg_suffix = f'_nperseg{nperseg}' if nperseg != 256 else ''
                if model == ('Linear'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_voltage/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (spectrogram'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_abs{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (FFT)'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_realimag{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('BrainBERT'):
                    granularity = -1
                    filename = f'/om2/user/zaho/BrainBERT/eval_results_lite_{split_type}/brainbert_frozen_mean_granularity_{granularity}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('CNF-1 (frozen)'):
                    filename = f'/om2/user/zaho/bfm/eval_results_lite_{split_type}/M_nst19_dm192_dmb64_nh12_nl2_5_nes50_nf_nUTP_beT_pmt0.9_mtp100.0_SU_eeL_rBBFM_XXXt/frozen_bin_epoch6/population_btbank{subject_id}_{trial_id}_{task}.json'

                    
                if not os.path.exists(filename):
                    print(f"Warning: File {filename} not found, skipping...")
                    continue

                with open(filename, 'r') as json_file:
                    data = json.load(json_file)
                if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']:
                    data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
                else:
                    data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window'] # for BrainBERT only
                value = np.nanmean([fold_result['test_roc_auc'] for fold_result in data['folds']])
                subject_trial_means.append(value)
            performance_data[task][model] = {
                'mean': np.mean(subject_trial_means),
                'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
            }

    import pandas as pd
    # Load and process data for the POPT model
    for popt_model_i, popt_model in enumerate(popt_models):
        popt_csv_path = popt_csv_paths[popt_model_i]
        if os.path.exists(popt_csv_path):
            # Read the CSV file
            popt_data = pd.read_csv(popt_csv_path)
            # Group by subject_id, trial_id, and task_name to calculate mean across folds
            for task in task_list.keys():
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
                        print(f"Warning: No data found for subject {subject_id}, trial {trial_id}, task {task} in POPT results ({popt_csv_path})")
                
                if subject_trial_means:
                    performance_data[task][popt_model] = {
                        'mean': np.mean(subject_trial_means),
                        'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
                    }
                else:
                    performance_data[task][popt_model] = {
                        'mean': np.nan,
                        'sem': np.nan
                    }
        else:
            print(f"Warning: POPT results file {popt_csv_path} not found")
            # Set NaN values for POPT model if file doesn't exist
            for task in task_list.keys():
                performance_data[task][popt_model] = {
                    'mean': np.nan,
                    'sem': np.nan
                }

    # Create figure with 4x5 grid - reduced size
    n_cols = 5
    n_rows = math.ceil((len(task_list)+1)/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8/5*n_cols, 6/4*n_rows+.2))

    # Flatten axs for easier iteration
    axs_flat = axs.flatten()

    # Color palette
    colors = sns.color_palette("husl", 4)

    # Model color palette using two different colormaps
    linear_models = [model for model in models if model.startswith('Linear')]
    brainbert_models = [model for model in models if model.startswith('BrainBERT')]
    popt_model = [model for model in models if model.startswith('PopT')]
    cnf1_model = [model for model in models if model.startswith('CNF-1 (frozen)')]
    
    linear_colors = sns.color_palette("viridis", len(linear_models))
    brainbert_colors = sns.color_palette("plasma", len(brainbert_models))
    popt_colors = sns.color_palette("magma", len(popt_model))
    cnf1_colors = sns.color_palette("viridis", len(cnf1_model))

    linear_colors = sns.color_palette("gray", len(linear_models))
    brainbert_colors = sns.color_palette("plasma", len(brainbert_models))
    popt_colors = sns.color_palette("viridis", len(popt_model))
    cnf1_colors = sns.color_palette("Reds", len(cnf1_model))
    # Combine colors in the correct order
    model_colors = linear_colors + popt_colors + brainbert_colors + cnf1_colors

    # Bar width
    bar_width = 0.2
    
    # Calculate overall performance for each model
    overall_performance = {}
    for model in models:
        means = [performance_data[task][model]['mean'] for task in task_list.keys()]
        sems = [performance_data[task][model]['sem'] for task in task_list.keys()]
        overall_performance[model] = {
            'mean': np.nanmean(means),
            'sem': np.sqrt(np.sum(np.array(sems)**2)) / len(sems)  # Combined SEM
        }
    
    # Plot overall performance in first axis
    first_ax = axs_flat[0]
    for i, model in enumerate(models):
        print(i, model)
        perf = overall_performance[model]
        first_ax.bar(i * bar_width, perf['mean'], bar_width,
                    yerr=perf['sem'],
                    color=model_colors[i],
                    capsize=6)
    
    first_ax.set_title('Overall', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    if metric == 'accuracy':
        first_ax.set_ylim(0.2, 1.0)
    else:
        first_ax.set_ylim(0.495, .6)
        first_ax.set_yticks([0.5, 0.6])
    first_ax.set_xticks([])
    first_ax.set_ylabel(metric)
    first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    first_ax.spines['top'].set_visible(False)
    first_ax.spines['right'].set_visible(False)
    first_ax.tick_params(axis='y')

    # Plot counter - start from 1 for remaining plots
    plot_idx = 1

    for task, chance_level in task_list.items():
        ax = axs_flat[plot_idx]
        
        # Plot bars for each model
        x = np.arange(len(models))
        for i, model in enumerate(models):
            perf = performance_data[task][model]
            ax.bar(i * bar_width, perf['mean'], bar_width,
                    yerr=perf['sem'], 
                    color=model_colors[i],
                    capsize=6)
        
        # Customize plot
        ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
        if metric == 'accuracy':
            ax.set_ylim(0.2, 1.0)
        else:
            ax.set_ylim(0.49, 0.87)
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
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[i]) for i in range(len(models))]
    handle_names = models
    if False:
        handles.append(chance_line)
        handle_names += ["Chance"]
    fig.legend(handles, handle_names,
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.1),
              ncol=4 if len(models) == 3 else 3,
              frameon=False)

    # Adjust layout with space at the bottom for legend
    plt.tight_layout(rect=[0, 0.2 if len(task_list)<10 or len(models)>3 else 0.1, 1, 1], w_pad=0.4)
    
    # Save figure
    path_save = f'paper_figures/F3_decoding_performance/eval_lite_{split_type}{nperseg_suffix}.pdf'
    plt.savefig(path_save, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {path_save}')
    plt.close()

if __name__ == "__main__":
    create_performance_figure()