import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import matplotlib as mpl
from evaluation.neuroprobe import config as neuroprobe_config


mpl.rcParams.update({
    'font.size': 14,         
    'axes.titlesize': 16,    
    'axes.labelsize': 14,    
    'xtick.labelsize': 12,   
    'ytick.labelsize': 12,   
})

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True,
                    help='Task name (e.g., global_flow, speech, onset, etc., or all_tasks_avg)')
parser.add_argument('--epoch', type=int, default=40, help='Model epoch to use (default: 40)')
parser.add_argument('--base_dir', type=str, default='/runs/data/eval_results_frozen_features_',
                    help='Base directory prefix before split type (default: /runs/data/eval_results_frozen_features_)')
parser.add_argument('--output_dir', type=str, default='/analyses/figures/log_reg_matrices/',
                    help='Output directory to save plots (default: /analyses/figures/log_reg_matrices/)')
args = parser.parse_args()

task = args.task
epoch = args.epoch
base_dir = args.base_dir
output_dir = args.output_dir

subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
subject_boundaries = []
current_subject = subject_trials[0][0]
for i, (subj, _) in enumerate(subject_trials):
    if subj != current_subject:
        subject_boundaries.append(i)
        current_subject = subj

split_dirs = {
    split: os.path.join(f"{base_dir}{split}", f"model_epoch{epoch}")
    for split in ['SS_SM', 'SS_DM', 'DS_SM', 'DS_DM']
}


pair_idx = {pair: i for i, pair in enumerate(subject_trials)}
n = len(subject_trials)
labels = [f'S{sid}T{tid}' for sid, tid in subject_trials]

# === Task handling ===
if task == "all_tasks_avg":
    all_tasks = neuroprobe_config.NEUROPROBE_TASKS  # e.g., ['onset', 'speech', 'global_flow', ...]
    task_matrices = []

    for task_name in all_tasks:
        auroc_matrix = np.full((n, n), np.nan)

        for i, (train_sub, train_trial) in enumerate(subject_trials):
            for j, (test_sub, test_trial) in enumerate(subject_trials):
                if train_sub == test_sub and train_trial == test_trial:
                    split = 'SS_SM'
                elif train_sub == test_sub:
                    split = 'SS_DM'
                elif train_trial == test_trial:
                    split = 'DS_SM'
                else:
                    split = 'DS_DM'

                json_path = os.path.join(
                    split_dirs[split],
                    f'population_btbank{test_sub}_{test_trial}_{task_name}.json'
                )

                if not os.path.exists(json_path):
                    print(f"Missing: {json_path}")
                    continue

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    folds = data['evaluation_results'][f'btbank{test_sub}_{test_trial}']['population']['one_second_after_onset']['folds']

                    if split == 'SS_SM':
                        if folds:
                            mean_auroc = np.mean([fold['test_roc_auc'] for fold in folds])
                            auroc_matrix[i, j] = mean_auroc
                    else:
                        matching_folds = [
                            fold for fold in folds
                            if fold.get('train_subject_id') == train_sub and fold.get('train_trial_id') == train_trial
                        ]
                        if matching_folds:
                            mean_auroc = np.mean([fold['test_roc_auc'] for fold in matching_folds])
                            auroc_matrix[i, j] = mean_auroc
                except Exception as e:
                    print(f"Error in {json_path}: {e}")

        task_matrices.append(auroc_matrix)

    # Compute average AUROC matrix
    stacked = np.stack([m for m in task_matrices if m is not None])
    avg_matrix = np.nanmean(stacked, axis=0)
    df = pd.DataFrame(avg_matrix, index=labels, columns=labels)

    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Mean AUROC'}, annot_kws={"size": 12})
    plt.title(f'Average AUROC Matrix over all tasks')
    plt.xlabel('Test (Subject, Trial)')
    plt.ylabel('Train (Subject, Trial)')
    for boundary in subject_boundaries:
        plt.axhline(boundary, color='white', linewidth=1.5)
        plt.axvline(boundary, color='white', linewidth=1.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'auroc_matrix_all_tasks_avg_epoch{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved average AUROC matrix to {save_path}')

else:
    # === Single task flow (same as before) ===
    auroc_matrix = np.full((n, n), np.nan)

    for i, (train_sub, train_trial) in enumerate(subject_trials):
        for j, (test_sub, test_trial) in enumerate(subject_trials):
            if train_sub == test_sub and train_trial == test_trial:
                split = 'SS_SM'
            elif train_sub == test_sub:
                split = 'SS_DM'
            elif train_trial == test_trial:
                split = 'DS_SM'
            else:
                split = 'DS_DM'

            json_path = os.path.join(
                split_dirs[split],
                f'population_btbank{test_sub}_{test_trial}_{task}.json'
            )

            if not os.path.exists(json_path):
                print(f"File {json_path} does not exist")
                continue

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                folds = data['evaluation_results'][f'btbank{test_sub}_{test_trial}']['population']['one_second_after_onset']['folds']

                if split == 'SS_SM':
                    if folds:
                        mean_auroc = np.mean([fold['test_roc_auc'] for fold in folds])
                        auroc_matrix[i, j] = mean_auroc
                else:
                    matching_folds = [
                        fold for fold in folds
                        if fold.get('train_subject_id') == train_sub and fold.get('train_trial_id') == train_trial
                    ]
                    if matching_folds:
                        mean_auroc = np.mean([fold['test_roc_auc'] for fold in matching_folds])
                        auroc_matrix[i, j] = mean_auroc
            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    df = pd.DataFrame(auroc_matrix, index=labels, columns=labels)

    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'AUROC'}, annot_kws={"size": 12})
    plt.title(f'AUROC Matrix for task: {task}')
    plt.xlabel('Test (Subject, Trial)')
    plt.ylabel('Train (Subject, Trial)')
    for boundary in subject_boundaries:
        plt.axhline(boundary, color='white', linewidth=1.5)
        plt.axvline(boundary, color='white', linewidth=1.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'auroc_matrix_{task}_epoch{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved AUROC matrix to {save_path}')
