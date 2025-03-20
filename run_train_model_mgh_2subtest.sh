#!/bin/bash
#SBATCH --job-name=mgh_2         
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=384G
#SBATCH -t 16:00:00         
#SBATCH --array=1-9 
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials_options=(
    "mgh1_0,mgh1_1,mgh1_2,mgh1_3"
    "mgh40_3,mgh40_4"
    "mgh1_0,mgh1_1,mgh1_2,mgh1_3,mgh40_3,mgh40_4"
)
random_string_options=(
    "MGH_2S_1"
    "MGH_2S_2"
    "MGH_2S_3"
)

eval_subject_trials=""
wandb_project="mgh_tests"

idx=$((SLURM_ARRAY_TASK_ID-1))
train_subject_trials=${train_subject_trials_options[$((idx % ${#train_subject_trials_options[@]}))]}
random_string=${random_string_options[$((idx / ${#train_subject_trials_options[@]}))]}

python -u train_model.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials "" \
    --wandb_project $wandb_project