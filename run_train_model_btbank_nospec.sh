#!/bin/bash
#SBATCH --job-name=btb_ns          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -t 16:00:00         
#SBATCH --array=1-16
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="btbank1_0,btbank1_1,btbank2_4,btbank2_5,btbank3_1,btbank3_2,btbank7_1,btbank10_1"
eval_subject_trials="btbank1_2,btbank2_6,btbank3_0,btbank7_0,btbank10_0"
random_string_options=("NS2")
symmetric_loss_options=(1)
learning_rate_options=(0.0021 0.003 0.0042 0.006)
future_bin_idx_options=(1 2)
spec_options=(0 1)

wandb_project="btbank_tests"

idx=$((SLURM_ARRAY_TASK_ID-1))
random_string=${random_string_options[$((idx % ${#random_string_options[@]}))]}
symmetric_loss=${symmetric_loss_options[$(((idx / ${#random_string_options[@]}) % ${#symmetric_loss_options[@]}))]}
learning_rate=${learning_rate_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]}) % ${#learning_rate_options[@]}))]}
future_bin_idx=${future_bin_idx_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]} / ${#learning_rate_options[@]}) % ${#future_bin_idx_options[@]}))]}
spec=${spec_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]} / ${#learning_rate_options[@]} / ${#future_bin_idx_options[@]}) % ${#spec_options[@]}))]}

echo "random_string: $random_string"
echo "symmetric_loss: $symmetric_loss"
echo "learning_rate: $learning_rate"
echo "future_bin_idx: $future_bin_idx"
echo "spec: $spec"

python -u train_model.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string \
    --symmetric_loss $symmetric_loss \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate \
    --future_bin_idx $future_bin_idx \
    --spectrogram $spec \
    --n_epochs 100