#!/bin/bash
#SBATCH --job-name=btb_jepa          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=196G
#SBATCH -t 6:00:00         
#SBATCH --array=1-144
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="btbank1_0,btbank1_1,btbank2_4,btbank2_5,btbank3_1,btbank3_2,btbank7_1,btbank10_1"
eval_subject_trials="btbank1_2,btbank2_6,btbank3_0,btbank7_0,btbank10_0"
random_string_options=("JEPAM1" "JEPAM2" "JEPAM3")
symmetric_loss_options=(1)
learning_rate_options=(0.0015 0.0021 0.003 0.0042 0.006 0.0084)
momentum_options=(0.94 0.98)
spec_options=(1 0)
future_bin_idx_options=(1 2 3 4)

wandb_project="jepa_tests"

# Calculate indices for two parallel jobs
idx1=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 ))
idx2=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 + 1 ))

n_rs=${#random_string_options[@]}
n_sl=${#symmetric_loss_options[@]}
n_lr=${#learning_rate_options[@]}
n_m=${#momentum_options[@]}
n_sp=${#spec_options[@]}
n_fb=${#future_bin_idx_options[@]}

# Convert indices for first job
random_string1=${random_string_options[$((idx1 % n_rs))]}
symmetric_loss1=${symmetric_loss_options[$(((idx1 / n_rs) % n_sl))]}
learning_rate1=${learning_rate_options[$(((idx1 / (n_rs * n_sl)) % n_lr))]}
momentum1=${momentum_options[$(((idx1 / (n_rs * n_sl * n_lr)) % n_m))]}
spec1=${spec_options[$(((idx1 / (n_rs * n_sl * n_lr * n_m)) % n_sp))]}
future_bin_idx1=${future_bin_idx_options[$(((idx1 / (n_rs * n_sl * n_lr * n_m * n_sp)) % n_fb))]}

# Convert indices for second job
random_string2=${random_string_options[$((idx2 % n_rs))]}
symmetric_loss2=${symmetric_loss_options[$(((idx2 / n_rs) % n_sl))]}
learning_rate2=${learning_rate_options[$(((idx2 / (n_rs * n_sl)) % n_lr))]}
momentum2=${momentum_options[$(((idx2 / (n_rs * n_sl * n_lr)) % n_m))]}
spec2=${spec_options[$(((idx2 / (n_rs * n_sl * n_lr * n_m)) % n_sp))]}
future_bin_idx2=${future_bin_idx_options[$(((idx2 / (n_rs * n_sl * n_lr * n_m * n_sp)) % n_fb))]}

echo "Job 1 - RS: $random_string1, SL: $symmetric_loss1, LR: $learning_rate1, M: $momentum1, SPEC: $spec1, FB: $future_bin_idx1"
echo "Job 2 - RS: $random_string2, SL: $symmetric_loss2, LR: $learning_rate2, M: $momentum2, SPEC: $spec2, FB: $future_bin_idx2"

python -u train_model_jepa.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string1 \
    --symmetric_loss $symmetric_loss1 \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate1 \
    --spectrogram $spec1 \
    --momentum $momentum1 \
    --future_bin_idx $future_bin_idx1 \
    --n_epochs 100 &

python -u train_model_jepa.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string2 \
    --symmetric_loss $symmetric_loss2 \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate2 \
    --spectrogram $spec2 \
    --momentum $momentum2 \
    --future_bin_idx $future_bin_idx2 \
    --n_epochs 100 &

wait