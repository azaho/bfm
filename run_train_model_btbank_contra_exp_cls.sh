#!/bin/bash
#SBATCH --job-name=btb_ns          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=196G
#SBATCH -t 6:00:00         
#SBATCH --array=14-18 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="btbank1_0,btbank1_1,btbank2_4,btbank2_5,btbank3_1,btbank3_2,btbank7_1,btbank10_1"
eval_subject_trials="btbank1_2,btbank2_6,btbank3_0,btbank7_0,btbank10_0"
random_string_options=("CONT_EXP_CLS1" "CONT_EXP_CLS2" "CONT_EXP_CLS3")
symmetric_loss_options=(1)
learning_rate_options=(0.0015 0.0021 0.003 0.0042 0.006 0.0084)
future_bin_idx_options=(1)
spec_options=(1)
use_cls_token_options=(0 1)

wandb_project="btbank_tests"

# Calculate indices for two parallel jobs
idx1=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 ))
idx2=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 + 1 ))

n_rs=${#random_string_options[@]}
n_sl=${#symmetric_loss_options[@]}
n_lr=${#learning_rate_options[@]}
n_fb=${#future_bin_idx_options[@]}
n_sp=${#spec_options[@]}
n_uct=${#use_cls_token_options[@]}

# Convert indices for first job
random_string1=${random_string_options[$((idx1 % n_rs))]}
symmetric_loss1=${symmetric_loss_options[$(((idx1 / n_rs) % n_sl))]}
learning_rate1=${learning_rate_options[$(((idx1 / (n_rs * n_sl)) % n_lr))]}
future_bin_idx1=${future_bin_idx_options[$(((idx1 / (n_rs * n_sl * n_lr)) % n_fb))]}
spec1=${spec_options[$(((idx1 / (n_rs * n_sl * n_lr * n_fb)) % n_sp))]}
use_cls_token1=${use_cls_token_options[$(((idx1 / (n_rs * n_sl * n_lr * n_fb * n_sp)) % n_uct))]}

# Convert indices for second job
random_string2=${random_string_options[$((idx2 % n_rs))]}
symmetric_loss2=${symmetric_loss_options[$(((idx2 / n_rs) % n_sl))]}
learning_rate2=${learning_rate_options[$(((idx2 / (n_rs * n_sl)) % n_lr))]}
future_bin_idx2=${future_bin_idx_options[$(((idx2 / (n_rs * n_sl * n_lr)) % n_fb))]}
spec2=${spec_options[$(((idx2 / (n_rs * n_sl * n_lr * n_fb)) % n_sp))]}
use_cls_token2=${use_cls_token_options[$(((idx2 / (n_rs * n_sl * n_lr * n_fb * n_sp)) % n_uct))]}

echo "Job 1 - RS: $random_string1, SL: $symmetric_loss1, LR: $learning_rate1, FB: $future_bin_idx1, SPEC: $spec1, UCT: $use_cls_token1"
echo "Job 2 - RS: $random_string2, SL: $symmetric_loss2, LR: $learning_rate2, FB: $future_bin_idx2, SPEC: $spec2, UCT: $use_cls_token2"

# note: change train_model_fbi_combined.py to train_model.py for the non-combined version
python -u train_model.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string1 \
    --symmetric_loss $symmetric_loss1 \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate1 \
    --future_bin_idx $future_bin_idx1 \
    --spectrogram $spec1 \
    --use_cls_token $use_cls_token1 \
    --n_epochs 100 &

python -u train_model.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string2 \
    --symmetric_loss $symmetric_loss2 \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate2 \
    --future_bin_idx $future_bin_idx2 \
    --spectrogram $spec2 \
    --use_cls_token $use_cls_token2 \
    --n_epochs 100 &

wait