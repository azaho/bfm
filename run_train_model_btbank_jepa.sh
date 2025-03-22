#!/bin/bash
#SBATCH --job-name=btb_jepa          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -t 6:00:00         
#SBATCH --array=1-32
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="btbank1_0,btbank1_1,btbank2_4,btbank2_5,btbank3_1,btbank3_2,btbank7_1,btbank10_1"
eval_subject_trials="btbank1_2,btbank2_6,btbank3_0,btbank7_0,btbank10_0"
random_string_options=("JEPA3")
symmetric_loss_options=(1)
learning_rate_options=(0.0021 0.003 0.0015 0.0008)
momentum_options=(0.96 0.95 0.94 0.93)
spec_options=(1 0)

wandb_project="jepa_tests"

idx=$((SLURM_ARRAY_TASK_ID-1))
random_string=${random_string_options[$((idx % ${#random_string_options[@]}))]}
symmetric_loss=${symmetric_loss_options[$(((idx / ${#random_string_options[@]}) % ${#symmetric_loss_options[@]}))]}
learning_rate=${learning_rate_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]}) % ${#learning_rate_options[@]}))]}
momentum=${momentum_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]} / ${#learning_rate_options[@]}) % ${#momentum_options[@]}))]}
spec=${spec_options[$(((idx / ${#random_string_options[@]} / ${#symmetric_loss_options[@]} / ${#learning_rate_options[@]} / ${#momentum_options[@]}) % ${#spec_options[@]}))]}

echo "random_string: $random_string"
echo "symmetric_loss: $symmetric_loss"
echo "learning_rate: $learning_rate"
echo "momentum: $momentum"
echo "spec: $spec"

python -u train_model_jepa.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string \
    --symmetric_loss $symmetric_loss \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --learning_rate $learning_rate \
    --spectrogram $spec \
    --momentum $momentum \
    --n_epochs 100