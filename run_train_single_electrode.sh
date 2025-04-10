#!/bin/bash
#SBATCH --job-name=bfm_se          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=196G
#SBATCH -t 6:00:00         
#SBATCH --array=1-1 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="btbank1_0,btbank1_1,btbank2_4,btbank2_5,btbank3_1,btbank3_2,btbank7_1,btbank10_1"
eval_subject_trials="btbank1_2,btbank2_6,btbank3_0,btbank7_0,btbank10_0"
random_string="T1"

wandb_project="single_electrode1"

nvidia-smi

# note: change train_model_fbi_combined.py to train_model.py for the non-combined version
python -u train_model_single_electrode.py  --cache_subjects 1 \
    --num_workers_dataloaders 4 \
    --random_string $random_string \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials $eval_subject_trials \
    --wandb_project $wandb_project \
    --n_epochs 100 \
    --batch_size 10000