#!/bin/bash
#SBATCH --job-name=mgh_max          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=64 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH -t 160:00:00         
#SBATCH --array=1-1 
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p yanglab
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

train_subject_trials="mgh1_0,mgh1_1,mgh1_2,mgh1_3,mgh2_0,mgh2_1,mgh3_0,mgh3_1,mgh3_2,mgh4_0,mgh5_0,mgh6_0,mgh6_1,mgh7_0,mgh7_1,mgh7_2,mgh8_0,mgh9_0,mgh9_1,mgh9_2,mgh10_0,mgh10_1,mgh10_2,mgh10_3,mgh10_4,mgh11_0,mgh11_1,mgh12_0,mgh12_1,mgh13_0,mgh14_0,mgh14_1,mgh15_0,mgh16_0,mgh16_1,mgh16_2,mgh16_3,mgh16_4,mgh16_5,mgh16_6,mgh17_0,mgh17_1,mgh18_0,mgh18_1,mgh19_0,mgh19_1,mgh20_0,mgh21_0,mgh22_0,mgh23_0,mgh23_1,mgh24_0,mgh25_0,mgh25_1,mgh26_0,mgh27_0,mgh27_1,mgh28_0,mgh28_1,mgh29_0,mgh29_1,mgh30_0,mgh30_1,mgh31_0,mgh32_0,mgh32_1,mgh33_0,mgh34_0,mgh34_1,mgh34_2,mgh34_3,mgh34_4,mgh34_5,mgh34_6,mgh35_0,mgh36_0,mgh36_1,mgh36_2,mgh36_3,mgh39_0,mgh40_0,mgh40_1,mgh40_2,mgh40_3,mgh40_4,mgh40_5,mgh40_6,mgh41_0,mgh42_0,mgh43_0,mgh43_1,mgh43_2,mgh44_0,mgh45_0,mgh46_0,mgh47_0,mgh48_0,mgh49_0,mgh50_0,mgh50_1,mgh51_0,mgh52_0,mgh53_0,mgh54_0,mgh54_1,mgh55_0,mgh56_0,mgh57_0,mgh57_1,mgh59_0,mgh60_0,mgh60_1,mgh61_0,mgh62_0,mgh62_1,mgh62_2"
eval_subject_trials=""

random_string="MGH_MAX"
wandb_project="mgh_tests"

python -u train_model.py  --cache_subjects 0 \
    --num_workers_dataloaders 60 \
    --random_string $random_string \
    --train_subject_trials $train_subject_trials \
    --eval_subject_trials "" \
    --wandb_project $wandb_project \
    --save_model_every_n_epochs 1