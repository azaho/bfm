#!/bin/bash
#SBATCH --job-name=mgh_max          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH -t 6:00:00         
#SBATCH --array=1-1 
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p yanglab
source ../bfm_ic2/.venv/bin/activate
export TMPDIR=/om2/scratch/tmp

python -u train_model_eeg_ieeg.py