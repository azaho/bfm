#!/bin/bash
#SBATCH --job-name=bfm_se          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1  # i only do this here to avoid volta gpus which have python 2.6
#SBATCH --mem=512G
#SBATCH -t 16:00:00         
#SBATCH --array=1-1 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p yanglab

# Export path to use Python 3 instead of Python 2
#export PATH=/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH

source .venv/bin/activate

pwd

export TMPDIR=/om2/scratch/tmp

# Check Python version
python --version

nvidia-smi

# note: change train_model_fbi_combined.py to train_model.py for the non-combined version
python -u train_model_single_electrode_3.py --n_samples_per_bin 4
wait
