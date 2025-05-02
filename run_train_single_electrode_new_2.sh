#!/bin/bash
#SBATCH --job-name=bfm_se          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1  # i only do this here to avoid volta gpus which have python 2.6
#SBATCH --mem=64G
#SBATCH -t 16:00:00         
#SBATCH --array=1-4 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal

# Export path to use Python 3 instead of Python 2
#export PATH=/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH

source .venv/bin/activate

pwd

export TMPDIR=/om2/scratch/tmp

resolution_options=(2 4 8 16 32 64 128 256)

idx=$(($SLURM_ARRAY_TASK_ID - 1))

resolution1=${resolution_options[$((idx*2))]}
resolution2=${resolution_options[$((idx*2+1))]}

# Check Python version
python --version

nvidia-smi

# note: change train_model_fbi_combined.py to train_model.py for the non-combined version
python -u train_model_single_electrode_2.py --resolution $resolution1 &
python -u train_model_single_electrode_2.py --resolution $resolution2 &
wait
