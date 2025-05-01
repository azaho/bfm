#!/bin/bash
#SBATCH --job-name=bfm_se          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1  # i only do this here to avoid volta gpus which have python 2.6
#SBATCH --mem=64G
#SBATCH -t 6:00:00         
#SBATCH --array=1-20 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p use-everything

# Export path to use Python 3 instead of Python 2
#export PATH=/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH

source .venv/bin/activate

pwd


export TMPDIR=/om2/scratch/tmp

d_hilbert_options=(2 3 4 5 6)
d_embed_options=(128)
bits_per_sample_options=(3 4 5 6)

idx=$(( ($SLURM_ARRAY_TASK_ID-1) ))

d_hilbert=${d_hilbert_options[$((idx % ${#d_hilbert_options[@]}))]}
d_embed=${d_embed_options[$((idx / ${#d_hilbert_options[@]} % ${#d_embed_options[@]}))]}
bits_per_sample=${bits_per_sample_options[$((idx / (${#d_hilbert_options[@]} * ${#d_embed_options[@]})))]}

# Check Python version
python --version


nvidia-smi

# note: change train_model_fbi_combined.py to train_model.py for the non-combined version
python -u train_model_single_electrode_new.py --d_hilbert $d_hilbert --d_embed $d_embed --bits_per_sample $bits_per_sample