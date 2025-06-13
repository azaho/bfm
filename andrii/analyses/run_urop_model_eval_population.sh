#!/bin/bash
#SBATCH --job-name=urop_model_extract_features          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 1:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-30      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output run_logs/%A_%a.out # STDOUT
#SBATCH --error run_logs/%A_%a.err # STDERR
#SBATCH -p normal

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Create arrays of subject IDs and trial IDs that correspond to array task ID
# for all subject_trials in the dataset
declare -a subjects=(1 2 3 7 10)
declare -a trials=(2 6 0 0 0)
declare -a eval_names=(
    "volume"
    "speech"
    "onset"
    "speaker"
)
declare -a model_epochs=(0 20 40 60 80 100)

# Calculate indices for this task
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % 5 ))
MODEL_EPOCH_IDX=$(( (($SLURM_ARRAY_TASK_ID-1) / 5) % 6 ))
MODEL_DIR="M_nst8_dm192_nh12_nl5_5_eaM_projN_eeL_fb1_cls_lr0.003_rCONT_EXP_CLS1"

# Get subject, trial and eval name for this task
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]} 
MODEL_EPOCH=${model_epochs[$MODEL_EPOCH_IDX]}

nvidia-smi

EVAL_NAMES_STR=$(IFS=,; echo "${eval_names[*]}")
echo ""
echo ""
echo "--------------------------------"
echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, all evals"
python -u analyses/urop_model_extract_features.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_names $EVAL_NAMES_STR --verbose --model_dir $MODEL_DIR --model_epoch $MODEL_EPOCH --batch_size 10

# M_nst8_dm192_nh12_nl5_5_lr0.003_rSL1