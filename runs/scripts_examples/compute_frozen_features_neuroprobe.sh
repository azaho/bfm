#!/bin/bash
#SBATCH --job-name=bfm_cff          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#####SBATCH --constraint=24GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-24  # 285 if doing mini btbench
#SBATCH --output runs/logs/cff_%A_%a.out # STDOUT
#SBATCH --error runs/logs/cff_%A_%a.err # STDERR
#SBATCH -p use-everything

nvidia-smi

export PYTHONUNBUFFERED=1
source .venv/bin/activate

model_dir="OM_wd0.0_dr0.0_rX1"

declare -a epochs=(0 40)

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "global_flow_angle"
    "local_flow_angle" 
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "delta_pitch"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
    "speaker"
)
# Create comma-separated string of eval names
EVAL_NAMES_STR=$(IFS=,; echo "${eval_names[*]}")

PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#subjects[@]} ))
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]}
EPOCH_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} ))
EPOCH=${epochs[$EPOCH_IDX]}

echo "Running for subject $SUBJECT_ID, trial $TRIAL_ID, eval $EVAL_NAMES_STR"

python -u analyses/compute_frozen_features_neuroprobe.py --model_dir $model_dir/ \
        --subject_id $SUBJECT_ID --trial_id $TRIAL_ID --eval_tasks $EVAL_NAMES_STR --model_epoch $EPOCH --batch_size 50