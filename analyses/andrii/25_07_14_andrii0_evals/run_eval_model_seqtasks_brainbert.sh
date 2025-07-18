#!/bin/bash
#SBATCH --job-name=eval_model          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --constraint=10GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-192  # 285 if doing mini btbench
#SBATCH --output runs/logs/%A_%a.out # STDOUT
#SBATCH --error runs/logs/%A_%a.err # STDERR
#SBATCH -p use-everything

nvidia-smi

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a model_dirs=(
    # "andrii0_lr0.003_wd0.001_dr0.0_rR1_t20250714_121055"
    # "andrii0_lr0.003_wd0.0_dr0.2_rR1_t20250714_121055"
    "andrii_brainbert_lr0.003_wd0.0_dr0.2_rR2_t20250716_001553"
)
BATCH_SIZE=300 # takes ~<10G of RAM

declare -a model_epochs=(0 1 10 20) #10 40)

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

declare -a splits_type=(
    # "SS_SM"
    "SS_DM"
    # "DS_DM"
)

declare -a classifier_type=(
    "linear"
    #"cnn"
    #"transformer"
)

declare -a feature_type=(
    "keepall"
    "meanE"
    # "cls"
    "meanT"
    "meanT_meanE"
    # "meanT_cls"
)

EVAL_STR=$(IFS=,; echo "${eval_names[*]}")

# Calculate indices for this task
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#subjects[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} % ${#splits_type[@]} ))
CLASSIFIER_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} / ${#splits_type[@]} % ${#classifier_type[@]} ))
MODEL_DIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} / ${#splits_type[@]} / ${#classifier_type[@]} % ${#model_dirs[@]} ))
MODEL_EPOCH_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} / ${#splits_type[@]} / ${#classifier_type[@]} / ${#model_dirs[@]} % ${#model_epochs[@]} ))
FEATURE_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#subjects[@]} / ${#splits_type[@]} / ${#classifier_type[@]} / ${#model_dirs[@]} / ${#model_epochs[@]} % ${#feature_type[@]} ))

# Get subject, trial and eval name for this task
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
CLASSIFIER_TYPE=${classifier_type[$CLASSIFIER_TYPE_IDX]}
MODEL_DIR=${model_dirs[$MODEL_DIR_IDX]}
MODEL_EPOCH=${model_epochs[$MODEL_EPOCH_IDX]}
FEATURE_TYPE=${feature_type[$FEATURE_TYPE_IDX]}
save_dir="runs/analyses/andrii/25_07_14_andrii0_evals/eval_results_lite_${SPLITS_TYPE}"

echo "Running eval for subject $SUBJECT, trial $TRIAL, classifier $CLASSIFIER_TYPE, model $MODEL_DIR, epoch $MODEL_EPOCH, feature type $FEATURE_TYPE"
echo "Save dir: $save_dir"
echo "Split type: $SPLITS_TYPE"

# Add the -u flag to Python to force unbuffered output
python -u analyses/andrii/25_07_14_andrii0_evals/eval_model.py \
    --eval_name $EVAL_STR \
    --subject_id $SUBJECT \
    --trial_id $TRIAL \
    --model_dir $MODEL_DIR \
    --model_epoch $MODEL_EPOCH \
    --batch_size $BATCH_SIZE \
    --save_dir $save_dir \
    --split_type $SPLITS_TYPE \
    --feature_type $FEATURE_TYPE \
    --only_1second