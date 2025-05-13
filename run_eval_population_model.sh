#!/bin/bash
#SBATCH --job-name=e_bb_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx001,dgx002
#SBATCH --mem=96G
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-1  # 285 if doing mini btbench
#SBATCH --output eval_logs/%A_%a.out # STDOUT
#SBATCH --error eval_logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate
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
declare -a time_granularity=(
    #1
    #4
    #16
    -1
)
declare -a splits_type=(
    #"SS_SM"
    "SS_DM"
)

model_dirname="M_nst19_dm192_dmb64_nh12_nl2_5_nes50_nf_nUTP_beT_pmt0.9_mtp100.0_SU_eeL_rBBFM_XXXt"
model_epoch=6

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
TIME_GRANULARITY_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#time_granularity[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#time_granularity[@]} % ${#splits_type[@]} ))   

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
TIME_GRANULARITY=${time_granularity[$TIME_GRANULARITY_IDX]}

SAVE_DIR="eval_results_lite_${SPLITS_TYPE}"

nvidia-smi

echo "Using python: $(which python)"
echo "Using python version: $(python --version)"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, splits_type $SPLITS_TYPE, time_granularity $TIME_GRANULARITY"
echo "Command: python -u eval_population_model.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --only_1second --time_granularity $TIME_GRANULARITY --model_dirname $model_dirname --model_epoch $model_epoch"

# Add the -u flag to Python to force unbuffered output
/om2/user/zaho/BrainBERT/.venv/bin/python -u eval_population_model.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --only_1second --time_granularity $TIME_GRANULARITY --model_dirname $model_dirname --model_epoch $model_epoch