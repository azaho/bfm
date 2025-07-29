#!/bin/bash
#SBATCH --job-name=decode_all_splits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --exclude=dgx001,dgx002
#SBATCH -p use-everything
#SBATCH --array=1-684
#SBATCH --output=runs/logs/decode_%A_%a.out
#SBATCH --error=runs/logs/decode_%A_%a.err

# Activate environment
nvidia-smi
source /om2/user/zaho/bfm/.venv/bin/activate

CONFIG_PATH=$1
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: sbatch analyses/bhadra/25_07_02_pipeline/submit_decode_job.sh path/to/model_config.json"
    exit 1
fi

# Extract fields from config
eval "$(python analyses/bhadra/25_07_02_pipeline/parse_config.py "$CONFIG_PATH")"

total_tasks=${#TASKS[@]}
total_pairs=${#subjects[@]}

PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID - 1) % total_pairs ))
TASK_IDX=$(( (($SLURM_ARRAY_TASK_ID - 1) / total_pairs) % total_tasks ))
EPOCH_IDX=$(( ($SLURM_ARRAY_TASK_ID - 1) / (total_pairs * total_tasks) ))

TRAIN_SUBJECT=${subjects[$PAIR_IDX]}
TRAIN_TRIAL=${trials[$PAIR_IDX]}
TASK=${TASKS[$TASK_IDX]}
MODEL_EPOCH=${PRIMARY_EPOCHS[$EPOCH_IDX]}

echo "Running decoding for $PRIMARY_TITLE | epoch: $MODEL_EPOCH | task: $TASK | train: S$TRAIN_SUBJECT T$TRAIN_TRIAL"

mkdir -p runs/logs/locks

for i in "${!trials[@]}"; do
    TEST_SUBJECT=${subjects[$i]}
    TEST_TRIAL=${trials[$i]}

    if [ "$TRAIN_SUBJECT" -eq "$TEST_SUBJECT" ] && [ "$TRAIN_TRIAL" -eq "$TEST_TRIAL" ]; then
    SPLIT="SS_SM"
    elif [ "$TRAIN_SUBJECT" -eq "$TEST_SUBJECT" ]; then
        SPLIT="SS_DM"
    elif [ "$TRAIN_TRIAL" -eq "$TEST_TRIAL" ]; then
        SPLIT="DS_SM"
    else
        SPLIT="DS_DM"
    fi

    SAVE_DIR="${EVAL_RESULTS_ROOT}/${PRIMARY_TITLE}/eval_results_frozen_features_${SPLIT}"

    LOCKFILE="runs/logs/locks/${PRIMARY_TITLE}_${SPLIT}_btbank${TEST_SUBJECT}_${TEST_TRIAL}_${TASK}_epoch${MODEL_EPOCH}.lock"

    flock "$LOCKFILE" python -u analyses/bhadra/25_07_02_pipeline/eval_frozen_features.py \
        --train_subject_id $TRAIN_SUBJECT \
        --train_trial_id $TRAIN_TRIAL \
        --test_subject_id $TEST_SUBJECT \
        --test_trial_id $TEST_TRIAL \
        --task $TASK \
        --model_epoch $MODEL_EPOCH \
        --split_type $SPLIT \
        --features_root $PRIMARY_PATH/frozen_features_neuroprobe \
        --save_dir $SAVE_DIR \
        --verbose
done