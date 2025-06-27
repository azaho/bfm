#!/bin/bash
#SBATCH --job-name=decode_all_splits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --exclude=dgx001,dgx002
#SBATCH -p use-everything
#SBATCH --array=1-912
#SBATCH --output=runs/logs/decode_%A_%a.out
#SBATCH --error=runs/logs/decode_%A_%a.err

# Activate environment
nvidia-smi
source .venv/bin/activate

CONFIG_PATH=$1
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: sbatch submit_decode_jobs.sh path/to/model_config.json"
    exit 1
fi

# Extract fields from config
TASKS=($(jq -r '.tasks[]' "$CONFIG_PATH"))
SPLITS=($(jq -r '.splits[]' "$CONFIG_PATH"))
MODEL_TITLES=($(jq -r '.models[].title' "$CONFIG_PATH"))
MODEL_PATHS=($(jq -r '.models[].path' "$CONFIG_PATH"))
MODEL_EPOCHS=($(jq -r '.models[].epoch' "$CONFIG_PATH"))
EVAL_RESULTS_ROOT=$(jq -r '.eval_results_output_root' "$CONFIG_PATH")

# 12 subject-trial pairs
subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
trials=(1 2 0 4 0 1 0 1 0 1 0 1)

total_tasks=${#TASKS[@]}
total_pairs=${#subjects[@]}
total_splits=${#SPLITS[@]}
total_models=${#MODEL_TITLES[@]}
total_jobs=$((total_models * total_tasks * total_pairs * total_splits))

if [ "$SLURM_ARRAY_TASK_ID" -gt "$total_jobs" ]; then
  echo "Array task ID exceeds total combinations. Exiting."
  exit 1
fi

PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID - 1) % total_pairs ))
SPLIT_IDX=$(( (($SLURM_ARRAY_TASK_ID - 1) / total_pairs) % total_splits ))
TASK_IDX=$(( (($SLURM_ARRAY_TASK_ID - 1) / (total_pairs * total_splits)) % total_tasks ))
MODEL_IDX=$(( ($SLURM_ARRAY_TASK_ID - 1) / (total_pairs * total_splits * total_tasks) ))

TRAIN_SUBJECT=${subjects[$PAIR_IDX]}
TRAIN_TRIAL=${trials[$PAIR_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}
TASK=${TASKS[$TASK_IDX]}
MODEL_NAME="${MODEL_TITLES[$MODEL_IDX]}"
MODEL_DIR="${MODEL_PATHS[$MODEL_IDX]}"
MODEL_EPOCH="${MODEL_EPOCHS[$MODEL_IDX]}"

SAVE_DIR="${EVAL_RESULTS_ROOT}/${MODEL_NAME// /_}/eval_results_frozen_features_${SPLIT}"

echo "Running decoding for $MODEL_NAME | split: $SPLIT | task: $TASK | train: S$TRAIN_SUBJECT T$TRAIN_TRIAL"

mkdir -p runs/logs/locks

for i in "${!trials[@]}"; do
    TEST_SUBJECT=${subjects[$i]}
    TEST_TRIAL=${trials[$i]}

    if [ "$SPLIT" == "SS_SM" ] && [ "$TEST_SUBJECT" -ne "$TRAIN_SUBJECT" -o "$TEST_TRIAL" -ne "$TRAIN_TRIAL" ]; then
        continue
    elif [ "$SPLIT" == "SS_DM" ] && [ "$TEST_SUBJECT" -ne "$TRAIN_SUBJECT" -o "$TEST_TRIAL" -eq "$TRAIN_TRIAL" ]; then
        continue
    elif [ "$SPLIT" == "DS_SM" ] && [ "$TEST_SUBJECT" -eq "$TRAIN_SUBJECT" -o "$TEST_TRIAL" -ne "$TRAIN_TRIAL" ]; then
        continue
    elif [ "$SPLIT" == "DS_DM" ] && [ "$TEST_SUBJECT" -eq "$TRAIN_SUBJECT" -o "$TEST_TRIAL" -eq "$TRAIN_TRIAL" ]; then
        continue
    fi

    LOCKFILE="runs/logs/locks/${MODEL_NAME// /_}_${SPLIT}_btbank${TEST_SUBJECT}_${TEST_TRIAL}_${TASK}.lock"

    flock "$LOCKFILE" python -u /om2/user/brupesh/bfm/analyses/bhadra/eval_frozen_features.py \
        --train_subject_id $TRAIN_SUBJECT \
        --train_trial_id $TRAIN_TRIAL \
        --test_subject_id $TEST_SUBJECT \
        --test_trial_id $TEST_TRIAL \
        --task $TASK \
        --model_epoch $MODEL_EPOCH \
        --split_type $SPLIT \
        --features_root $MODEL_DIR/frozen_features_neuroprobe \
        --save_dir $SAVE_DIR \
        --verbose

done
