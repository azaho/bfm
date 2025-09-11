#!/bin/bash
#SBATCH --job-name=e_p_lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --constraint=24GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-192
#SBATCH --output runs/logs_ft/%A_%a.out # STDOUT
#SBATCH --error runs/logs_ft/%A_%a.err # STDERR
#SBATCH -p use-everything

nvidia-smi

export PYTHONUNBUFFERED=1
export ROOT_DIR_BRAINTREEBANK=/om2/user/zaho/braintreebank/braintreebank/
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
eval_names=(
    $(IFS=,; echo "${eval_names[*]}") # This line is needed to convert the array to a comma-separated string. Feel free to remove it if you don't need it.
)

declare -a model_dirs=(
    "andrii0_lr0.003_wd0.0_dr0.2_rR2_t20250905_141853"
    "mse_ar_lr0.003_wd0.0_dr0.2_rR2_t20250905_141854"
    "mse_mtm_lr0.003_wd0.0_dr0.2_rR2_t20250905_141646"
    "mse_rm_lr0.003_wd0.0_dr0.2_rR2_t20250905_141848"
)

declare -a model_epochs=(
    0 30
)

declare -a finetuning_feature_aggregation_methods=(
    "meanT_meanE"
    "keepall"
)

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
MODEL_DIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#model_dirs[@]} ))
MODEL_EPOCH_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#model_dirs[@]} % ${#model_epochs[@]} ))
FINETUNING_FEATURE_AGGREGATION_METHOD_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#model_dirs[@]} / ${#model_epochs[@]} % ${#finetuning_feature_aggregation_methods[@]} ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
MODEL_DIR=${model_dirs[$MODEL_DIR_IDX]}
MODEL_EPOCH=${model_epochs[$MODEL_EPOCH_IDX]}
FINETUNING_FEATURE_AGGREGATION_METHOD=${finetuning_feature_aggregation_methods[$FINETUNING_FEATURE_AGGREGATION_METHOD_IDX]}
echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
echo "Model dir: $MODEL_DIR"
echo "Model epoch: $MODEL_EPOCH"
echo "Finetuning feature aggregation method: $FINETUNING_FEATURE_AGGREGATION_METHOD"

# Add the -u flag to Python to force unbuffered output
python -u analyses/neuroprobe_eval_finetune.py \
    --eval_name $EVAL_NAME \
    --subject_id $SUBJECT \
    --trial_id $TRIAL \
    --model_dir $MODEL_DIR \
    --model_epoch $MODEL_EPOCH \
    --finetuning_batch_size 96 \
    --random_seed 42 \
    --finetuning_feature_aggregation_method $FINETUNING_FEATURE_AGGREGATION_METHOD