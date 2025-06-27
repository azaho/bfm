#!/bin/bash

# Activate environment
nvidia-smi
source .venv/bin/activate

CONFIG_PATH=$1
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: ./generate_all_figures.sh path/to/model_config.json"
    exit 1
fi

echo "Generating figures for config: $CONFIG_PATH"

TASKS=($(jq -r '.tasks[]' "$CONFIG_PATH"))
SPLITS=($(jq -r '.splits[]' "$CONFIG_PATH"))
EVAL_RESULTS_ROOT=$(jq -r '.eval_results_output_root' "$CONFIG_PATH")
FIGURE_OUTPUT_DIR=$(jq -r '.figure_output_dir' "$CONFIG_PATH")
EPOCH=$(jq -r '.models[0].epoch' "$CONFIG_PATH")
MODEL_NAME=$(jq -r '.models[0].title' "$CONFIG_PATH")

for split in "${SPLITS[@]}"; do
    echo "Generating bar plot for split: $split"
    python /om2/user/brupesh/bfm/analyses/neuroprobe_generate_figure.py --split_type "$split" --save_dir $FIGURE_OUTPUT_DIR/$MODEL_NAME/
done

for task in "${TASKS[@]}"; do
    echo "Generating AUROC matrix for task: $task"
    python /om2/user/brupesh/bfm/analyses/bhadra/cross_subj_matrices.py \
        --task "$task" \
        --epoch $EPOCH \
        --base_dir $EVAL_RESULTS_ROOT/$MODEL_NAME/eval_results_frozen_features_ \
        --output_dir $FIGURE_OUTPUT_DIR/$MODEL_NAME/matrices/
done
