#!/bin/bash

# Activate environment
nvidia-smi
export PYTHONPATH=/om2/user/brupesh/bfm:$PYTHONPATH
source /om2/user/zaho/bfm/.venv/bin/activate

CONFIG_PATH=$1
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: ./generate_all_figures.sh path/to/model_config.json"
    exit 1
fi

echo "Generating figures for config: $CONFIG_PATH"

# Extract fields from config
eval "$(python analyses/bhadra/25_07_02_pipeline/parse_config.py "$CONFIG_PATH")"

# Run generate_bars.py once per split type (derived dynamically inside Python)
echo "Generating comparison bar plots for all tasks"
for SPLIT_TYPE in SS_SM SS_DM DS_DM; do
    echo "Generating bar plots for split: $SPLIT_TYPE"
    python analyses/bhadra/25_07_02_pipeline/generate_bars.py \
        --primary_title "$PRIMARY_TITLE" \
        --primary_epochs "${PRIMARY_EPOCHS[@]}" \
        --primary_eval_results_path "$PRIMARY_EVAL_RESULTS_PATH" \
        --comparison_models_json "$COMPARISON_MODELS" \
        --save_dir "$FIGURE_OUTPUT_DIR/$PRIMARY_TITLE/bar_plots/" \
        --split_type "$SPLIT_TYPE"
done

# Now generate AUROC matrices per epoch and task
for epoch in "${PRIMARY_EPOCHS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Generating AUROC matrix for task: $task (epoch $epoch)"
        python analyses/bhadra/25_07_02_pipeline/cross_subj_matrices.py \
            --task "$task" \
            --epoch "$epoch" \
            --base_dir "$EVAL_RESULTS_ROOT/$PRIMARY_TITLE/eval_results_frozen_features_" \
            --output_dir "$FIGURE_OUTPUT_DIR/$PRIMARY_TITLE/matrices/epoch$epoch/"
    done
    python analyses/bhadra/25_07_02_pipeline/cross_subj_matrices.py \
        --task all_tasks_avg \
        --epoch "$epoch" \
        --base_dir "$EVAL_RESULTS_ROOT/$PRIMARY_TITLE/eval_results_frozen_features_" \
        --output_dir "$FIGURE_OUTPUT_DIR/$PRIMARY_TITLE/matrices/epoch$epoch/"
done