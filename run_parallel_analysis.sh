#!/bin/bash
#SBATCH --job-name=parallel_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -p normal
#SBATCH --requeue

source .venv/bin/activate

# Configuration
MODEL_DIR="andrii0_wd0.0001_dr0.1_rTEMP"
MODEL_EPOCH=100

# List of subject/trial pairs to analyze
subject_trials=(
    "1 0" "1 1" "1 2"
    "2 0" "2 1" "2 2" "2 3" "2 4" "2 5" "2 6"
    "3 0" "3 1" "3 2"
    "4 0" "4 1" "4 2"
    "7 0" "7 1"
    "10 0" "10 1"
)

# Process each subject/trial pair
for pair in "${subject_trials[@]}"; do
    read -r subject_id trial_id <<< "$pair"
    
    echo "Processing subject ${subject_id}, trial ${trial_id}"
    
    python -m analyses.compute_key_electrodes --model_dir $MODEL_DIR --model_epoch $MODEL_EPOCH --overwrite --subject_id $subject_id --trial_id $trial_id
    python -m analyses.roshnipm.visualize_attention --single_file "runs/data/$MODEL_DIR/key_electrodes/model_epoch$MODEL_EPOCH/key_electrodes_btbank${subject_id}_${trial_id}.npy"
    
    echo "Completed subject ${subject_id}, trial ${trial_id}"
    echo "----------------------------------------"
done