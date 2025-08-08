#!/bin/bash
#SBATCH --job-name=neural_regressions
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=384G
#SBATCH -t 24:00:00      
#SBATCH --output runs/logs/%A_%a.out
#SBATCH --error runs/logs/%A_%a.err
#SBATCH --array=1-30
#SBATCH -p normal

conda activate myenv

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# Define all subject-trial pairs from NEUROPROBE_FULL_SUBJECT_TRIALS
subject_trial_pairs=(
    "1,0" "1,1" "1,2"
    "2,0" "2,1" "2,2" "2,3" "2,4" "2,5" "2,6"
    "3,0" "3,1" "3,2"
    "4,0" "4,1" "4,2"
    "5,0"
    "6,0" "6,1" "6,4"
    "7,0" "7,1"
    "8,0"
    "9,0"
    "10,0" "10,1"
)

# Feature type options
feat_options=("clip" "dinov2" "audio")

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_pairs=${#subject_trial_pairs[@]}
n_feat=${#feat_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    pair_idx=$((idx % n_pairs))
    feat_idx=$((idx / n_pairs % n_feat))
    
    subject_trial=${subject_trial_pairs[$pair_idx]}
    feat=${feat_options[$feat_idx]}
    
    # Parse subject and trial from the pair
    IFS=',' read -r subject_id trial_id <<< "$subject_trial"
    
    echo "Job $((i+1)) - Subject: $subject_id, Trial: $trial_id, Feature: $feat"
    
    cd /om2/user/inimai/bfm && python -u analyses/inimai/regressions/clip_neural_regressions_v3.py \
        --subject_id $subject_id \
        --trial_id $trial_id \
        --feat $feat &
done

wait
