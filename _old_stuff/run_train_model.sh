#!/bin/bash
#SBATCH --job-name=btb_ns          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=196G
#SBATCH -t 16:00:00         
#SBATCH --array=1-6 # needs to go to 72
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp
# Set the number of parallel jobs to run per array task
n_in_parallel=1

train_subject_trials="btbank3_0,btbank3_1,btbank3_2"
eval_subject_trials="btbank3_0" #,btbank3_1,btbank3_2"
random_string_options=("STS_CONCAT2")

n_electrodes_subset_options=(124) #(1 2 4 8 16 32 64 124)
normalize_features_options=(0 1)
sample_timebin_size_options=(0.125 0.0625 0.03125) # 0.015625 0.0078125) # (0.125 0.0625 0.03125)
spectrogram_options=(0)

wandb_project="granular_tests"


# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))

n_rs=${#random_string_options[@]}
n_nes=${#n_electrodes_subset_options[@]}
n_nf=${#normalize_features_options[@]}
n_sp=${#spectrogram_options[@]}
n_sts=${#sample_timebin_size_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    random_string=${random_string_options[$((idx % n_rs))]}
    n_electrodes_subset=${n_electrodes_subset_options[$((idx / n_rs % n_nes))]}
    normalize_features=${normalize_features_options[$((idx / (n_rs * n_nes) % n_nf))]}
    spectrogram=${spectrogram_options[$((idx / (n_rs * n_nes * n_nf) % n_sp))]}
    sample_timebin_size=${sample_timebin_size_options[$((idx / (n_rs * n_nes * n_nf * n_sp) % n_sts))]}

    echo "Job $((i+1)) - RS: $random_string, NES: $n_electrodes_subset, NF: $normalize_features, SP: $spectrogram, STS: $sample_timebin_size"
    
    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u train_model.py  --cache_subjects 1 \
        --num_workers_dataloaders 4 \
        --batch_size 72 \
        --random_string $random_string \
        --n_electrodes_subset $n_electrodes_subset \
        --train_subject_trials $train_subject_trials \
        --eval_subject_trials $eval_subject_trials \
        --wandb_project $wandb_project \
        --n_epochs 100 \
        --sample_timebin_size $sample_timebin_size \
        --normalize_features $normalize_features \
        --spectrogram $spectrogram &
done

wait