#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mem=196G
#SBATCH -t 16:00:00      
#SBATCH --array=1-8
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=1

train_subject_trials="btbank3_1"
eval_subject_trials="btbank3_0" #,btbank3_1,btbank3_2"
random_string_options=("X7_NT")

n_electrodes_subset_options=(50) #(1 2 4 8 16 32 64 124)
weight_decay_options=(0.0 0.0001 0.0005 0.001)
lr_schedule_options=("linear")
warmup_steps_options=(100)
init_identity_options=(0)
spectrogram_power_options=(0 1)

wandb_project="XX_tests"


# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))

n_rs=${#random_string_options[@]}
n_nes=${#n_electrodes_subset_options[@]}
n_wd=${#weight_decay_options[@]}
n_lr_schedule=${#lr_schedule_options[@]}
n_ws=${#warmup_steps_options[@]}
n_ii=${#init_identity_options[@]}
n_sp=${#spectrogram_power_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    random_string=${random_string_options[$((idx % n_rs))]}
    n_electrodes_subset=${n_electrodes_subset_options[$((idx / n_rs % n_nes))]}
    weight_decay=${weight_decay_options[$((idx / n_rs / n_nes % n_wd))]}
    lr_schedule=${lr_schedule_options[$((idx / n_rs / n_nes / n_wd % n_lr_schedule))]}
    warmup_steps=${warmup_steps_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule % n_ws))]}
    init_identity=${init_identity_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws % n_ii))]}
    spectrogram_power=${spectrogram_power_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii % n_sp))]}

    echo "Job $((i+1)) - RS: $random_string, NES: $n_electrodes_subset, WD: $weight_decay, LRS: $lr_schedule, WSS: $warmup_steps, II: $init_identity, SP: $spectrogram_power"
    
    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u train_model_tcn3.py  --cache_subjects 1 \
        --num_workers_dataloaders 4 \
        --batch_size 100 \
        --random_string $random_string \
        --n_electrodes_subset $n_electrodes_subset \
        --train_subject_trials $train_subject_trials \
        --eval_subject_trials $eval_subject_trials \
        --wandb_project $wandb_project \
        --weight_decay $weight_decay \
        --lr_schedule $lr_schedule \
        --warmup_steps $warmup_steps \
        --init_identity $init_identity \
        --spectrogram 1 \
        --spectrogram_power $spectrogram_power \
        --n_epochs 100 &
done

wait