#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:1
#SBATCH --constraint=40GB
#SBATCH --mem=96G
#SBATCH -t 16:00:00      
#SBATCH --array=1-2
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=1

train_subject_trial_options=(
    "btbank3_0"
)
eval_subject_trials="btbank3_1" #,btbank3_1,btbank3_2"
random_string_options=("BBFM_M_1")

n_electrodes_subset_options=(50) #(1 2 4 8 16 32 64 124)
weight_decay_options=(0.0)
lr_schedule_options=("linear")
warmup_steps_options=(100) # XXX going back to fast warmup
init_identity_options=(1)
future_bin_idx_options=(1 2)
bin_encoder_options=("linear") # "transformer")

wandb_project="BBFM_min_tests"

sample_timebin_size=0.125 #0.0625


# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))

n_rs=${#random_string_options[@]}
n_nes=${#n_electrodes_subset_options[@]}
n_wd=${#weight_decay_options[@]}
n_lr_schedule=${#lr_schedule_options[@]}
n_ws=${#warmup_steps_options[@]}
n_ii=${#init_identity_options[@]}
n_fb=${#future_bin_idx_options[@]}
n_be=${#bin_encoder_options[@]}
n_ts=${#train_subject_trial_options[@]}

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
    future_bin_idx=${future_bin_idx_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii % n_fb))]}
    bin_encoder=${bin_encoder_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb % n_be))]}
    train_subject_trials=${train_subject_trial_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be % n_ts))]}

    echo "Job $((i+1)) - RS: $random_string, NES: $n_electrodes_subset, WD: $weight_decay, LRS: $lr_schedule, WSS: $warmup_steps, II: $init_identity, FBIN: $future_bin_idx, BE: $bin_encoder"
    
    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u train_model.py  --cache_subjects 1 \
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
        --future_bin_idx $future_bin_idx \
        --bin_encoder $bin_encoder \
        --sample_timebin_size $sample_timebin_size \
        --n_epochs 100 &
done

wait