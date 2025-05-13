#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:1
#SBATCH --constraint=48GB
#SBATCH --mem=240G
#SBATCH -t 16:00:00      
#SBATCH --array=1-24
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=2

train_subject_trial_options=(
    #"btbank1_0,btbank2_1,btbank2_2,btbank2_3,btbank2_5,btbank2_6,btbank3_2,btbank4_2,btbank5_0,btbank6_0,btbank6_1,btbank6_4,btbank8_0,btbank9_0,btbank1_1,btbank2_0,btbank3_1,btbank4_0,btbank7_0,btbank10_0"
    "btbank1_0,btbank2_1,btbank2_2,btbank2_3,btbank2_5,btbank2_6,btbank3_2,btbank4_2,btbank5_0,btbank6_0,btbank6_1,btbank6_4,btbank8_0,btbank9_0,btbank1_2,btbank3_2,btbank4_1,btbank7_1,btbank10_1"
)
eval_subject_trials="btbank3_1" #,btbank3_1,btbank3_2"
random_string_options=("BBFM_X1")

n_electrodes_subset_options=(50) #(1 2 4 8 16 32 64 124)
weight_decay_options=(0.0)
lr_schedule_options=("linear")
warmup_steps_options=(500) # XXX going back to fast warmup
init_identity_options=(0 1)
future_bin_idx_options=(1)
bin_encoder_options=("transformer") # "transformer")
use_temperature_param_options=(0)
show_a_embedding_options=(0.0 0.2 1.0)
show_b_embedding_options=(1.0)
separate_unembed_options=(0 1)
p_masked_timebins_options=(0.8 0.9)
max_temperature_param_options=(100.0 10.0)
n_layers_electrode_options=(2 4)
d_model_options=(64 128)

wandb_project="BFMM_ULCA"

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
n_se=${#separate_unembed_options[@]}
n_ut=${#use_temperature_param_options[@]}
n_sa=${#show_a_embedding_options[@]}
n_sb=${#show_b_embedding_options[@]}
n_pmt=${#p_masked_timebins_options[@]}
n_mtp=${#max_temperature_param_options[@]}
n_le=${#n_layers_electrode_options[@]}
n_dmo=${#d_model_options[@]}

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
    use_temperature_param=${use_temperature_param_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts % n_ut))]}
    show_a_embedding=${show_a_embedding_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut % n_sa))]}
    show_b_embedding=${show_b_embedding_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa % n_sb))]}
    separate_unembed=${separate_unembed_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb % n_se))]}
    p_masked_timebins=${p_masked_timebins_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se % n_pmt))]}
    max_temperature_param=${max_temperature_param_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt % n_mtp))]}
    n_layers_electrode=${n_layers_electrode_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp % n_le))]}
    d_model=${d_model_options[$((idx / n_rs / n_nes / n_wd / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp / n_le % n_dmo))]}

    echo "Job $((i+1)) - RS: $random_string, NES: $n_electrodes_subset, WD: $weight_decay, LRS: $lr_schedule, WSS: $warmup_steps, II: $init_identity, FBIN: $future_bin_idx, BE: $bin_encoder, SE: $separate_unembed, UT: $use_temperature_param, SA: $show_a_embedding, SB: $show_b_embedding, PMT: $p_masked_timebins, MTP: $max_temperature_param, NLE: $n_layers_electrode, DMO: $d_model"
    
    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u train_model_new.py  --cache_subjects 1 \
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
        --use_temperature_param $use_temperature_param \
        --p_show_a_embedding $show_a_embedding \
        --p_show_b_embedding $show_b_embedding \
        --separate_unembed $separate_unembed \
        --p_masked_timebins $p_masked_timebins \
        --max_temperature_param $max_temperature_param \
        --n_epochs 20 \
        --n_layers_electrode $n_layers_electrode \
        --d_model_bin $d_model &
done

wait