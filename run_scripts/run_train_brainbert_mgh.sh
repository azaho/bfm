#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=240G
#SBATCH -t 48:00:00      
#SBATCH --array=1-16
#SBATCH --output run_logs/%A_%a.out
#SBATCH --error run_logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=1

train_subject_trial_options=(
    "mgh14_0"
)
eval_subject_trials=""

random_string_options=("MGH1")

learning_rate_options=(0.003) # 0.001 0.01)
max_n_electrodes_options=(20) #(1 2 4 8 16 32 64 124)
weight_decay_options=(0.0)
spectrogram_options=(0 1)
loss_type_options=("contrastive" "l2")
lr_schedule_options=("linear")
warmup_steps_options=(100) # XXX going back to fast warmup
init_identity_options=(0)
future_bin_idx_options=(0)
bin_encoder_options=("transformer") # "transformer")
use_temperature_param_options=(1)
show_a_embedding_options=(0.0)
show_b_embedding_options=(1.0)
separate_unembed_options=(0 1)
p_masked_timebins_options=(0.2 0.5)
max_temperature_param_options=(1000.0)
n_layers_electrode_options=(6)
d_model_options=(384)
first_kernel_options=(256)
causal_options=(0)

wandb_project="BB2"

sample_timebin_size=0.125 #0.0625


# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))

n_rs=${#random_string_options[@]}
n_nes=${#max_n_electrodes_options[@]}
n_wd=${#weight_decay_options[@]}
n_sp=${#spectrogram_options[@]}
n_lt=${#loss_type_options[@]}
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
n_fk=${#first_kernel_options[@]}
n_lr=${#learning_rate_options[@]}
n_co=${#causal_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    random_string=${random_string_options[$((idx % n_rs))]}
    max_n_electrodes=${max_n_electrodes_options[$((idx / n_rs % n_nes))]}
    weight_decay=${weight_decay_options[$((idx / n_rs / n_nes % n_wd))]}
    spectrogram=${spectrogram_options[$((idx / n_rs / n_nes / n_wd % n_sp))]}
    loss_type=${loss_type_options[$((idx / n_rs / n_nes / n_wd / n_sp % n_lt))]}
    lr_schedule=${lr_schedule_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt % n_lr_schedule))]}
    warmup_steps=${warmup_steps_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule % n_ws))]}
    init_identity=${init_identity_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws % n_ii))]}
    future_bin_idx=${future_bin_idx_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii % n_fb))]}
    bin_encoder=${bin_encoder_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb % n_be))]}
    train_subject_trials=${train_subject_trial_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be % n_ts))]}
    use_temperature_param=${use_temperature_param_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts % n_ut))]}
    show_a_embedding=${show_a_embedding_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut % n_sa))]}
    show_b_embedding=${show_b_embedding_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa % n_sb))]}
    separate_unembed=${separate_unembed_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb % n_se))]}
    p_masked_timebins=${p_masked_timebins_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se % n_pmt))]}
    max_temperature_param=${max_temperature_param_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt % n_mtp))]}
    n_layers_electrode=${n_layers_electrode_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp % n_le))]}
    d_model=${d_model_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp / n_le / n_dmo % n_dmo))]}
    first_kernel=${first_kernel_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp / n_le / n_dmo % n_fk))]}
    causal=${causal_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp / n_le / n_dmo / n_fk % n_co))]}
    learning_rate=${learning_rate_options[$((idx / n_rs / n_nes / n_wd / n_sp / n_lt / n_lr_schedule / n_ws / n_ii / n_fb / n_be / n_ts / n_ut / n_sa / n_sb / n_se / n_pmt / n_mtp / n_le / n_dmo / n_fk / n_lr % n_lr))]}

    echo "Job $((i+1)) - RS: $random_string, NES: $max_n_electrodes, WD: $weight_decay, SP: $spectrogram, LT: $loss_type, LRS: $lr_schedule, WSS: $warmup_steps, II: $init_identity, FBIN: $future_bin_idx, BE: $bin_encoder, SE: $separate_unembed, UT: $use_temperature_param, SA: $show_a_embedding, SB: $show_b_embedding, PMT: $p_masked_timebins, MTP: $max_temperature_param, NLE: $n_layers_electrode, DMO: $d_model, FK: $first_kernel, CO: $causal, LR: $learning_rate"
    
    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u brainbert_train.py  --cache_subjects 1 \
        --num_workers_dataloaders 4 \
        --batch_size 256 \
        --random_string $random_string \
        --max_n_electrodes $max_n_electrodes \
        --train_subject_trials $train_subject_trials \
        --eval_subject_trials "" \
        --weight_decay $weight_decay \
        --wandb_project "" \
        --spectrogram $spectrogram \
        --loss_type $loss_type \
        --lr_schedule $lr_schedule \
        --warmup_steps $warmup_steps \
        --init_identity $init_identity \
        --future_bin_idx $future_bin_idx \
        --bin_encoder $bin_encoder \
        --sample_timebin_size $sample_timebin_size \
        --use_temperature_param $use_temperature_param \
        --causal $causal \
        --learning_rate $learning_rate \
        --p_show_a_embedding $show_a_embedding \
        --p_show_b_embedding $show_b_embedding \
        --separate_unembed $separate_unembed \
        --p_masked_timebins $p_masked_timebins \
        --max_temperature_param $max_temperature_param \
        --n_epochs 100 \
        --n_layers_electrode $n_layers_electrode \
        --d_model_bin $d_model \
        --first_kernel $first_kernel &
done

wait