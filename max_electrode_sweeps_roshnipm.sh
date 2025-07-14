#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 16:00:00      
#SBATCH --array=1-10
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp
export CUDA_VISIBLE_DEVICES=0,1

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are dixed
train_subject_trials="btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1"
eval_subject_trials="btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1"

# these parameters are swept over
max_n_electrodes_options=(64 96 128 160 192)

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_el=${#max_n_electrodes_options[@]}
n_total=$((n_el))

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    max_n_electrodes=${max_n_electrodes_options[$((idx % n_el))]}
    random_string="roshnipm_pair_nocommon_max_n_electrodes${max_n_electrodes}"

    log_out="runs/logs/${random_string}.out"
    log_err="runs/logs/${random_string}.err"

    python -u pretrain.py  --training.setup_name roshnipm_pair_nocommon \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.batch_size 64 \
        --model.context_length 1 \
        --training.max_n_electrodes $max_n_electrodes \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        > "$log_out" 2> "$log_err" &
done

wait