#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 16:00:00      
#SBATCH --array=1-16
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp
export CUDA_VISIBLE_DEVICES=0,1

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are dixed
train_subject_trials="btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1"
eval_subject_trials="btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1"

# these parameters are swept over
dropout_options=(0.0 0.1 0.2 0.3)
weight_decay_options=(0.0 0.001 0.01 0.1)

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_dr=${#dropout_options[@]}
n_wd=${#weight_decay_options[@]}
n_total=$((n_dr * n_wd))

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    dropout=${dropout_options[$((idx % n_dr))]}
    weight_decay=${weight_decay_options[$((idx / n_dr))]}
    random_string="andrii0_dropout${dropout}_wd${weight_decay}"

    log_out="runs/logs/${random_string}.out"
    log_err="runs/logs/${random_string}.err"

    python -u pretrain.py  --training.setup_name andrii0 \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        --training.dropout $dropout \
        --training.weight_decay $weight_decay \
        > "$log_out" 2> "$log_err" &
done

wait