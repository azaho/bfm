#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 16:00:00      
#SBATCH --output runs/logs/%A_%a.out
#SBATCH --error runs/logs/%A_%a.err
#SBATCH --array=1-12
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are dixed
train_subject_trials="btbank1_0,btbank2_1,btbank2_2,btbank2_3,btbank2_5,btbank2_6,btbank3_2,btbank4_2,btbank5_0,btbank6_0,btbank6_1,btbank6_4,btbank8_0,btbank9_0,btbank1_1,btbank1_2,btbank2_0,btbank2_4,btbank3_0,btbank3_1,btbank4_0,btbank4_1,btbank7_0,btbank7_1,btbank10_0,btbank10_1"
#eval_subject_trials="btbank1_1,btbank1_2,btbank2_0,btbank2_4,btbank3_0,btbank3_1,btbank4_0,btbank4_1,btbank7_0,btbank7_1,btbank10_0,btbank10_1"
eval_subject_trials="btbank1_1,btbank2_0,btbank3_0,btbank4_0,btbank7_0,btbank10_0" # just half, for speed
wandb_project="WANDB_PROJECT_NAME"

# these parameters are swept over
random_string_options=("R1" "R2")
dropout_options=(0.0 0.1 0.2)
weight_decay_options=(0.0 0.001)

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_rs=${#random_string_options[@]}
n_dr=${#dropout_options[@]}
n_wd=${#weight_decay_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    dropout=${dropout_options[$((idx % n_dr))]}
    random_string=${random_string_options[$((idx / n_rs % n_rs))]}
    weight_decay=${weight_decay_options[$((idx / n_rs / n_dr % n_wd))]}

    echo "Job $((i+1)) - RS: $random_string - Dropout: $dropout - Weight Decay: $weight_decay"
    python -u pretrain.py  --training.setup_name andrii0 \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.batch_size 100 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        --training.dropout $dropout \
        --training.weight_decay $weight_decay \
        --cluster.wandb_project $wandb_project &
done

wait