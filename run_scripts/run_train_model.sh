#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 16:00:00      
#SBATCH --array=1-3
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

n_in_parallel=1

train_subject_trial_options=(
    "btbank1_0,btbank2_1,btbank2_2,btbank2_3,btbank2_5,btbank2_6,btbank3_2,btbank4_2,btbank5_0,btbank6_0,btbank6_1,btbank6_4,btbank8_0,btbank9_0,btbank7_0,btbank10_0"
)
eval_subject_trials="btbank1_1,btbank1_2,btbank2_0,btbank2_4,btbank3_0,btbank3_1,btbank4_0,btbank4_1,btbank7_0,btbank7_1,btbank10_0,btbank10_1"
random_string_options=("T")

dropout_options=(0.0 0.1 0.2)

wandb_project="OW"

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))

n_ts=${#train_subject_trial_options[@]}
n_rs=${#random_string_options[@]}
n_dr=${#dropout_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    
    dropout=${dropout_options[$((idx % n_dr))]}
    train_subject_trials=${train_subject_trial_options[$((idx / n_ts % n_ts))]}
    random_string=${random_string_options[$((idx / n_ts % n_rs))]}

    echo "Job $((i+1)) - RS: $random_string - TS: $train_subject_trials - Dropout: $dropout"

    # note: change train_model_fbi_combined.py to train_model.py for the non-combined version
    python -u train_model.py  --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.batch_size 100 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        --model.transformer.dropout $dropout \
        --cluster.wandb_project $wandb_project &
done

wait