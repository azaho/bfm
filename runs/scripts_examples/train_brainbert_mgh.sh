#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=128G
#SBATCH -t 6:00:00      
#SBATCH --output runs/logs/%A_%a.out
#SBATCH --error runs/logs/%A_%a.err
#SBATCH --array=1-3
#SBATCH -p normal
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

echo "Running on: $(hostname)"


n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are dixed
train_subject_trials="mgh1_2"
eval_subject_trials=""
wandb_project="BB_MGH"

# these parameters are swept over
random_string_options=("R1")
dropout_options=(0.0 0.1 0.2)
weight_decay_options=(0.0)
n_layers_options=(5 8 10)

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_rs=${#random_string_options[@]}
n_dr=${#dropout_options[@]}
n_wd=${#weight_decay_options[@]}
n_nl=${#n_layers_options[@]}

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    dropout=${dropout_options[$((idx % n_dr))]}
    random_string=${random_string_options[$((idx / n_rs % n_rs))]}
    weight_decay=${weight_decay_options[$((idx / n_rs / n_dr % n_wd))]}
    n_layers=${n_layers_options[$((idx / n_rs / n_dr / n_wd % n_nl))]}

    echo "Job $((i+1)) - RS: $random_string - Dropout: $dropout - Weight Decay: $weight_decay - N Layers: $n_layers"
    python -u pretrain.py  --training.setup_name andrii_brainbert \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.batch_size 100 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials "$eval_subject_trials" \
        --training.dropout $dropout \
        --training.weight_decay $weight_decay \
        --model.transformer.n_layers $n_layers \
        --cluster.wandb_project $wandb_project &
done

wait