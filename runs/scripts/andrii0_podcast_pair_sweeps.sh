#!/bin/bash
#SBATCH --job-name=bfm_podcast_pair_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
####SBATCH --constraint=ampere
#SBATCH --mem=32G
#SBATCH -t 12:00:00      
#SBATCH --array=1-7
#SBATCH -p normal
#SBATCH --requeue
#SBATCH --exclude=node100
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are fixed
# train on subjects 1-6 (will create all possible pairs automatically)
train_subject_trials="podcast01_0,podcast02_0,podcast03_0,podcast04_0,podcast05_0,podcast06_0,podcast07_0,podcast08_0,podcast09_0"
# eval on subjects 7-9 (will create all possible pairs automatically)
eval_subject_trials="podcast07_0,podcast08_0,podcast09_0"
model_name="andrii0_podcast_pair_bert"

# these parameters are swept over - increased regularization for overfitting
cl_options=(1 2 3 4 8 12 16)

# Fixed hyperparameters
dropout=0.2
weight_decay=0.01

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_cl=${#cl_options[@]}

# Track wandb run directories for syncing
wandb_run_dirs=()

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    cl=${cl_options[$((idx % n_cl))]}
    random_string="cl${cl}_402040_$(date +%s)"

    log_out="runs/logs/${model_name}_cl${cl}_402040.out"
    log_err="runs/logs/${model_name}_cl${cl}_402040.err"

    # Store the expected wandb run directory name for this run
    wandb_run_dirs+=("runs/wandb/wandb/offline-run-*-${model_name}_wd${weight_decay}_dr${dropout}_r${random_string}")

    python -u pretrain.py  --training.setup_name $model_name \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.max_n_electrodes 117 \
        --training.n_epochs 200 \
        --training.batch_size 32 \
        --training.p_test 0.3 \
        --model.context_length $cl \
        --training.future_bin_idx 0 \
        --cluster.eval_model_every_n_epochs 5 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        --training.dropout $dropout \
        --training.weight_decay $weight_decay \
        --training.eval_tasks "" \
        --cluster.wandb_project podcast \
        --cluster.wandb_entity andrii-mit \
        > "$log_out" 2> "$log_err" &
done

wait

# Automatically sync the runs that were just created
echo "Syncing wandb runs..."
for pattern in "${wandb_run_dirs[@]}"; do
    for run_dir in $pattern; do
        if [ -d "$run_dir" ]; then
            echo "Syncing $run_dir"
            wandb sync "$run_dir"
        fi
    done
done 