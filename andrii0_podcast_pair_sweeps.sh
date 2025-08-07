#!/bin/bash
#SBATCH --job-name=bfm_podcast_pair_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 12:00:00      
#SBATCH --array=1-16
#SBATCH -p normal
#SBATCH --requeue
#SBATCH --exclude=node100
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Check for at least 40 GiB free on the assigned GPU
REQUIRED_MEM=40960  # 40 GiB in MiB
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")

if [ "$FREE_MEM" -lt "$REQUIRED_MEM" ]; then
    echo "Not enough free GPU memory on GPU $GPU_ID: ${FREE_MEM} MiB available, ${REQUIRED_MEM} MiB required. Exiting."
    exit 1
fi

n_in_parallel=1 # How many jobs to run in parallel on the same job (on the same GPU!)

# these parameters are fixed
# train on subjects 1-6 (will create all possible pairs automatically)
train_subject_trials="podcast01_0,podcast02_0,podcast03_0,podcast04_0,podcast05_0,podcast06_0,podcast07_0,podcast08_0,podcast09_0"
# eval on subjects 7-9 (will create all possible pairs automatically)
eval_subject_trials="podcast07_0,podcast08_0,podcast09_0"
model_name="andrii0_podcast_pair"

# these parameters are swept over
dropout_options=(0.0 0.1 0.2 0.3)
weight_decay_options=(0.0 0.001 0.01 0.1)

# Calculate indices for parallel jobs
base_idx=$(( ($SLURM_ARRAY_TASK_ID-1) * n_in_parallel ))
n_dr=${#dropout_options[@]}
n_wd=${#weight_decay_options[@]}
n_total=$((n_dr * n_wd))

# Track wandb run directories for syncing
wandb_run_dirs=()

# Launch n_in_parallel jobs
for i in $(seq 0 $(( n_in_parallel - 1 ))); do
    idx=$(( base_idx + i ))
    
    # Convert index to parameter selections
    dropout=${dropout_options[$((idx % n_dr))]}
    weight_decay=${weight_decay_options[$((idx / n_dr))]}
    random_string="modified_$(date +%s)"

    log_out="runs/logs/${model_name}_modified_wd${weight_decay}_dr${dropout}.out"
    log_err="runs/logs/${model_name}_modified_wd${weight_decay}_dr${dropout}.err"

    # Store the expected wandb run directory name for this run
    wandb_run_dirs+=("runs/wandb/wandb/offline-run-*-${model_name}_wd${weight_decay}_dr${dropout}_r${random_string}")

    python -u pretrain.py  --training.setup_name $model_name \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.max_n_electrodes 117 \
        --training.n_epochs 200 \
        --training.batch_size 64 \
        --training.p_test 0.2 \
        --model.context_length 2 \
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