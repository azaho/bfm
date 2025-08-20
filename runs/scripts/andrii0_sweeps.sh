#!/bin/bash
#SBATCH --job-name=bfm_xx          
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
####SBATCH --constraint=ampere
#SBATCH --mem=384G
#SBATCH -t 48:00:00      
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

# these parameters are dixed
# train on Mr. Fox, Cars 2, and Megamind
train_subject_trials="btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1,btbank6_0,btbank1_1,btbank5_0"
# eval on Cars 2 and one Megamind pair (Neuroprobe lite limitations)
eval_subject_trials="btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1"

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
    random_string="andrii0_dropout${dropout}_wd${weight_decay}"

    log_out="runs/logs/${random_string}.out"
    log_err="runs/logs/${random_string}.err"

    # Store the expected wandb run directory name for this run
    wandb_run_dirs+=("runs/wandb/wandb/offline-run-*-${random_string}")

    python -u pretrain.py  --training.setup_name andrii0 \
        --cluster.cache_subjects 1 \
        --cluster.num_workers_dataloaders 4 \
        --training.max_n_electrodes 128 \
        --training.n_epochs 100 \
        --cluster.eval_model_every_n_epochs 5 \
        --training.random_string $random_string \
        --training.train_subject_trials $train_subject_trials \
        --training.eval_subject_trials $eval_subject_trials \
        --training.dropout $dropout \
        --training.weight_decay $weight_decay \
        --cluster.wandb_project roshnipm \
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