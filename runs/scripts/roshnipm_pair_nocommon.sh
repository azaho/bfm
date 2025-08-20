#!/bin/bash
#SBATCH --job-name=roshnipm_pair_nocommon      # custom name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1        # request 2 A100 GPUs
#SBATCH --mem=100G
#SBATCH --time=48:00:00         # 48‑hour runtime
#SBATCH --output=runs/logs/%j.out
#SBATCH --error=runs/logs/%j.err
#SBATCH -p normal

source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

dropout=0.2
weight_decay=0.01
random_string="bestpaired_$(date +%s)"
model_name="roshnipm_pair_nocommon"

# Run your training script
python -u pretrain.py  --training.setup_name $model_name \
    --cluster.cache_subjects 1 \
    --cluster.num_workers_dataloaders 4 \
    --training.max_n_electrodes 64 \
    --training.n_epochs 100 \
    --training.batch_size 32 \
    --training.train_subject_trials btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1 \
    --training.eval_subject_trials btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1 \
    --model.context_length 8 \
    --training.future_bin_idx 0 \
    --cluster.eval_model_every_n_epochs 5 \
    --training.random_string $random_string \
    --training.dropout $dropout \
    --training.weight_decay $weight_decay \
    --training.eval_tasks onset,speech,gpt2_surprisal \
    --cluster.wandb_project roshnipm \
    --cluster.wandb_entity andrii-mit \
    > "$log_out" 2> "$log_err" &