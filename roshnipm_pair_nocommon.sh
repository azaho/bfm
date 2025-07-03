#!/bin/bash
#SBATCH --job-name=roshnipm_pair_nocommon      # custom name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2        # request 2 A100 GPUs
#SBATCH --mem=384G
#SBATCH --time=48:00:00         # 48‑hour runtime
#SBATCH --output=runs/logs/%j.out
#SBATCH --error=runs/logs/%j.err
#SBATCH -p normal

source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

# Run your training script
python -u pretrain.py \
  --training.setup_name roshnipm_pair_nocommon \
  --cluster.cache_subjects 1 \
  --cluster.eval_at_beginning 1 \
  --training.train_subject_trials btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1 \
  --training.eval_subject_trials btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1 \
  --training.max_n_electrodes 64 \
  --cluster.eval_model_every_n_epochs 5 \
  --training.eval_tasks speech,gpt2_surprisal \
  --training.n_epochs 500
