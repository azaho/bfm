#!/bin/bash
#SBATCH --job-name=bfm_experiment          # Name of the job
#SBATCH --ntasks=1             # 2 tasks total
#SBATCH --cpus-per-task=16    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:a100:1     # Request 1 A100 GPU
#SBATCH --mem=192G
#SBATCH -t 6:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-6      # Run 9 pairs of jobs (18 total jobs)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p yanglab # use-everything
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

# Define parameter arrays
learning_rates=(0.002 0.0014 0.001 0.0028 0.004 0.0057 0.008 0.0011)
d_models=(192) # 240 288) 
n_layers=(5) # 10 20)
weight_decays=(0.0)
batch_sizes=(100 50 200)
random_strings=("BS_1" "BS_2" "BS_3" "BS_4")

# Calculate indices for two parallel jobs
idx1=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 ))
idx2=$(( ($SLURM_ARRAY_TASK_ID-1) * 2 + 1 ))

n_lr=${#learning_rates[@]}
n_d=${#d_models[@]}
n_l=${#n_layers[@]}
n_ws=${#weight_decays[@]}
n_rs=${#random_strings[@]}
n_bs=${#batch_sizes[@]}

# Convert indices for first job
layer_idx1=$(( idx1 % n_l ))
d_idx1=$(( (idx1 / n_l) % n_d ))
lr_idx1=$(( (idx1 / (n_l * n_d)) % n_lr ))
ws_idx1=$(( (idx1 / (n_l * n_d * n_lr)) % n_ws ))
bs_idx1=$(( (idx1 / (n_l * n_d * n_lr * n_ws)) % n_bs ))
rs_idx1=$(( (idx1 / (n_l * n_d * n_lr * n_ws * n_bs)) % n_rs ))

# Convert indices for second job
layer_idx2=$(( idx2 % n_l ))
d_idx2=$(( (idx2 / n_l) % n_d ))
lr_idx2=$(( (idx2 / (n_l * n_d)) % n_lr ))
ws_idx2=$(( (idx2 / (n_l * n_d * n_lr)) % n_ws ))
bs_idx2=$(( (idx2 / (n_l * n_d * n_lr * n_ws)) % n_bs ))
rs_idx2=$(( (idx2 / (n_l * n_d * n_lr * n_ws * n_bs)) % n_rs ))

# Get parameter values for first job
LR1=${learning_rates[$lr_idx1]}
D_MODEL1=${d_models[$d_idx1]}
N_LAYERS1=${n_layers[$layer_idx1]}
WS1=${weight_decays[$ws_idx1]}
BS1=${batch_sizes[$bs_idx1]}
RS1=${random_strings[$rs_idx1]}

# Get parameter values for second job
LR2=${learning_rates[$lr_idx2]}
D_MODEL2=${d_models[$d_idx2]}
N_LAYERS2=${n_layers[$layer_idx2]}
WS2=${weight_decays[$ws_idx2]}
BS2=${batch_sizes[$bs_idx2]}
RS2=${random_strings[$rs_idx2]}

echo "Job 1 - LR: $LR1, D_MODEL: $D_MODEL1, N_LAYERS: $N_LAYERS1, WS: $WS1, BS: $BS1, RS: $RS1"
echo "Job 2 - LR: $LR2, D_MODEL: $D_MODEL2, N_LAYERS: $N_LAYERS2, WS: $WS2, BS: $BS2, RS: $RS2"

python -u train_model.py --learning_rate $LR1 --d_model $D_MODEL1 --n_layers_electrode $N_LAYERS1 --n_layers_time $N_LAYERS1 --weight_decay $WS1 --batch_size $BS1 --random_string $RS1 &
python -u train_model.py --learning_rate $LR2 --d_model $D_MODEL2 --n_layers_electrode $N_LAYERS2 --n_layers_time $N_LAYERS2 --weight_decay $WS2 --batch_size $BS2 --random_string $RS2 &
wait