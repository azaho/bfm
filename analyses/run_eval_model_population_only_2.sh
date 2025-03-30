#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 3:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-5      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p yanglab

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Create arrays of subject IDs and trial IDs that correspond to array task ID
# for all subject_trials in the dataset
declare -a subjects=(1 2 3 7 10)
declare -a trials=(2 6 0 0 0)
declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "global_flow_angle"
    "local_flow_angle" 
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "delta_pitch"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
    "speaker"
)

# Calculate indices for this task
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % 5 ))

# Get subject, trial and eval name for this task
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]} 

EVAL_NAMES_STR=$(IFS=,; echo "${eval_names[*]}")
echo ""
echo ""
echo "--------------------------------"
echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, all evals"
python -u analyses/btbench_eval_population.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_names $EVAL_NAMES_STR --verbose --model_dir M_nst8_dm192_nh12_nl5_5_stbs0.03125_mxt96_mbfNone_lr0.003_rMTB3 --overwrite

# M_nst8_dm192_nh12_nl5_5_lr0.003_rSL1
# M_nst8_dm192_nh12_nl5_5_stbs0.03125_mxt96_mbfNone_lr0.003_rMTB3
# M_nst8_dm192_nh12_nl5_5_stbs0.0625_mxt48_mbfNone_lr0.003_rMTB3