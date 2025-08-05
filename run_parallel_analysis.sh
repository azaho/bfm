#!/bin/bash
#SBATCH --job-name=parallel_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -p normal
#SBATCH --requeue

source .venv/bin/activate
python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 2 --trial_id 6
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank2_6_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 3 --trial_id 0
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank3_0_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 3 --trial_id 2
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank3_2_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 4 --trial_id 0
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank4_0_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 4 --trial_id 1
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank4_1_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 4 --trial_id 2
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank4_2_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 7 --trial_id 0
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank7_0_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 7 --trial_id 1
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank7_1_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 10 --trial_id 0
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank10_0_onset_attention.npy"

python -m analyses.compute_key_electrodes --model_dir andrii0_wd0.0001_dr0.1_rTEMP --model_epoch 100 --overwrite --subject_id 10 --trial_id 1
python -m analyses.roshnipm.visualize_attention --single_file "runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/key_electrodes_btbank10_1_onset_attention.npy"