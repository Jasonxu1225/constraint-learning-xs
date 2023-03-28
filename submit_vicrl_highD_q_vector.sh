#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=VICRL
task_name="train-highD-VICRL_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate cn-py37
cd ./interface/
python train_icrl.py ../config/highD_distance_constraint/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20.yaml -s 321 -n 5 -l "$log_dir"
