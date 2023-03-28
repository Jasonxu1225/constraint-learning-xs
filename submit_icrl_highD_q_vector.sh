#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=ICRL
task_name="train-highD-ICRL_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
#conda activate galen-cr37
conda activate cn-py37
cd ./interface/
python train_icrl.py ../config/highD_velocity_distance_constraint/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20.yaml -s 123 -n 5 -l "$log_dir"
