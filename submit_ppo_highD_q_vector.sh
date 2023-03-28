#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=PPO
task_name="train-highD-PPO-1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python train_ppo.py ../config/highD_velocity_distance_constraint/train_ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20.yaml -n 5 -s 123 -l "$log_dir"