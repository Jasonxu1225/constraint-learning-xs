#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=collect
task_name="collect_data-highD_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python generate_data_for_constraint_inference.py -n 5 -mn train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-Sep-25-2022-07:13-seed_123 -tn PPO-Lag-highD-distance -ct no-too_closed-off_road
#python generate_data_for_constraint_inference.py -n 5 -mn train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-24-2022-00:53-seed_123 -tn PPO-Lag-highD-distance -ct no-too_closed
#python generate_data_for_constraint_inference.py -n 5 -mn train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-24-2022-00:53-seed_123 -tn PPO-Lag-highD-distance -ct no-too_closed