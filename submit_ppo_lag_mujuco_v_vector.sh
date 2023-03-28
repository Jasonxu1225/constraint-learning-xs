#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=LAG
task_name="train-mujoco-LAG-PPO"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
export PATH=/pkgs/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/pkgs/mjpro150/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/galen/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source /pkgs/anaconda3/bin/activate
conda activate cn-py37
pip install -e ./mujuco_environment
cd ./interface/
python train_ppo.py ../config/mujoco_AntWall-v0/train_ppo_lag_AntWall-v0_nit-50-plr-1e-3.yaml -n 5 -s 123 -l "$log_dir"
python train_ppo.py ../config/mujoco_AntWall-v0/train_ppo_lag_AntWall-v0_nit-50-plr-1e-3.yaml -n 5 -s 321 -l "$log_dir"
python train_ppo.py ../config/mujoco_AntWall-v0/train_ppo_lag_AntWall-v0_nit-50-plr-1e-3.yaml -n 5 -s 456 -l "$log_dir"
python train_ppo.py ../config/mujoco_AntWall-v0/train_ppo_lag_AntWall-v0_nit-50-plr-1e-3.yaml -n 5 -s 654 -l "$log_dir"
python train_ppo.py ../config/mujoco_AntWall-v0/train_ppo_lag_AntWall-v0_nit-50-plr-1e-3.yaml -n 5 -s 666 -l "$log_dir"
echo shell finish running