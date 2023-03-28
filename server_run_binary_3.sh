#!/bin/bash
task_name="train-binary_3"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-server-${task_name}-${launch_time}.out"
source /data/Galen/miniconda3-4.12.0/bin/activate
source activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
export CUDA_VISIBLE_DEVICES=3
#python train_icrl.py ../config/mujuco_HCWithPos-v0/train_Binary_HCWithPos-v0_with-action_sub-1.yaml -n 5 -s 123 -l "$log_dir"
#python train_icrl.py ../config/mujuco_HCWithPos-v0/train_Binary_HCWithPos-v0_with-action_sub-1.yaml -n 5 -s 321 -l "$log_dir"
#python train_icrl.py ../config/mujuco_HCWithPos-v0/train_Binary_HCWithPos-v0_with-action_sub-1.yaml -n 5 -s 456 -l "$log_dir"
#python train_icrl.py ../config/mujuco_HCWithPos-v0/train_Binary_HCWithPos-v0_with-action_sub-1.yaml -n 5 -s 654 -l "$log_dir"
#python train_icrl.py ../config/mujuco_HCWithPos-v0/train_Binary_HCWithPos-v0_with-action_sub-1.yaml -n 5 -s 666 -l "$log_dir"
python train_icrl.py ../config/highD_velocity_constraint/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1.yaml -n 5 -s 123 -l "$log_dir"
python train_icrl.py ../config/highD_velocity_constraint/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1.yaml -n 5 -s 321 -l "$log_dir"
python train_icrl.py ../config/highD_velocity_constraint/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1.yaml -n 5 -s 666 -l "$log_dir"
cd ../