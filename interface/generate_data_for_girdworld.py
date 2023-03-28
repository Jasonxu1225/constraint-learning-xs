import math
import os
import random
import numpy as np
import pickle as pkl
import yaml
from tqdm import tqdm

# -----------------------
# Main
# -----------------------
import gym
# import mujuco_environment.custom_envs

config_path = '../mujuco_environment/custom_envs/envs/configs/WGW-setting1.yaml'
with open(config_path, "r") as config_file:
    env_configs = yaml.safe_load(config_file)
env = gym.make(id='WGW-v0', **env_configs)
indices = 0

actions = [
    [0, 0, 4, 4, 4, 6, 6, 6, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2],
]

actions_percentage = {
    'v1': [0.5, 0.5],
    'v2': [0.9, 0.1],
    'v3': [0.1, 0.9],
}

actions_version = 'v1'
trajectory_numbers = 50

save_gridworld_data_path = '../data/expert_data/TestWallGridWorld-{0}/'.format(actions_version)
if not os.path.exists(save_gridworld_data_path):
    os.mkdir(save_gridworld_data_path)

choice_indices = [i for i in range(len(actions))]
indices = random.choices(choice_indices, weights=actions_percentage[actions_version], k=trajectory_numbers)
for j in tqdm(range(trajectory_numbers)):  # default 50
    state = env.reset()
    s_traj = []
    a_traj = []
    r_traj = []
    action_num = 0
    while True:
        action = actions[indices[j]][action_num]
        action_num += 1
        s_traj.append(state)
        a_traj.append(action)
        r_traj.append(0)

        state, reward, done, info = env.step(action)
        # env.render()

        if done:
            print(str(a_traj))
            s_traj = np.array(s_traj, dtype=np.float32)
            a_traj = np.array(a_traj, dtype=np.float32)
            break
    pkl.dump({'original_observations': s_traj,
              'actions': a_traj,
              'reward': r_traj,
              'reward_sum': 0,
              },
             open(save_gridworld_data_path + "scene_{0}_len-{1}.pkl".format(j, len(s_traj)), "wb"))
