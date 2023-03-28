import os

import numpy
import scipy.stats as stats

from interface.plot_results.plot_results_dirs import get_plot_results_dir
from utils.data_utils import read_running_logs


max_reward = float('inf')
min_reward = 0
# plot_key = ['reward', 'reward_valid', 'constraint']
# constraint_keys = 'constraint'

# env_id = 'HCWithPos-v0'
# max_episodes = 6000
# method_names_labels = [
#     "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3",
#     "GAIL_HCWithPos-v0_with-action",
#     "Binary_HCWithPos-v0_with-action",
#     "ICRL_Pos_with-action", ]

# env_id = 'AntWall-V0'
# max_episodes = 15000
# title = 'Blocked Ant'
# method_names_labels = [
#     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1",
#     "GAIL_AntWall-v0_with-action",
#     "Binary_AntWall-v0_with-action_nit-50",
#     "ICRL_AntWall_with-action_nit-50",
# ]

# env_id = 'InvertedPendulumWall-v0'
# max_episodes = 80000
# method_names_labels = [
#     "VICRL_PendulumWall",
#     "GAIL_PendulumWall",  # 'GAIL',
#     "Binary_PendulumWall",  # 'Binary',
#     "ICRL_Pendulum",  # 'ICRL',
# ]

# env_id = 'WalkerWithPos-v0'
# max_episodes = 40000
# method_names_labels = [
#     "VICRL_Walker-v0_p-9e-3-1e-3_cl-64-64",
#     "GAIL_Walker",  # 'GACL'
#     "Binary_Walker",  # 'Binary
#     "ICRL_Walker",  # 'ICRL',
# ]

# env_id = 'SwimmerWithPos-v0'
# max_episodes = 10000
# title = 'Blocked Swimmer'
# method_names_labels = [
#     "VICRL_SwmWithPos-v0_update_b-5e-1_piv-5",
#     "GAIL_SwmWithPos-v0",
#     "Binary_SwmWithPos-v0_update_b-5e-1",
#     "ICRL_SwmWithPos-v0_update_b-5e-1",
# ]

# plot_key = ['reward', 'reward_valid',  'is_over_speed']
# env_id = 'highD_velocity_constraint'
# max_episodes = 2000
# constraint_keys = 'is_over_speed'
#
# method_names_labels = [
#     "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40",
#     "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40",
#     "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40",
#     "ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40",
# ]

plot_key = ['reward', 'reward_valid',  'is_too_closed']
env_id = 'highD_distance_constraint'
max_episodes = 2000
constraint_keys = 'is_too_closed'
method_names_labels = [
    'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20',
    'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20',
    'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_no-buffer_dm-20',
    'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20',
]


mode = 'train'

# plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
log_path_dict = get_plot_results_dir(env_id)

target_rewards_results = []
target_constraint_results = []

for method_name in method_names_labels:
    all_pvalues_reward = []
    all_pvalues_constraint = []
    for log_path_idx in range(len(log_path_dict[method_name])):
        log_path = log_path_dict[method_name][log_path_idx]
        monitor_path_all = []
        run_files = os.listdir(log_path)
        for file in run_files:
            if 'monitor' in file:
                monitor_path_all.append(log_path + file)

        # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
        results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                   read_keys=plot_key,
                                                                   max_reward=max_reward,
                                                                   min_reward=min_reward,
                                                                   max_episodes=max_episodes + float(
                                                                       max_episodes / 5),
                                                                   constraint_keys=[constraint_keys])

        if "VICRL" in method_name:
            target_rewards_results.append(results['reward_valid'])
            target_constraint_results.append(results[constraint_keys])
            continue
        else:
            try:
                min_len = min(len(target_rewards_results[log_path_idx]), len(results['reward_valid']))
                significance_test_reward = stats.wilcoxon(target_rewards_results[log_path_idx][:min_len][-100:],
                                                          results['reward_valid'][:min_len][-100:])
                reward_pvalue = significance_test_reward.pvalue
                all_pvalues_reward.append(reward_pvalue)
            except:
                pass
            try:
                significance_test_constraint = stats.wilcoxon(target_constraint_results[log_path_idx][:min_len][-100:],
                                                              results[constraint_keys][:min_len][-100:])
                constraint_pvalue = significance_test_constraint.pvalue
                all_pvalues_constraint.append(constraint_pvalue)
            except:
                pass
    print("{0}, reward {1}, Constraint {2}".format(method_name,
          numpy.mean(all_pvalues_reward),
          numpy.mean(all_pvalues_constraint)))
