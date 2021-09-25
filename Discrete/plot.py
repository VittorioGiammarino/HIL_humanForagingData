#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:36:58 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %% Load Data

TrainingSet_tot = np.load("./Expert_data/TrainingSet.npy")
Labels_tot = np.load("./Expert_data/Labels.npy")
Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
Time = np.load("./Expert_data/Time.npy", allow_pickle=True).tolist()
Reward_eval_human = np.load("./Expert_data/Reward_eval_human.npy")
Real_Traj_eval_human = np.load("./Expert_data/Real_Traj_eval_human.npy", allow_pickle=True).tolist()
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()
Real_Time_eval_human = np.load("./Expert_data/Real_Time_eval_human.npy", allow_pickle=True).tolist()
Coins_location = np.load("./Expert_data/Coins_location.npy")
    
threshold = np.mean(Real_Reward_eval_human)
Rand_traj = 2
TrainingSet = Trajectories[Rand_traj]
Labels = Rotation[Rand_traj]
size_data = len(Trajectories[Rand_traj])
coins_location = Coins_location[Rand_traj,:,:] 

len_trajs = []
for i in range(len(Trajectories)):
    len_trajs.append(len(Trajectories[i]))
    
mean_len_trajs = int(np.mean(len_trajs))

# %%

PPO_IL = []
TRPO_IL = []
UATRPO_IL = []
HPPO_IL = []
for i in range(8):
    with open(f'results/FlatRL/evaluation_PPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        PPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/FlatRL/evaluation_TRPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        TRPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/FlatRL/evaluation_UATRPO_IL_True_GAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
        UATRPO_IL.append(np.load(f, allow_pickle=True))
        
    # with open(f'results/HRL/evaluation_HPPO_HIL_True_HGAIL_False_Mixed_False_Foraging_{i}.npy', 'rb') as f:
    #     HPPO_IL.append(np.load(f, allow_pickle=True))
        
    with open(f'results/HRL/evaluation_HPPO_HIL_True_delayed_4_Foraging_{i}.npy', 'rb') as f:
        HPPO_IL.append(np.load(f, allow_pickle=True))
            
Real_Reward_eval_human = np.load("./Expert_data/Real_Reward_eval_human.npy", allow_pickle=True).tolist()    
threshold = np.mean(Real_Reward_eval_human)

    
# %%

steps = np.linspace(0,6e6,len(PPO_IL[0]))
Human_average_performance = threshold*np.ones((len(steps),))

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, PPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, PPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, PPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, PPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, PPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, PPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, PPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('PPO')
plt.savefig('Figures/PPO.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, HPPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, HPPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, HPPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, HPPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, HPPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, HPPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, HPPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, HPPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('HPPO')
plt.savefig('Figures/HPPO.pdf', format='pdf')

# %%

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, TRPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, TRPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, TRPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, TRPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, TRPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, TRPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, TRPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, TRPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('TRPO')
plt.savefig('Figures/TRPO.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, UATRPO_IL[0], label='seed 0', c=clrs[0])
ax.plot(steps, UATRPO_IL[1], label='seed 1', c=clrs[1])
ax.plot(steps, UATRPO_IL[2], label='seed 2', c=clrs[2])
ax.plot(steps, UATRPO_IL[3], label='seed 3', c=clrs[3])
ax.plot(steps, UATRPO_IL[4], label='seed 4', c=clrs[4])
ax.plot(steps, UATRPO_IL[5], label='seed 5', c=clrs[5])
ax.plot(steps, UATRPO_IL[6], label='seed 6', c=clrs[6])
ax.plot(steps, UATRPO_IL[7], label='seed 7', c=clrs[7])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[8])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('UATRPO')
plt.savefig('Figures/UATRPO.pdf', format='pdf')

# %%

PPO_mean = np.mean(np.array(PPO_IL),0)
PPO_std = np.std(np.array(PPO_IL),0)

HPPO_mean = np.mean(np.array(HPPO_IL),0)
HPPO_std = np.std(np.array(HPPO_IL),0)

TRPO_mean = np.mean(np.array(TRPO_IL),0)
TRPO_std = np.std(np.array(TRPO_IL),0)

UATRPO_mean = np.mean(np.array(UATRPO_IL),0)
UATRPO_std = np.std(np.array(UATRPO_IL),0)

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_mean, label='PPO', c=clrs[0])
ax.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[0])
ax.plot(steps, HPPO_mean, label='HPPO', c=clrs[4])
ax.fill_between(steps, HPPO_mean-HPPO_std, HPPO_mean+HPPO_std, alpha=0.2, facecolor=clrs[4])
ax.plot(steps, TRPO_mean, label='TRPO', c=clrs[1])
ax.fill_between(steps, TRPO_mean-TRPO_std, TRPO_mean+TRPO_std, alpha=0.2, facecolor=clrs[1])
ax.plot(steps, UATRPO_mean, label='UATRPO', c=clrs[3])
ax.fill_between(steps, UATRPO_mean-UATRPO_std, UATRPO_mean+UATRPO_std, alpha=0.2, facecolor=clrs[3])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[2])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Comparison')
plt.savefig('Figures/on_policy_comparison.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 9)
ax.plot(steps, PPO_mean, label='PPO', c=clrs[0])
ax.fill_between(steps, PPO_mean-PPO_std, PPO_mean+PPO_std, alpha=0.2, facecolor=clrs[0])
ax.plot(steps, HPPO_mean, label='HPPO', c=clrs[4])
ax.fill_between(steps, HPPO_mean-HPPO_std, HPPO_mean+HPPO_std, alpha=0.2, facecolor=clrs[4])
# ax.plot(steps, TRPO_mean, label='TRPO', c=clrs[1])
# ax.fill_between(steps, TRPO_mean-TRPO_std, TRPO_mean+TRPO_std, alpha=0.2, facecolor=clrs[1])
# ax.plot(steps, UATRPO_mean, label='UATRPO', c=clrs[3])
# ax.fill_between(steps, UATRPO_mean-UATRPO_std, UATRPO_mean+UATRPO_std, alpha=0.2, facecolor=clrs[3])
ax.plot(steps, Human_average_performance, "--", label='Humans', c=clrs[2])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Comparison')
plt.savefig('Figures/PPO_vs_HPPO.pdf', format='pdf')

# %%

