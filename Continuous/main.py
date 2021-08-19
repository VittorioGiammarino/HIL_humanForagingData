#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:55:31 2021

@author: vittorio
"""

import numpy as np
import torch
import gym
import argparse
import os

from tensorflow import keras
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
import multiprocessing.pool

from evaluation import HierarchicalStochasticSampleTrajMDP
from utils_main import Show_DataSet
import World
import BatchBW_HIL_pytorch
import Behavioral_cloning

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
Folders = [6] #, 7, 11, 12, 15]

TrainingSet, Labels, Trajectories, Rotation, Time, _, _, _, _ = Show_DataSet(Folders, 'distr_only')
_,_,_,_,_, Reward_eval_human, Real_Traj_eval_human, Real_Reward_eval_human, Real_Time_eval_human = Show_DataSet(Folders, 'full_coins')

# %%
threshold = np.mean(Real_Reward_eval_human)
Rand_traj = 2

nTraj = Rand_traj%10
folder = Folders[int(Rand_traj/10)]
coins_location = World.Foraging.CoinLocation(folder, nTraj+1, 'full_coins')

max_epoch_per_traj = len(Trajectories[Rand_traj])
action_samples = Rotation[Rand_traj][:,0].reshape(len(Rotation[Rand_traj][:,0]),1)
env = World.Foraging.env(folder, nTraj)

# def test(labels, env, max_epoch_per_traj, number_of_trajectories, reset = 'random', initial_state = np.array([0,0,0,8])):
#     traj = [[None]*1 for _ in range(number_of_trajectories)]
#     control = [[None]*1 for _ in range(number_of_trajectories)]
#     Reward_array = np.empty((0,0),int)

#     for t in range(number_of_trajectories):
#         done = False
#         obs = env.reset(reset, initial_state)
#         size_input = len(obs)
#         x = np.empty((0,size_input),int)
#         x = np.append(x, obs.reshape((1,size_input)), axis=0)
#         u_tot = np.empty((0,1))
#         Reward = 0
        
#         for k in range(0, max_epoch_per_traj):
#             # given action, draw next state
#             obs, reward, done, _ = env.step(labels[k,:])
#             Reward = Reward + reward
#             x = np.append(x, obs.reshape((1,size_input)), axis=0)

#             if done == True:
#                 break

#         traj[t] = x
#         control[t]=u_tot
#         Reward_array = np.append(Reward_array, Reward)

#     return traj, control, Reward_array    


# traj, control, Reward_array = test(action_samples, env, max_epoch_per_traj, 1, reset = 'standard', initial_state = Trajectories[Rand_traj][0,:])
    

# sigma1 = 0.5
# circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
# sigma2 = 1.1
# circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
# sigma3 = 1.8
# circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
# sigma4 = 1.3
# circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
# fig, ax = plt.subplots()
# ax.add_artist(circle1)
# ax.add_artist(circle2)
# ax.add_artist(circle3)
# ax.add_artist(circle4)
# plot_data = plt.scatter(traj[0][:,0], traj[0][:,1], c=Time[Rand_traj][:], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
# plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
# cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
# cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Processed Human data, Reward {}'.format(Reward_eval_human[Rand_traj]))
# # plt.savefig('Figures/FiguresExpert/Processed_human_traj.eps', format='eps')
# plt.show()  

time = np.linspace(0,480,len(Real_Traj_eval_human[Rand_traj][:,0])) 

sigma1 = 0.5
circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(0.1*Real_Traj_eval_human[Rand_traj][:,0], 0.1*Real_Traj_eval_human[Rand_traj][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real Human Data, Reward {}'.format(Real_Reward_eval_human[Rand_traj]))
# plt.savefig('Figures/FiguresExpert/Real_human_traj{}.eps'.format(Rand_traj), format='eps')
plt.show() 

sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig3, ax3 = plt.subplots()
plot_data = plt.scatter(Trajectories[Rand_traj][:,0], Trajectories[Rand_traj][:,1], c=Trajectories[Rand_traj][:,2], marker='o', cmap='bwr')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig3.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['no coins', 'see coins'])
ax3.add_artist(circle1)
ax3.add_artist(circle2)
ax3.add_artist(circle3)
ax3.add_artist(circle4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Options for supervised learning')
# plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
plt.show()   


fig4, ax4 = plt.subplots()
plot_data = plt.plot(Time[Rand_traj][:-1], Rotation[Rand_traj][:,0])
plot_data = plt.plot(Time[Rand_traj][:-1], Trajectories[Rand_traj][:,2])
plt.xlabel('time')
plt.ylabel('actions')
# plt.title('Options for supervised learning')
# plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
plt.show()   

fig5, ax5 = plt.subplots()
plot_data = plt.plot(Time[Rand_traj][:-1], Rotation[Rand_traj][:,0])
plt.xlabel('time')
plt.ylabel('actions')
# plt.title('Options for supervised learning')
# plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
plt.show()   

# %% HIL

parser = argparse.ArgumentParser()
parser.add_argument("--number_options", default=2, type=int)     # number of options
parser.add_argument("--policy", default="TD3")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

env = World.Foraging.env(folder, nTraj)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space[0]
action_dim = env.action_space.shape()[0] 
max_action = env.action_space.high()
option_dim = args.number_options
termination_dim = 2
state_samples = Trajectories[Rand_traj][:,:]
action_samples = Rotation[Rand_traj][:,0].reshape(len(Rotation[Rand_traj][:,0]),1)
batch_size = 3200
M_step_epochs = 100
l_rate_pi_lo = 0.001
Agent_continuous_BatchHIL = BatchBW_HIL_pytorch.BatchBW(max_action, state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, M_step_epochs, batch_size, l_rate_pi_lo, l_rate_pi_lo, l_rate_pi_lo)
N=10
eval_episodes = 10
max_epoch = len(state_samples)

# %%
epochs = 200
Options = state_samples[:,2]
Agent_continuous_BatchHIL.pretrain_pi_hi(epochs, Options)
Labels_b = Agent_continuous_BatchHIL.prepare_labels_pretrain_pi_b(Options)
Agent_continuous_BatchHIL.pretrain_pi_b(epochs, Labels_b[0], 0)
Agent_continuous_BatchHIL.pretrain_pi_b(epochs, Labels_b[1], 1)

# %%
Loss = 100000
avg_reward_torch = []
for i in range(N):
    print(f"Iteration {i+1}/{N}")
    loss = Agent_continuous_BatchHIL.Baum_Welch()
    if loss > Loss:
        Agent_continuous_BatchHIL.reset_learning_rates(l_rate_pi_lo/2, l_rate_pi_lo/2, l_rate_pi_lo/2)
    Loss = loss
    [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
     TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_continuous_BatchHIL, env, max_epoch, eval_episodes, 'standard', state_samples[0,:])
    avg_reward = np.sum(RewardBatch_torch)/eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")


# %% Behavioral Cloning attempt

state_samples = Trajectories[Rand_traj]
action_samples = Rotation[Rand_traj][:,0].reshape(len(Rotation[Rand_traj][:,0]),1)
batch_size = 3200
M_step_epochs = 10
l_rate_pi_lo = 0.001
Agent_continuous_BC = Behavioral_cloning.TanhGaussianActor.BehavioralCloning(max_action, state_dim, action_dim, state_samples, action_samples, M_step_epochs, batch_size, l_rate_pi_lo)
N=50
eval_episodes = 10
max_epoch = len(state_samples)

Loss = 100000
avg_reward_torch = []
for i in range(N):
    print(f"Iteration {i+1}/{N}")
    loss = Agent_continuous_BC.train()
    if loss > Loss:
        Agent_continuous_BC.reset_learning_rate(l_rate_pi_lo/2)
    Loss = loss
    [trajBatch_torch, controlBatch_torch, RewardBatch_torch] = Agent_continuous_BC.StochasticSampleTrajMDP(env, max_epoch, eval_episodes, 'standard', state_samples[0,:])
    avg_reward = np.sum(RewardBatch_torch)/eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")


# %%
[trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
  TerminationBatch_torch, RewardBatch_torch] = Agent_continuous_BatchHIL.HierarchicalStochasticSampleTrajMDP(env, max_epoch, eval_episodes, 'standard', state_samples[0,:])

# %%
coins_location = World.Foraging.CoinLocation(folder, nTraj+1, 'full_coins')
time = np.linspace(0,480,len(trajBatch_torch[0][:,0])) 

index = 0

sigma1 = 0.5
circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(trajBatch_torch[index][:,0], trajBatch_torch[index][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('simulated Traj, Reward {}'.format(RewardBatch_torch[index]))
# plt.savefig('Figures/FiguresExpert/Processed_human_traj.eps', format='eps')
plt.show() 






