#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
import torch
import argparse
import os

from tensorflow import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
import H_SAC
import H_TD0
import multiprocessing
import multiprocessing.pool

from utils import Encode_Data
from BatchBW_HIL_torch import BatchBW
from evaluation import HierarchicalStochasticSampleTrajMDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %% HIL
parser = argparse.ArgumentParser()
#General
parser.add_argument("--number_options", default=2, type=int)     # number of options
parser.add_argument("--policy", default="HSAC")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--seed", default=3, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
#HIL
parser.add_argument("--HIL", default=True, type=bool)         # Batch size for HIL
parser.add_argument("--size_data_set", default=size_data, type=int)         # Batch size for HIL
parser.add_argument("--batch_size_HIL", default=32, type=int)         # Batch size for HIL
parser.add_argument("--maximization_epochs_HIL", default=10, type=int) # Optimization epochs HIL
parser.add_argument("--l_rate_HIL", default=0.001, type=float)         # Optimization epochs HIL
parser.add_argument("--N_iterations", default=11, type=int)            # Number of EM iterations
parser.add_argument("--pi_hi_supervised", default=True, type=bool)     # Supervised pi_hi
parser.add_argument("--pi_hi_supervised_epochs", default=200, type=int)  
#HTD0
parser.add_argument("--init_critic", default=True, type=bool)   
parser.add_argument("--HTD0_timesteps", default=3e5, type=int)    # Max time steps to run environment
# HRL
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
parser.add_argument("--eval_freq", default=15e3, type=int)        # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=3e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--alpha", default=0.2, type=int)            # SAC entropy regularizer term
parser.add_argument("--critic_freq", default=2, type=int)        # Frequency of delayed critic updates
parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--load_model_path", default="") 
# Evaluation
parser.add_argument("--evaluation_episodes", default=10, type=int)
parser.add_argument("--evaluation_max_n_steps", default=size_data, type=int)
args = parser.parse_args()

file_name = f"{args.policy}_HIL_{args.HIL}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")
   
if not os.path.exists("./results"):
    os.makedirs("./results")
   
if not os.path.exists("./models"):
    os.makedirs("./models")
    
if not os.path.exists(f"./models/{file_name}"):
    os.makedirs(f"./models/{file_name}")
    
if not os.path.exists("./models/HIL"):
    os.makedirs("./models/HIL")
    
if not os.path.exists("./models/H_TD0"):
    os.makedirs("./models/H_TD0")

env = World.Foraging.env(coins_location)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)

# %%

state_dim = state_samples.shape[1] # encoded state dim
action_dim = len(np.unique(action_samples)) # number of possible actions
option_dim = args.number_options
termination_dim = 2
batch_size = args.batch_size_HIL
M_step_epochs = args.maximization_epochs_HIL
l_rate = args.l_rate_HIL

kwargs = {
	"state_dim": state_dim,
    "action_dim": action_dim,
    "option_dim": option_dim,
    "termination_dim": termination_dim,
    "state_samples": state_samples,
    "action_samples": action_samples,
    "M_step_epoch": M_step_epochs,
    "batch_size": batch_size,
    "l_rate": l_rate,
    "encoding_info": encoding_info
    }

Agent_BatchHIL_torch = BatchBW(**kwargs)
N = args.N_iterations
eval_episodes = args.evaluation_episodes
max_epoch = args.evaluation_max_n_steps

# %%

if args.HIL:
    if args.pi_hi_supervised:
        epochs = args.pi_hi_supervised_epochs
        Options = state_samples[:,2]
        Agent_BatchHIL_torch.pretrain_pi_hi(epochs, Options)
        Labels_b = Agent_BatchHIL_torch.prepare_labels_pretrain_pi_b(Options)
        Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[0], 0)
        Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[1], 1)

    Loss = 100000
    evaluation_HIL = []
    for i in range(N):
        print(f"Iteration {i+1}/{N}")
        loss = Agent_BatchHIL_torch.Baum_Welch()
        if loss > Loss:
            Agent_BatchHIL_torch.reset_learning_rate(l_rate/10)
        Loss = loss
        [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
         TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_BatchHIL_torch, env, max_epoch, eval_episodes, 'standard', TrainingSet[0,:])
        avg_reward = np.sum(RewardBatch_torch)/eval_episodes
        evaluation_HIL.append(avg_reward)
        
        print("---------------------------------------")
        print(f"Torch, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        
    # Save
    np.save(f"./results/HIL_{args.env}_{args.seed}", evaluation_HIL)
    Agent_BatchHIL_torch.save(f"./models/HIL/HIL_{args.env}_{args.seed}")

# %%

time = np.linspace(0,480,len(trajBatch_torch[0][:,0])) 

index = np.argmax(RewardBatch_torch)

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
plt.title('HIL, Reward {}'.format(RewardBatch_torch[index]))
# plt.savefig('Figures/FiguresExpert/Processed_human_traj.eps', format='eps')
plt.show() 


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
plot_data = plt.scatter(trajBatch_torch[index][:,0], trajBatch_torch[index][:,1], c=OptionsBatch_torch[index][:-1], marker='o',cmap='bwr')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['no coins', 'see coins'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('HIL, Reward {}'.format(RewardBatch_torch[index]))
# plt.savefig('Figures/FiguresExpert/Processed_human_traj.eps', format='eps')
plt.show() 

# %% Train Critics with HTD0

if args.init_critic:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "encoding_info": encoding_info,
        "l_rate_critic": 3e-4, 
        "discount": 0.99,
        "tau": 0.005,
        "eta": 1e-7, 
        }
   
    Train_Critic = H_TD0.H_TD0(**kwargs)
    Train_Critic.load(f"./models/HIL/HIL_{args.env}_{args.seed}")
    
    state, done = env.reset('standard', TrainingSet[0,:]), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    initial_option = 0
    initial_b = 1
    option = Train_Critic.select_option(state, initial_b, initial_option)
    for t in range(int(args.HTD0_timesteps)):
    		
        episode_timesteps += 1
        state = torch.FloatTensor(state.reshape(1,-1)).to(device) 
        action = Train_Critic.select_action(state,option)
        
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        termination = Train_Critic.select_termination(next_state, option)
        
        if termination == 1:
            cost = Train_Critic.eta
        else:
            cost = 0
    
        # Store data in replay buffer
        state_encoded = Train_Critic.encode_state(state.flatten())
        action_encoded = Train_Critic.encode_action(action)
        next_state_encoded = Train_Critic.encode_state(next_state.flatten())
        Train_Critic.Buffer[option].add(state_encoded, action_encoded, next_state_encoded, reward, cost, done_bool)
        
        next_option = Train_Critic.select_option(next_state, termination, option)
    
        state = next_state
        option = next_option
        episode_reward += reward
    
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            Train_Critic.train(option, args.batch_size)
    
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset('standard', TrainingSet[0,:]), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            initial_option = 0
            initial_b = 1
            option = Train_Critic.select_option(state, initial_b, initial_option)   
            
    Train_Critic.save(f"./models/H_TD0/H_TD0_{args.env}_{args.seed}")
# %% Train Policy HSAC
    
if args.policy == "HSAC":
    kwargs = {
    	"state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "encoding_info": encoding_info,
        "l_rate_pi_lo": 1e-2,
        "l_rate_pi_hi": 3e-4,
        "l_rate_pi_b": 3e-4,
        "l_rate_critic": 1e-2, 
        "discount": 0.99,
        "tau": 0.05, 
        "eta": 1e-5, 
        "pi_b_freq": 10,
        "pi_hi_freq": 20,
        "alpha": 1,
        "critic_freq": 2
        }
    Agent_HRL = H_SAC.H_SAC(**kwargs)
    if args.load_model and args.HIL:
    	Agent_HRL.load_actor(f"./models/HIL/HIL_{args.env}_{args.seed}")
    # if args.load_model and args.init_critic:
    # 	Agent_HRL.load_critic(f"./models/H_TD0/H_TD0_{args.env}_{args.seed}")

# Evaluate untrained policy
evaluation_HRL = []
[trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
  TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_HRL, env, max_epoch, eval_episodes, 'standard', TrainingSet[0,:])
avg_reward = np.sum(RewardBatch_torch)/eval_episodes
evaluation_HRL.append(avg_reward)

print("---------------------------------------")
print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
print("---------------------------------------")

state, done = env.reset('standard', TrainingSet[0,:]), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

initial_option = 0
initial_b = 1

option = Agent_HRL.select_option(state, initial_b, initial_option)

for t in range(int(args.max_timesteps)):
		
    episode_timesteps += 1
    state = torch.FloatTensor(state.reshape(1,-1)).to(device) 

    if t < args.start_timesteps:
        action = env.random_sample()    
    elif args.policy == "HSAC":
        action = Agent_HRL.select_action(state,option)
    
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
    
    termination = Agent_HRL.select_termination(next_state, option)
    
    if termination == 1:
        cost = Agent_HRL.eta
    else:
        cost = 0

    # Store data in replay buffer
    state_encoded = Agent_HRL.encode_state(state.flatten())
    action_encoded = Agent_HRL.encode_action(action)
    next_state_encoded = Agent_HRL.encode_state(next_state.flatten())
    Agent_HRL.Buffer[option].add(state_encoded, action_encoded, next_state_encoded, reward, cost, done_bool)
    
    next_option = Agent_HRL.select_option(next_state, termination, option)

    state = next_state
    option = next_option
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
        Agent_HRL.train(option, args.batch_size)

    if done: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset('standard', TrainingSet[0,:]), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 
        initial_option = 0
        initial_b = 1
        option = Agent_HRL.select_option(state, initial_b, initial_option)

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
        [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
         TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_HRL, env, max_epoch, eval_episodes, 'standard', TrainingSet[0,:])
        avg_reward = np.sum(RewardBatch_torch)/eval_episodes
        evaluation_HRL.append(avg_reward)
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        
        np.save(f"./results/{file_name}", evaluation_HRL)
        Agent_HRL.save_actor(f"./models/{file_name}/{file_name}")
        Agent_HRL.save_critic(f"./models/{file_name}/{file_name}")




