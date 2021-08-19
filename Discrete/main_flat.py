#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:00:09 2021

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
import multiprocessing
import multiprocessing.pool

from utils import Encode_Data
from BatchBW_HIL_torch import BatchBW

from evaluation import HierarchicalStochasticSampleTrajMDP
from evaluation import eval_policy

import TRPO

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
parser.add_argument("--number_options", default=1, type=int)     # number of options
parser.add_argument("--policy", default="TRPO")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--number_steps_per_iter", default=50000, type=int) # Time steps initial random policy is used 25e3
parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
parser.add_argument("--max_iter", default=3e6/50000, type=int)    # Max time steps to run environment
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
parser.add_argument("--init_critic", default=False, type=bool)   
parser.add_argument("--HTD0_timesteps", default=3e5, type=int)    # Max time steps to run environment
# HRL
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--alpha", default=0.2, type=int)            # SAC entropy regularizer term
parser.add_argument("--critic_freq", default=2, type=int)        # Frequency of delayed critic updates
parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--load_model_path", default="") 
# Evaluation
parser.add_argument("--evaluation_episodes", default=10, type=int)
parser.add_argument("--evaluation_max_n_steps", default=size_data, type=int)
args = parser.parse_args()

file_name = f"{args.policy}_HIL_{args.HIL}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, HIL: {args.HIL}, Env: {args.env}, Seed: {args.seed}")
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
 
# Initialize policy        
if args.policy == "TRPO":
    kwargs = {
     "state_dim": state_dim,
     "action_dim": action_dim,
     "encoding_info": encoding_info,
     "num_steps_per_rollout": args.number_steps_per_iter
     }
     # Target policy smoothing is scaled wrt the action scale
    policy = TRPO.TRPO(**kwargs)
    if args.load_model and args.HIL:
    	policy.load_actor(f"./models/HIL/HIL_{args.env}_{args.seed}", HIL=args.HIL)
    
     
#  # Initialize policy        
# if args.policy == "UATRPO":
#     kwargs = {
#      "state_dim": state_dim,
#      "action_dim": action_dim,
#      "max_action": max_action,
#      }
#      # Target policy smoothing is scaled wrt the action scale
#     policy = UATRPO.UATRPO(**kwargs)
     
# if args.policy == "PPO":
#     kwargs = {
#      "state_dim": state_dim,
#      "action_dim": action_dim,
#      "max_action": max_action,
#     }
#      # Target policy smoothing is scaled wrt the action scale
#     policy = PPO.PPO(**kwargs)
     
# if args.GAIL:
#     kwargs = {
#      "state_dim": state_dim,
#      "action_dim": action_dim,
#      "expert_states": TrainingSet_tot,
#      "expert_actions": Labels_tot,
#      }
#     IRL = GAIL.Gail(**kwargs)
         	
 # Evaluate untrained policy
evaluations = [eval_policy(policy, env, args.seed, 0)]

for i in range(int(args.max_iter)):
		
    # if args.GAIL:
    #     rollout_states, rollout_actions = policy.GAE(env, args.GAIL, IRL.discriminator)
    #     IRL.update(rollout_states, rollout_actions)
    #     policy.train()
    # else:
        
    rollout_states, rollout_actions = policy.GAE(env)
    policy.train(Entropy = True)
         

    # Evaluate episode
    if (i + 1) % args.eval_freq == 0:
         evaluations.append(eval_policy(policy, env, args.seed, i+1))

if args.save_model: 
    np.save(f"./results/evaluation_{file_name}", evaluations)


