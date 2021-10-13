#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
import torch
import argparse
import os
import numpy as np
import pickle

import World

from utils import Encode_Data
from BatchBW_HIL_torch import BatchBW

from evaluation import evaluate_H
from evaluation import eval_policy

import H_PPO
import PPO

import H_SAC
import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

def HIL(env, args, seed):
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
        
    print("---------------------------------------")
    print(f"HIL: {args.HIL}, Traj: {args.coins}, nOptions: {args.number_options}, Supervised: {args.pi_hi_supervised}, Seed: {args.seed}")
    print("---------------------------------------")
    
    TrainingSet = Trajectories[args.coins]
    Labels = Rotation[args.coins]
   
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)
    
    state_dim = state_samples.shape[1]
    action_dim = env.action_size
    option_dim = args.number_options
    termination_dim = 2
    
    kwargs = {
    	"state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "state_samples": state_samples,
        "action_samples": action_samples,
        "M_step_epoch": args.maximization_epochs_HIL,
        "batch_size": args.batch_size_HIL,
        "l_rate": args.l_rate_HIL,
        "encoding_info": encoding_info
        }
    
    Agent_BatchHIL_torch = BatchBW(**kwargs)
    
    if args.pi_hi_supervised:
        if args.number_options == 2:
            Options = np.copy(state_samples[:,2])
                
        if args.number_options == 3:
            Options = np.copy(state_samples[:,2])      
            for s in range(len(Options)):
                if state_samples[s,1] > 6.5 or state_samples[s,1] < -6.5 or state_samples[s,0] > 6.5 or state_samples[s,0] < -6.5:
                    if Options[s] == 0:
                        Options[s] = 2            
            
        epochs = args.pi_hi_supervised_epochs
        Agent_BatchHIL_torch.pretrain_pi_hi(epochs, Options)
        Labels_b = Agent_BatchHIL_torch.prepare_labels_pretrain_pi_b(Options)
        for i in range(args.number_options):
            Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[i], i)
            

    Loss = 100000
    evaluation_HIL = []
    avg_reward = evaluate_H(seed, Agent_BatchHIL_torch, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
    evaluation_HIL.append(avg_reward)
    for i in range(args.N_iterations):
        print(f"Iteration {i+1}/{args.N_iterations}")
        loss = Agent_BatchHIL_torch.Baum_Welch()
        if loss > Loss:
            Agent_BatchHIL_torch.reset_learning_rate(args.l_rate_HIL/10)
        Loss = loss
        avg_reward = evaluate_H(seed, Agent_BatchHIL_torch, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_HIL.append(avg_reward)
        
    # Save
    np.save(f"./results/HRL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}", evaluation_HIL)
    
    if not os.path.exists(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}"):
        os.makedirs(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}")
    
    Agent_BatchHIL_torch.save(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}/HIL")
    
    
def HRL(env, args, seed):
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    TrainingSet = Trajectories[args.coins]
    Labels = Rotation[args.coins]
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)
    state_dim = state_samples.shape[1]
    action_dim = env.action_size
    option_dim = args.number_options
    termination_dim = 2
             
    if args.policy == "HPPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_HRL = H_PPO.H_PPO(**kwargs)
        
        if args.load_model and args.HIL and args.load_HIL_model:
            Agent_HRL.load_actor(f"./models/HIL_ablation_study/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.load_HIL_model_seed}/HIL")         
        elif args.load_model and args.HIL:
        	Agent_HRL.load_actor(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}/HIL") 
                  	
        # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_HRL.append(avg_reward) 
        
        for i in range(int(args.max_iter)):
                        
            _, _, _, _ = Agent_HRL.GAE(env)
            Agent_HRL.train(Entropy = True)
                 
            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
                evaluation_HRL.append(avg_reward)      
                 
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "PPO":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_RL = PPO.PPO(**kwargs)
        
        if args.load_model and args.HIL and args.load_HIL_model:
            Agent_RL.load_actor(f"./models/HIL_ablation_study/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.load_HIL_model_seed}/HIL")         
        elif args.load_model and args.HIL:
        	Agent_RL.load_actor(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}/HIL") 
                  	
        # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, Agent_RL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_RL.append(avg_reward) 
        
        for i in range(int(args.max_iter)):
                        
            _, _ = Agent_RL.GAE(env)
            Agent_RL.train(Entropy = True)
                 
            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, Agent_RL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
                evaluation_RL.append(avg_reward)      
                 
        return evaluation_RL, Agent_RL
    
    if args.policy == "HSAC":
        kwargs = {
        	"state_dim": state_dim,
            "action_dim": action_dim,
            "option_dim": option_dim,
            "termination_dim": termination_dim,
            "encoding_info": encoding_info,
        }

        Agent_HRL = H_SAC.H_SAC(**kwargs)
        
        if args.load_model and args.HIL and args.load_HIL_model:
            Agent_HRL.load_actor(f"./models/HIL_ablation_study/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.load_HIL_model_seed}/HIL")         
        elif args.load_model and args.HIL:
        	Agent_HRL.load_actor(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}/HIL") 
                  	
        # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_HRL.append(avg_reward) 
        
        state, done = env.reset('random'), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        
        initial_option = 0
        initial_b = 1
        
        option = Agent_HRL.select_option(state, initial_b, initial_option)
        
        for t in range(int(args.number_steps_per_iter*args.max_iter)):
        		
            episode_timesteps += 1
            state = torch.FloatTensor(state.reshape(1,-1)).to(device) 
        
            if t < args.start_timesteps:
                action = env.random_sample()    
            else:
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
                Agent_HRL.train(option)
        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset('random'), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                initial_option = 0
                initial_b = 1
                option = Agent_HRL.select_option(state, initial_b, initial_option)
        
            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
                evaluation_HRL.append(avg_reward)  
                 
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "SAC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "encoding_info": encoding_info,
        }

        Agent_RL = SAC.SAC(**kwargs)
        
        if args.load_model and args.HIL and args.load_HIL_model:
            Agent_RL.load_actor(f"./models/HIL_ablation_study/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.load_HIL_model_seed}/HIL")         
        elif args.load_model and args.HIL:
        	Agent_RL.load_actor(f"./models/HRL/HIL/HIL_traj_{args.coins}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{seed}/HIL") 
                  	
        # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, Agent_RL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset('random'), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.random_sample()   
            else:
                action = Agent_RL.select_action(np.array(state))

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            state_encoded = Agent_RL.encode_state(state.flatten())
            action_encoded = Agent_RL.encode_action(action)
            next_state_encoded = Agent_RL.encode_state(next_state.flatten())
            Agent_RL.Buffer.add(state_encoded, action_encoded, next_state_encoded, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                Agent_RL.train()

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset('random'), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, Agent_RL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, Agent_RL
    
    
def train(env, args, seed): 
    
    # Set seeds
    env.Seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.HIL and not args.load_HIL_model:
        HIL(env, args, seed)
        
    evaluations, policy = HRL(env, args, seed)
    
    return evaluations, policy


if __name__ == "__main__":
    
    def load_obj(name):
        with open('results/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    Coins_location = np.load("./Expert_data/Coins_location.npy")
    len_trajs = []
    for i in range(len(Trajectories)):
        len_trajs.append(len(Trajectories[i]))
        
    mean_len_trajs = int(np.mean(len_trajs))
    
    HIL_ablation_study_results = load_obj('HIL_ablation_study/Sorted_results')
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--number_options", default=3, type=int)     # number of options
    parser.add_argument("--policy", default="SAC")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=30000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=334, type=int)    # Max time steps to run environment
    parser.add_argument("--coins", default=2, type=int)
    #IL
    parser.add_argument("--HIL", action="store_false")          # Batch size for HIL
    parser.add_argument("--load_HIL_model", default=True, type=bool)         # Batch size for HIL
    parser.add_argument("--load_HIL_model_seed", default=0, type=int)         # Batch size for HIL
    parser.add_argument("--load_HIL_model_expert_traj", default=0, type=int)         # Batch size for HIL
    parser.add_argument("--size_data_set", default=3000, type=int)         # Batch size for HIL
    parser.add_argument("--batch_size_HIL", default=32, type=int)         # Batch size for HIL
    parser.add_argument("--maximization_epochs_HIL", default=10, type=int) # Optimization epochs HIL
    parser.add_argument("--l_rate_HIL", default=0.001, type=float)         # Optimization epochs HIL
    parser.add_argument("--N_iterations", default=11, type=int)            # Number of EM iterations
    parser.add_argument("--pi_hi_supervised", action="store_true")     # Supervised pi_hi
    parser.add_argument("--pi_hi_supervised_epochs", default=200, type=int)  
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = mean_len_trajs, type=int)
    args = parser.parse_args()
    
    file_name = f"{args.policy}_HIL_{args.HIL}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, HIL: {args.HIL}, nOptions: {args.number_options}, Supervised: {args.pi_hi_supervised}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
        
       
    if not os.path.exists("./results/HRL"):
        os.makedirs("./results/HRL")
               
    if not os.path.exists("./models/HRL/HIL/"):
        os.makedirs("./models/HRL/HIL")
        
    if args.policy == "PPO" or args.policy == "SAC":
        args.number_options = 1
        args.pi_hi_supervised = False
        
        file_name = f"{args.policy}_HIL_{args.HIL}_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, HIL: {args.HIL}, nOptions: {args.number_options}, Supervised: {args.pi_hi_supervised}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
    if not os.path.exists(f"./models/HRL/{file_name}"):
        os.makedirs(f"./models/HRL/{file_name}")
        
    if args.load_HIL_model:
        args.load_HIL_model_seed = HIL_ablation_study_results[f'Best_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}']['seed']
        args.load_HIL_model_expert_traj = HIL_ablation_study_results[f'Best_nOptions_{args.number_options}_supervised_{args.pi_hi_supervised}']['expert traj']
        args.coins = args.load_HIL_model_expert_traj
     
    coins_distribution = 2 # we standardize the exact coins position throughout the experiments    
    coins_location = Coins_location[coins_distribution,:,:] 
    env = World.Foraging.env(coins_location)

    evaluations, policy = train(env, args, args.seed)
    
    if args.save_model: 
        np.save(f"./results/HRL/evaluation_{file_name}", evaluations)
        policy.save_actor(f"./models/HRL/{file_name}/{file_name}")
        policy.save_critic(f"./models/HRL/{file_name}/{file_name}")
                
    

