#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:02:25 2021

@author: vittorio
"""

import copy
import numpy as np
import World
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def HierarchicalStochasticSampleTrajMDP(Hierarchical_policy, env, max_epoch_per_traj, number_of_trajectories, reset = 'random', initial_state = np.array([0,0,0,8])):
    traj = [[None]*1 for _ in range(number_of_trajectories)]
    control = [[None]*1 for _ in range(number_of_trajectories)]
    Option = [[None]*1 for _ in range(number_of_trajectories)]
    Termination = [[None]*1 for _ in range(number_of_trajectories)]
    Reward_array = np.empty((0,0),int)
   
    for option in range(0,Hierarchical_policy.option_dim):
        Hierarchical_policy.pi_lo[option].eval()  
        Hierarchical_policy.pi_b[option].eval()
    Hierarchical_policy.pi_hi.eval()
   
    for t in range(number_of_trajectories):
        current_state = env.reset(reset, initial_state)
        size_input = len(current_state)
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        x = np.empty((0, size_input))
        x = np.append(x, current_state.reshape(1, size_input), 0)
        u_tot = np.empty((0,0),int)
        cum_reward = 0 
    
        # Initial Option
        initial_option = 0
        initial_b = 1
        
        option = Hierarchical_policy.select_option(current_state, initial_b, initial_option)
        o_tot = np.append(o_tot,option)
        termination = Hierarchical_policy.select_termination(current_state, option)
        b_tot = np.append(b_tot, termination)
        option = Hierarchical_policy.select_option(current_state, termination, option)
        o_tot = np.append(o_tot,option)
     
        for k in range(0,max_epoch_per_traj):
            # draw action
            current_action = Hierarchical_policy.select_action(current_state, option)
            u_tot = np.append(u_tot,current_action)     
    
            # given action, draw next state
            obs, reward, done, _ = env.step(current_action)
            new_state = obs
            
            termination = Hierarchical_policy.select_termination(new_state, option)
            b_tot = np.append(b_tot, termination)
            option = Hierarchical_policy.select_option(new_state, termination, option)
            o_tot = np.append(o_tot,option)
            
            current_state = new_state
            x = np.append(x, current_state.reshape(1, size_input), 0)
            cum_reward = cum_reward + reward    
    
    
        traj[t] = x
        control[t]=u_tot
        Option[t]=o_tot
        Termination[t]=b_tot
        Reward_array = np.append(Reward_array, cum_reward)
    
    return traj, control, Option, Termination, Reward_array    

def eval_policy(policy, env_name, seed, training_iter, eval_episodes=10, reset = 'random', initial_state = np.array([0,0,0,8])):
	eval_env = env_name

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(reset, initial_state), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Seed {seed}, Iter {training_iter}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward