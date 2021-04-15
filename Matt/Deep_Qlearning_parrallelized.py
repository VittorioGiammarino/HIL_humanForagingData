#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:47:52 2021

@author: vittorio
"""

import World 
import numpy as np
from tensorflow import keras
import tensorflow as tf 
import multiprocessing
import multiprocessing.pool

# %%

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]

        return states, actions, rewards, states_     
        
        
class Q_learning_NN:
    def __init__(self, seed, Folder, expert_traj):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.Q_network = Q_learning_NN.NN_model(self)

    def NN_model(self):
        model = keras.Sequential([             
                keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                                
                keras.layers.Dense(self.env.action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2))
                                 ])              
        
        model.compile(optimizer='adam',
                               loss=tf.keras.losses.MeanSquaredError(),
                               metrics=['accuracy'])
        
        return model
    
    
    def Training_buffer(self, NEpisodes, seed):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        batch_size = 256
        Buffer = ReplayBuffer(30000, self.observation_space_size)
        
        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            current_state = self.env.reset('random')
            cum_reward = 0 
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)       
                
                obs, reward = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state)
                current_state = new_state
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                cum_reward = cum_reward + reward
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state = Buffer.sample_buffer(batch_size)
                
                    future_optimal_value = np.max(self.Q_network(new_state),1)
                    learned_value = reward + gamma*future_optimal_value
                
                    y = self.Q_network(state).numpy()
                    y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                    self.Q_network.fit(state, y, epochs=1, verbose = 0)
                                
                                
                    
            print("Episode {}: cumulative reward = {} (seed = {})".format(i_episode, cum_reward, seed))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x
            
        network_weights = self.Q_network.get_weights()
        return reward_per_episode, traj, network_weights 


def train(seed, Folders, Rand_traj, NEpisodes):
    agent_NN_Q_learning_buffer = Q_learning_NN(seed, Folders, Rand_traj)
    reward_per_episode, traj, network_weights = agent_NN_Q_learning_buffer.Training_buffer(NEpisodes, seed)
    
    return reward_per_episode, traj, network_weights

NEpisodes = 100
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2

Nprocesses = 40
pool = multiprocessing.Pool(processes=Nprocesses)
args = [(seed, Folders, Rand_traj, NEpisodes) for seed in range(Nprocesses)]
Q_learning_results = pool.starmap(train, args) 
pool.close()
pool.join()

# %%
with open('4_walls_coins_task/Q_learning_results.npy', 'wb') as f:
    np.save(f, Q_learning_results)