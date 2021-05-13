#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:35:49 2021

@author: vittorio
"""


import World 
import numpy as np
from tensorflow import keras
import tensorflow as tf 
import multiprocessing
import multiprocessing.pool

# %%

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


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
        
        
        
class Q_learning_NN_encoded:
    def __init__(self, seed, Folder, expert_traj, import_net=False, weights = 0):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size = self.coordinates + self.view + self.closest_coin_dir # first 2 are fore coordinates, second 2 for view, and 9 for closest coin direction
        self.Q_network = Q_learning_NN_encoded.NN_model(self)
        if import_net:
            self.Q_network.set_weights(weights)

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
    
    
    def Training_buffer(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        network_weights = [[None]*1 for _ in range(NEpisodes)]
        batch_size = 256
        Buffer = ReplayBuffer(30000, self.observation_space_size)
        
        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            current_state = self.env.reset(reset, initial_state)
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(self.view)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(self.closest_coin_dir)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
            cum_reward = 0 
            x = np.append(x, current_state_encoded.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state_encoded.reshape(1,self.observation_space_size)))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)       
                
                obs, reward = self.env.step(action)
                new_state = obs
                coordinates = new_state[0:2]
                psi = new_state[2]
                psi_encoded = np.zeros(self.view)
                psi_encoded[int(psi)]=1
                coin_dir_encoded = np.zeros(self.closest_coin_dir)
                coin_dir = new_state[3]
                coin_dir_encoded[int(coin_dir)]=1
                new_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
                Buffer.store_transition(current_state_encoded, action, reward, new_state_encoded)
                current_state_encoded = new_state_encoded
                x = np.append(x, current_state_encoded.reshape(1, self.observation_space_size), 0)
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
            network_weights[i_episode] = self.Q_network.get_weights()
            
        return reward_per_episode, traj, network_weights 
    

def Train(seed, Folders, Rand_traj, NEpisodes, reset, initial_state):
    agent_NN_Q_learning_buffer = Q_learning_NN_encoded(seed, Folders, Rand_traj)
    reward_per_episode, traj, network_weights = agent_NN_Q_learning_buffer.Training_buffer(NEpisodes, seed, reset, initial_state)
    
    return reward_per_episode, traj, network_weights



NEpisodes = 200
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2
reset = 'standard'
initial_state = np.array([0, -2.6, 0, 8])

# reward_per_episode, traj, network_weights = train(0, Folders, Rand_traj, NEpisodes, reset, initial_state)
Ncpu = 40
pool = MyPool(Ncpu)
args = [(seed, Folders, Rand_traj, NEpisodes, reset, initial_state) for seed in range(Ncpu)]
Q_learning_results_encoded = pool.starmap(Train, args) 
pool.close()
pool.join()

# %%
with open('4_walls_coins_task/Q_learning_results_encoded.npy', 'wb') as f:
    np.save(f, Q_learning_results_encoded)