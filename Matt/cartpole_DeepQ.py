#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:37:52 2021

@author: vittorio
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import tensorflow as tf 
from tensorflow import keras
import time

# %%

def random_policy(seed, NEpisodes):
    steps_per_episode =[]
    env = gym.make('CartPole-v1')
    env.seed(seed)
    env.action_space.seed(seed)  
    for i_episode in range(NEpisodes):
        observation = env.reset()
        for t in range(1000):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                steps_per_episode.append(t+1)
                break
    env.close()
    
    mean = np.mean(steps_per_episode)
    
    plt.figure()
    plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episode, 'g', label='Policy')
    plt.plot(np.linspace(1,NEpisodes,NEpisodes), mean*np.ones(NEpisodes), 'k--', label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Steps lasted')
    plt.legend()
    plt.title('random policy evaluation')
    plt.savefig('cartpole_random_policy.eps', format="eps")
    
    return steps_per_episode
    
    
class Q_learning_tabular:
    def __init__(self, seed, n_bins = (3, 3, 6, 6)):
        self.env = gym.make('CartPole-v1')
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)  
        # observation_space = [cart_position, cart_velocity, angle, pole_velocity]
        # action_space = [left, right]
        self.n_bins = n_bins
        
        # cart_velocity clipped between -0.2 and 0.2 
        # pole_velocity clipped between -0.87 and 0.87 
        self.lower_bounds = [self.env.observation_space.low[0], -0.2, self.env.observation_space.low[2], -np.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[0], 0.2, self.env.observation_space.high[2], np.radians(50)]
        
        self.Q_table = np.random.normal(0,1,(n_bins + (self.env.action_space.n,)))
        
    def discretizer(self, cart_position, cart_velocity, angle, pole_velocity) -> Tuple[int,...]:
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int,est.transform([[cart_position, cart_velocity, angle, pole_velocity]])[0]))
    
    def Training(self, NEpisodes):
        
        gamma = 0.99
        alpha = 0.1    
        epsilon = 0.1
        steps_per_episode =[]
        for i_episode in range(NEpisodes):
            current_state = Q_learning_tabular.discretizer(self, *self.env.reset())
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_table[current_state])
            
                if np.random.random() <= 1/((t+1)**2): 
                    action = self.env.action_space.sample()       
                
                obs, reward, done, _ = self.env.step(action)
                new_state = Q_learning_tabular.discretizer(self, *obs)
                
                future_optimal_value = np.max(self.Q_table[new_state])
                learned_value = reward + gamma*future_optimal_value     
                old_value = self.Q_table[current_state][action]
                
                self.Q_table[current_state][action] = (1- alpha)*old_value + alpha*learned_value
                
                current_state = new_state
                    
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,NEpisodes,NEpisodes), mean*np.ones(NEpisodes), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Training Tabular Q Learning')
        plt.savefig('cartpole_Train_tabular_Qlearning.eps', format="eps")  
        
        return steps_per_episode
        
    def Evaluation(self, NEpisodes):

        steps_per_episode =[]
        for i_episode in range(NEpisodes):
            current_state = Q_learning_tabular.discretizer(self, *self.env.reset())
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_table[current_state])
                   
                obs, reward, done, _ = self.env.step(action)
                new_state = Q_learning_tabular.discretizer(self, *obs)
                                
                current_state = new_state
                    
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,NEpisodes,NEpisodes), mean*np.ones(NEpisodes), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Evaluation Tabular Q Learning')
        plt.savefig('cartpole_Eval_tabular_Qlearning.eps', format="eps")  
        
        return steps_per_episode
    
        
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal       
        
        
class Q_learning_NN:
    def __init__(self, seed, n_bins = (3, 3, 6, 6)):
        self.env = gym.make('CartPole-v1')
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)  
        self.n_bins = n_bins
        # observation_space = [cart_position, cart_velocity, angle, pole_velocity]
        # action_space = [left, right]
        # cart_velocity clipped between -0.2 and 0.2 
        # pole_velocity clipped between -0.87 and 0.87 
        self.lower_bounds = [self.env.observation_space.low[0], -0.2, self.env.observation_space.low[2], -np.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[0], 0.2, self.env.observation_space.high[2], np.radians(50)]
        self.observation_space_size = self.env.observation_space.shape[0]
        self.Q_network = Q_learning_NN.NN_model(self)


    def NN_model(self):
        model = keras.Sequential([             
                keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                                
                keras.layers.Dense(self.env.action_space.n, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2))
                                 ])              
        
        model.compile(optimizer='adam',
                               loss=tf.keras.losses.MeanSquaredError(),
                               metrics=['accuracy'])
        
        return model
    
    def Training(self, NEpisodes):
        
        gamma = 0.9 
        epsilon = 0.1
        steps_per_episode =[]
        batch_size = 1
        alpha = 0.1
        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                epsilon = max(epsilon*0.99, 0.01)
            
                if np.random.random() <= epsilon: 
                    action = self.env.action_space.sample()       
                
                obs, reward, Done, _ = self.env.step(action)
                new_state = obs
                
                future_optimal_value = np.max(self.Q_network(new_state.reshape(1,len(new_state))),1)
                learned_value = reward + gamma*future_optimal_value*Done 
                
                old_value = self.Q_network(current_state.reshape(1,len(current_state))).numpy()
                y = old_value
                y[np.arange(batch_size),action] = (1-alpha)*old_value[np.arange(batch_size),action] + alpha*learned_value
            
                self.Q_network.fit(current_state.reshape(1,len(current_state)), y, epochs=1, verbose = 0)
                
                current_state = new_state
                                
                    
                if Done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
            # end = time.time()
            # if end-start > maxTime:
            #     break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,i_episode+1,i_episode+1),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,i_episode+1,i_episode+1), mean*np.ones(i_episode+1), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Training NN Q Learning')
        plt.savefig('cartpole_Tain_NN_Qlearning.eps', format="eps")  
        
        return steps_per_episode
    
    
    def Training_buffer(self, NEpisodes):
        
        gamma = 0.99 
        epsilon = 0.5
        steps_per_episode =[]
        batch_size = 256
        Buffer = ReplayBuffer(10000, self.observation_space_size)
        start = time.time()
        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = self.env.action_space.sample()       
                
                obs, reward, Done, _ = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state, Done)
                current_state = new_state
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state, done = Buffer.sample_buffer(batch_size)
                
                    future_optimal_value = np.max(self.Q_network(new_state),1)
                    learned_value = reward + gamma*future_optimal_value*done 
                
                    y = self.Q_network(state).numpy()
                    y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                    self.Q_network.fit(state, y, epochs=1, verbose = 0)
                                
                    
                if Done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
            # end = time.time()
            # if end-start > maxTime:
            #     break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,i_episode+1,i_episode+1),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,i_episode+1,i_episode+1), mean*np.ones(i_episode+1), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Training NN Q Learning')
        plt.savefig('cartpole_Tain_NN_Qlearning.eps', format="eps")  
        
        return steps_per_episode

    def Training_buffer_Double(self, NEpisodes):
        
        gamma = 0.99 
        epsilon = 0.5
        steps_per_episode =[]
        batch_size = 256
        Buffer = ReplayBuffer(10000, self.observation_space_size)
        start = time.time()
        
        Double_Q_net = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=5),
                               bias_initializer=keras.initializers.Zeros()),                                
            keras.layers.Dense(self.env.action_space.n, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=6))
                                 ])              
        
        Double_Q_net.compile(optimizer='adam',
                             loss=tf.keras.losses.MeanSquaredError(),
                             metrics=['accuracy'])
        
        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                b = np.argmax(Double_Q_net(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = self.env.action_space.sample()       
                    
                if np.random.random() <= epsilon: 
                    b = self.env.action_space.sample()     
                
                obs, reward, Done, _ = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state, Done)
                current_state = new_state
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state, done = Buffer.sample_buffer(batch_size)
                    
                    if np.random.random()<=0.5:
                
                        future_optimal_value = np.max(Double_Q_net(new_state),1)
                        learned_value = reward + gamma*future_optimal_value*done 
                
                        y = self.Q_network(state).numpy()
                        y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                        self.Q_network.fit(state, y, epochs=1, verbose = 0)
                        
                    else:
                        future_optimal_value = np.max(self.Q_network(new_state),1)
                        learned_value = reward + gamma*future_optimal_value*done 
                        y = Double_Q_net(state).numpy()
                        y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                        Double_Q_net.fit(state, y, epochs=1, verbose = 0)                        
                                
                  
                if Done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
            # end = time.time()
            # if end-start > maxTime:
            #     break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,i_episode+1,i_episode+1),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,i_episode+1,i_episode+1), mean*np.ones(i_episode+1), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Training NN Q Learning')
        plt.savefig('cartpole_Tain_NN_Qlearning.eps', format="eps")  
        
        return steps_per_episode
        
    def Evaluation(self, NEpisodes):

        steps_per_episode =[]

        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                   
                obs, reward, done, _ = self.env.step(action)
                current_state = obs
                    
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,NEpisodes,NEpisodes), mean*np.ones(NEpisodes), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Evaluation NN Q Learning')
        plt.savefig('cartpole_Eval_NN_Qlearning.eps', format="eps")  
        
        return steps_per_episode
        

NEpisodes = 50
rp_list = []
tabular_training_list = []
tabular_eval_list = []
NN_training_list = []
NN_eval_list = []
NN_buffer_training_list = []
NN_buffer_eval_list = []

for seed in range(1):
    # rp_list.append(random_policy(seed, NEpisodes))
    # agent_Q_learning_tabular = Q_learning_tabular(seed)
    # tabular_training_list.append(agent_Q_learning_tabular.Training(NEpisodes))
    # tabular_eval_list.append(agent_Q_learning_tabular.Evaluation(NEpisodes))
    # agent_NN_Q_learning = Q_learning_NN(seed)
    # NN_training_list.append(agent_NN_Q_learning.Training(NEpisodes))
    # NN_eval_list.append(agent_NN_Q_learning.Evaluation(NEpisodes))
    agent_NN_Q_learning_buffer = Q_learning_NN(seed)
    NN_buffer_training_list.append(agent_NN_Q_learning_buffer.Training_buffer(NEpisodes))
    NN_buffer_eval_list.append(agent_NN_Q_learning_buffer.Evaluation(NEpisodes))
    agent_NN_Double_Q_learning_buffer = Q_learning_NN(seed)
    agent_NN_Double_Q_learning_buffer.Training_buffer_Double(NEpisodes)
    agent_NN_Double_Q_learning_buffer.Evaluation(NEpisodes)


#%%

rp = np.asarray(rp_list)
tabular_training = np.asarray(tabular_training_list)
tabular_eval = np.asarray(tabular_eval_list)
NN_training = np.asarray(NN_training_list)
NN_eval = np.asarray(NN_eval_list)
NN_buffer_training = np.asarray(NN_buffer_training_list)
NN_buffer_eval = np.asarray(NN_buffer_eval_list)


# Training
ave_tab_train = np.mean(tabular_training,0)
std_tab_train = np.std(tabular_training,0)
ave_NN_train = np.mean(NN_training,0)
std_NN_train = np.std(NN_training,0)
ave_NN_buffer_train = np.mean(NN_buffer_training,0)
std_NN_buffer_train = np.std(NN_buffer_training,0)


fig, ax = plt.subplots()
ax.plot(np.arange(len(ave_tab_train)), ave_tab_train, label='Tabular Q-learning', c='b')
ax.fill_between(np.arange(len(ave_tab_train)), ave_tab_train-std_tab_train, ave_tab_train+std_tab_train, alpha=0.1, facecolor='b')
ax.plot(np.arange(len(ave_NN_train)), ave_NN_train, label='Deep Q-learning', c='g')
ax.fill_between(np.arange(len(ave_NN_train)), ave_NN_train-std_NN_train, ave_NN_train+std_NN_train, alpha=0.1, facecolor='g')
ax.plot(np.arange(len(ave_NN_buffer_train)), ave_NN_buffer_train, label='Deep Q-learning w/ buffer', c='k')
ax.fill_between(np.arange(len(ave_NN_buffer_train)), ave_NN_buffer_train-std_NN_buffer_train, ave_NN_buffer_train+std_NN_buffer_train, alpha=0.1, facecolor='k')
ax.legend(facecolor = '#d8dcd6')
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Reward')
ax.set_title('Training')
plt.savefig('Training_comparison.png', format='png')


# Evaluation
ave_rp = np.mean(rp,0)
std_rp = np.std(rp,0)
ave_tabular_eval = np.mean(tabular_eval,0)
std_tabular_eval = np.std(tabular_eval,0)
ave_NN_eval = np.mean(NN_eval,0)
std_NN_eval = np.std(NN_eval,0)
ave_NN_buffer_eval = np.mean(NN_buffer_eval,0)
std_NN_buffer_eval = np.std(NN_buffer_eval,0)

fig, ax = plt.subplots()
ax.plot(np.arange(len(ave_rp)), ave_rp, label='random Policy', c='r')
ax.fill_between(np.arange(len(ave_rp)), ave_rp-std_rp, ave_rp+std_rp, alpha=0.1, facecolor='r')
ax.plot(np.arange(len(ave_tabular_eval)), ave_tabular_eval, label='Tabular Q-learning', c='b')
ax.fill_between(np.arange(len(ave_tabular_eval)), ave_tabular_eval-std_tabular_eval, ave_tabular_eval+std_tabular_eval, alpha=0.1, facecolor='b')
ax.plot(np.arange(len(ave_NN_eval)), ave_NN_eval, label='Deep Q-learning', c='g')
ax.fill_between(np.arange(len(ave_NN_eval)), ave_NN_eval-std_NN_eval, ave_NN_eval+std_NN_eval, alpha=0.1, facecolor='g')
ax.plot(np.arange(len(ave_NN_buffer_eval)), ave_NN_buffer_eval, label='Deep Q-learning w/ buffer', c='k')
ax.fill_between(np.arange(len(ave_NN_buffer_eval)), ave_NN_buffer_eval-std_NN_buffer_eval, ave_NN_buffer_eval+std_NN_buffer_eval, alpha=0.1, facecolor='k')
ax.legend(facecolor = '#d8dcd6')
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Reward')
ax.set_title('Evaluation')
plt.savefig('Evaluation_comparison.png', format='png')

# plt.figure()
# plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episodeRP, 'r', label='Random Policy')
# plt.plot(np.linspace(1,NEpisodes,NEpisodes), steps_per_episodeTab, 'b', label='Tabular Q Learning')
# plt.plot(np.linspace(1,NEpisodes,NEpisodes), steps_per_episodeNN, 'g', label='NN Q Learning')
# plt.xlabel('Episode')
# plt.ylabel('Steps lasted')
# plt.legend()
# plt.title('Comparison')
# plt.savefig('Comparison_Evaluation_Qlearning.eps', format="eps")  


