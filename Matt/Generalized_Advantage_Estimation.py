#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:52:41 2021

@author: vittorio
"""

import numpy as np
import scipy.signal as sp_sig
import tensorflow as tf 
from tensorflow import keras
import World

# %%

    
class Generalized_Adv_Estimator:
    def __init__(self, seed, Folder, expert_traj, gamma, lam, import_net=False, weights = 0):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.critic = Generalized_Adv_Estimator.NN_model(self)
        if import_net:
            self.critic.set_weights(weights)
            
        self.gamma = gamma
        self.lam = lam
            
def NN_model(input_size, output_size, seed_init):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),        
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),                         
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init))
                             ])       

    model.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])     
    
    return model
    
    def discounted_sum(x,rate):
        return sp_sig.lfilter([1], [1, float(-rate)], x[::-1], axis=0)[::-1]

    def reward_calc(self, reward_traj,V,V_end):
        """Calculates advantages and reward-to-go"""
        r_all = np.concatenate((reward_traj,[V_end]),-1)
        V_all =  V #np.concatenate((V,[V_end]),-1)
        delta = r_all[:-1] + self.gamma * V_all[1:] - V_all[:-1]
        
        adv = Generalized_Adv_Estimator.discounted_sum(delta,self.gamma*self.lam)
        rtg = adv + V_all[:-1]

        adv = adv.astype('float32')
        rtg = rtg.astype('float32')

        return adv, rtg
    
    def Runner(self, initial_state, T=3000):
        
        reward_traj =[]
        traj = []
        control_traj = []        

        current_state = self.env.reset('standard', init_state=initial_state)
        cum_reward = 0 
        traj.append(current_state)
            
        for t in range(T):
            
            mean = np.argmax(self.critic(current_state.reshape(1,len(current_state))))
            action = int(np.random.normal(mean, 1.5, 1))%(self.env.action_size-1)
            obs, reward = self.env.step(action)
            current_state = obs
            traj.append(current_state)
            control_traj.append(action)
            reward_traj.append(reward)
            cum_reward = cum_reward + reward
        
                                
        # print("GAE: cumulative reward = {}".format(cum_reward))
        V_end = np.sum((1/self.env.action_size)*np.ones((1,self.env.action_size))*self.critic(current_state.reshape(1,len(current_state))).numpy())
        
        traj = np.array(traj)
        control_traj = np.array(control_traj)
        reward_traj = np.array(reward_traj)
        
        V = np.sum((1/self.env.action_size)*np.ones((len(traj),self.env.action_size))*self.critic(traj).numpy(),1)
        
        adv_traj, rtg_traj = Generalized_Adv_Estimator.reward_calc(self, reward_traj, V, V_end)
                
        return  adv_traj, rtg_traj, traj, control_traj, reward_traj

    
with open('RL_algorithms/DeepQ_Learning/Results/Q_learning_results_deeper.npy', 'rb') as f:
    Mixture_of_DQN = np.load(f, allow_pickle=True).tolist()
    
N_agents = 10

best_reward = np.zeros(N_agents)
best_episode = [[None]*1 for _ in range(N_agents)]
best_agent = np.zeros(N_agents)
best_net = [[None]*1 for _ in range(N_agents)]

for i in range(len(Mixture_of_DQN)):
    for j in range(len(Mixture_of_DQN[i][0])):
        temp = Mixture_of_DQN[i][0][j]
        
        for k in range(N_agents):
            if temp>best_reward[k]:
                best_reward[k]=Mixture_of_DQN[i][0][j]
                best_episode[k]=Mixture_of_DQN[i][1][j]
                best_agent[k]=i
                best_net[k]=Mixture_of_DQN[i][2][j]
                break    
    
NEpisodes = 100
best_index = 0 #np.argmax(best_reward) 
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2
seed = 33
lam = 0.97
gamma = 0.995
reset = 'standard'
initial_state = np.array([0, -2.6, 0, 8])

GAE = Generalized_Adv_Estimator(seed, Folders, Rand_traj, gamma, lam, import_net=True, weights = best_net[best_index])

cum_reward = 0
for i in range(NEpisodes):
    adv, rtg, traj, control, reward = GAE.Runner(initial_state)
    cum_reward = cum_reward + np.sum(reward)
    average = (cum_reward)/(i+1)
    print("GAE: cumulative reward = {}, average = {}".format(np.sum(reward), average))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    