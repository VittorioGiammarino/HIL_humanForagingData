#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:03:44 2021

@author: vittorio
"""


import numpy as np
import scipy.signal as sp_sig
import tensorflow as tf 
from tensorflow import keras
import World

# %%

def critic_model(input_size, output_size, seed_init):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),        
            keras.layers.Dense(256, activation='relu',
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),                         
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init))
                             ])       

    model.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])     
    
    return model


def actor_model(input_size, output_size, seed_init):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),        
            keras.layers.Dense(256, activation='relu',
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),                         
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init))
                             ])       

    model.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])     
    
    return model

    
class TRPO:
    def __init__(self, seed, Folder, expert_traj, gamma, lam, import_net=False, weights = 0):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
        
        self.critic = critic_model(self.observation_space_size, 1, seed)
        self.actor = actor_model(self.observation_space_size_encoded, self.env.action_size)
            
        self.gamma = gamma
        self.lam = lam
            
    
    def discounted_sum(x,rate):
        return sp_sig.lfilter([1], [1, float(-rate)], x[::-1], axis=0)[::-1]

    def reward_calc(self, reward_traj,V,V_end):
        """Calculates advantages and reward-to-go"""
        r_all = np.concatenate((reward_traj,[V_end]),-1)
        V_all =  V #np.concatenate((V,[V_end]),-1)
        delta = r_all[:-1] + self.gamma * V_all[1:] - V_all[:-1]
        
        adv = TRPO.discounted_sum(delta,self.gamma*self.lam)
        rtg = adv + V_all[:-1]

        adv = adv.astype('float32')
        rtg = rtg.astype('float32')

        return adv, rtg
    
    def Runner(self, initial_state, T=3000):
        
        reward_traj =[]
        traj = []
        control_traj = []        

        current_state = self.env.reset('standard', init_state=initial_state)
        coordinates = current_state[0:2]
        psi = current_state[2]
        psi_encoded = np.zeros(self.view)
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(self.closest_coin_dir)
        coin_dir = current_state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))        
        cum_reward = 0 
        traj.append(current_state)
            
        for t in range(T):
            
            # draw action
            prob_u = self.actor(current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            action = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
            obs, reward = self.env.step(action)
            current_state = obs
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(self.view)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(self.closest_coin_dir)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))   
            
            traj.append(current_state)
            control_traj.append(action)
            reward_traj.append(reward)
            cum_reward = cum_reward + reward
        
                                
        # print("GAE: cumulative reward = {}".format(cum_reward))
        V_end = self.critic(current_state.reshape(1,len(current_state))).numpy()
        
        traj = np.array(traj)
        control_traj = np.array(control_traj)
        reward_traj = np.array(reward_traj)
        
        V = self.critic(traj).numpy()
        
        adv_traj, rtg_traj = TRPO.reward_calc(self, reward_traj, V, V_end)
                
        return  adv_traj, rtg_traj, traj, control_traj, reward_traj
    
    
    
    
    