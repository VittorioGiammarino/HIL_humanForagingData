#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:59:17 2021

@author: vittorio
"""
import copy
import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TanhGaussianActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(TanhGaussianActor.NN_PI_LO, self).__init__()
            
            self.action_dim = action_dim
            self.l1 = nn.Linear(state_dim, 512)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(512,512)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(512, 2*action_dim)
            # nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            
            self.max_action = max_action
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            a = self.l3(a)
            mean = a[:,0:self.action_dim]
            std = torch.exp(a[:, self.action_dim:])
            output = torch.FloatTensor(self.max_action)*torch.tanh(torch.normal(mean,std))
            return output, mean, std
        
    class BehavioralCloning(object):
        def __init__(self, max_action, state_dim, action_dim, state_samples, action_samples, M_step_epoch, batch_size, l_rate_pi_lo):
            self.max_action = max_action
            self.state_dim = state_dim
            self.action_dim = action_dim 
            self.TrainingSet = state_samples
            self.Labels = action_samples
            self.epochs = M_step_epoch
            self.batch_size = batch_size
            # define policy
            self.pi_lo = TanhGaussianActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            # define optimizer 
            self.learning_rate = l_rate_pi_lo
            self.pi_lo_optimizer = torch.optim.Adam(self.pi_lo.parameters(), lr=self.learning_rate)
                
        def reset_learning_rate(self, l_rate_pi_lo):
            self.pi_lo_optimizer = torch.optim.Adam(self.pi_lo.parameters(), lr=l_rate_pi_lo)
                   
        def get_prob_pi_lo(self, mean, std, actions):
            action_dim = mean.shape[1]
            a_prob = torch.ones(len(actions),1)
            denominator = torch.ones(len(actions),1)
            for a in range(action_dim):
                a_prob *= (torch.exp(-(actions[:,a]-mean[:,a])**2/(2*std[:,a]**2))/(torch.sqrt(2*torch.FloatTensor([np.pi])*std[:,a]**2))).reshape(len(actions),1)
                denominator *= (torch.FloatTensor([self.max_action[a]])*(1-(torch.tanh(actions[:,a]))**2)).reshape(len(actions),1)
            return a_prob/((torch.abs(denominator)).clamp(1e-10,1e10))
        
        def get_log_likelihood_pi_lo(self, mean, std, actions):
            action_dim = mean.shape[1]
            Log_likelihood_a_prob = torch.zeros(len(actions),1)
            Log_denominator = torch.zeros(len(actions),1)
            for a in range(action_dim):
                 Log_likelihood_a_prob += (-0.5*((actions[:,a]-mean[:,a])/((std[:,a]).clamp(1e-20,1e20)))**2  - torch.log((std[:,a]).clamp(1e-20,1e20)) - torch.log(torch.sqrt(2*torch.FloatTensor([np.pi])))).reshape(len(actions),1) 
                 Log_denominator += (torch.log((torch.FloatTensor([self.max_action[a]])*(1-(torch.tanh(actions[:,a]))**2)).clamp(1e-10,1e10))).reshape(len(actions),1)
            return Log_likelihood_a_prob - Log_denominator

        def Loss(self, NN_actions, T, TrainingSet, Labels):
    # =============================================================================
    #         Compute batch loss function to minimize
    # =============================================================================
            _, mean, std = NN_actions(TrainingSet)
            log_likelihood_pi_lo = TanhGaussianActor.BehavioralCloning.get_log_likelihood_pi_lo(self, mean, std, Labels).cpu()
            loss = -torch.mean(log_likelihood_pi_lo)
                        
            return loss    
    
        def OptimizeLossBatch(self):
    # =============================================================================
    #         optimize loss in mini-batches
    # =============================================================================
            loss = 0
            n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
            
            for epoch in range(self.epochs):
                # print("\nStart of epoch %d" % (epoch,))
                    
                for n in range(n_batches):
                    # print("\n Batch %d" % (n+1,))
                    TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                    Labels = torch.FloatTensor(self.Labels[n*self.batch_size:(n+1)*self.batch_size]).to(device)
                    loss = TanhGaussianActor.BehavioralCloning.Loss(self, self.pi_lo, self.batch_size, TrainingSet, Labels)
            
                    self.pi_lo_optimizer.zero_grad()
                    loss.backward()  
                    # for name, param in self.pi_lo.named_parameters():
                    #     print(name, param.grad)                    
                    self.pi_lo_optimizer.step()  
             
            T = self.TrainingSet.shape[0]
            TrainingSet = torch.FloatTensor(self.TrainingSet).to(device)
            Labels = torch.FloatTensor(self.Labels).to(device)
            loss = TanhGaussianActor.BehavioralCloning.Loss(self, self.pi_lo, T, TrainingSet, Labels)
            return loss   
                
        def train(self):    
            loss = TanhGaussianActor.BehavioralCloning.OptimizeLossBatch(self)     
            print('Maximization done, Loss:',float(loss))    
            return loss
        
        def StochasticSampleTrajMDP(self, env, max_epoch_per_traj, number_of_trajectories, reset = 'random', initial_state = np.array([0,0,0,8])):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = env.reset(reset, initial_state)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,self.action_dim))
                Reward = 0
                
                for k in range(0, max_epoch_per_traj):
                    state = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
                    # draw action
                    output, _, _ = self.pi_lo(state)
                    u_tot = np.append(u_tot, output.cpu().data.numpy(), axis=0)
                    action = output.cpu().data.flatten().numpy()
                    # given action, draw next state
                    obs, reward, done, _ = env.step(action)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        break
        
                traj[t] = x
                control[t]=u_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Reward_array    







