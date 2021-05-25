#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:34:06 2021

@author: vittorio
"""

from tensorflow import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
import BatchBW_HIL
import OnlineBW_HIL
import OnlineBW_HIL_Neural
from utils_main import Show_DataSet
from utils_main import Show_Training
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
import multiprocessing.pool

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_results_second_attempt.npy', 'rb') as f:
    DeepSoftOC_learning_results = np.load(f, allow_pickle=True).tolist()
    
N_agents = 10

best_reward = np.zeros(N_agents)
best_traj = [[None]*1 for _ in range(N_agents)]
best_option = [[None]*1 for _ in range(N_agents)]
best_temrination = [[None]*1 for _ in range(N_agents)]
best_agent = np.zeros(N_agents)
best_pi_hi = [[None]*1 for _ in range(N_agents)]
best_pi_lo = [[None]*1 for _ in range(N_agents)]
best_pi_b = [[None]*1 for _ in range(N_agents)]
seeds = [31] #[13, 29, 33, 36, 27, 21, 15, 31]

for i in range(len(DeepSoftOC_learning_results)):
    for j in range(len(DeepSoftOC_learning_results[i][0])):
        temp = DeepSoftOC_learning_results[i][0][j]
        k = np.argmin(best_reward)
        if temp>best_reward[k]:
            best_reward[k]=DeepSoftOC_learning_results[i][0][j]
            best_traj[k]=DeepSoftOC_learning_results[i][1][j]
            best_option[k]=DeepSoftOC_learning_results[i][2][j]
            best_temrination[k]=DeepSoftOC_learning_results[i][3][j]
            best_pi_hi[k] = DeepSoftOC_learning_results[i][4][j]
            best_pi_lo[k] = DeepSoftOC_learning_results[i][5][j]
            best_pi_b[k] = DeepSoftOC_learning_results[i][6][j]
            best_agent[k]=j

coins_location = World.Foraging.CoinLocation(6, 2+1, 'full_coins') #np.random.randint(0,len(Time))

DOC_pi_hi = best_pi_hi[0]
DOC_pi_lo = best_pi_lo[0]
DOC_pi_b = best_pi_b[0]
Training_DOC = DeepSoftOC_learning_results[0][0]

# %%

with open('DOC/DOC_pi_hi.npy', 'wb') as f:
    np.save(f, DOC_pi_hi)
    
with open('DOC/DOC_pi_lo.npy', 'wb') as f:
    np.save(f, DOC_pi_lo)

with open('DOC/DOC_pi_b.npy', 'wb') as f:
    np.save(f, DOC_pi_b)
    
with open('DOC/Training_DOC.npy', 'wb') as f:
    np.save(f, Training_DOC)
    
    
    