#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:25:55 2021

@author: vittorio
"""

import World 
import BatchBW_HIL 
import OnlineBW_HIL
import numpy as np
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
from tensorflow import keras
import multiprocessing
import multiprocessing.pool

# %% Expert Data

with open('DataFromExpert/TrainingSet_Array.npy', 'rb') as f:
    TrainingSet_Array = np.load(f, allow_pickle=True).tolist()
    
with open('DataFromExpert/Labels_Array.npy', 'rb') as f:
    Labels_Array = np.load(f, allow_pickle=True).tolist()
    
with open('Comparison/Batch/results_batch.npy', 'rb') as f:
    results_batch = np.load(f, allow_pickle=True).tolist()
    
# %% pre-processing

List_TimeBatch = []

for seed in range(len(results_batch)):
    if seed == 0:
        List_TimeBatch.append(results_batch[seed][0][0])
    else:
        List_TimeBatch[0] = np.add(List_TimeBatch[0],results_batch[seed][0][0])
    
# normalize 
List_TimeBatch[0] = np.divide(List_TimeBatch[0],len(results_batch))
   
# %%
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = np.array([1, 2, 5, 10]) #number of trajectories generated

#%%

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


def DifferentTrainingSet(i, nTraj, TrainingSet_tot, Labels_tot, TimeBatch, seed):
    max_epoch = 200
    TrainingSet = np.concatenate((TrainingSet_tot[0:max_epoch*nTraj[i],:],TrainingSet_tot[0:max_epoch*nTraj[i],:],TrainingSet_tot[0:max_epoch*nTraj[i],:]),axis=0)
    Labels = np.concatenate((Labels_tot[0:max_epoch*nTraj[i]],Labels_tot[0:max_epoch*nTraj[i]],Labels_tot[0:max_epoch*nTraj[i]]),axis=0)
    option_space = 2
        
    #Stopping Time
    StoppingTime = TimeBatch[i]
        
    # Online BW for HIL with tabular parameterization: Training
    M_step_epoch = 30
    optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
    Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer)
    T_min = len(TrainingSet)/(3) - 20
    start_online_time = time.time()
    pi_hi_online, pi_lo_online, pi_b_online, likelihood_online, time_per_iteration = Agent_OnlineHIL.Online_Baum_Welch_together(T_min, StoppingTime)
    end_online_time = time.time()
    Online_time = end_online_time-start_online_time
    # Time_array_online = np.append(Time_array_online, Online_time)  
    # Likelihood_online_list.append(likelihood_online)
    # time_likelihood_online_list.append(time_per_iteration)
    
    # Batch Agent Evaluation
    nTraj_eval = 100
    # Online Agent Evaluation
    OnlineSim = World.LunarLander.Simulation(pi_hi_online, pi_lo_online, pi_b_online, Labels)
    [trajOnline, controlOnline, OptionsOnline, 
    TerminationOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch, nTraj_eval, seed)
    AverageRewardOnline = np.sum(rewardOnline)/nTraj_eval  
    STDOnline = np.std(rewardOnline)
    # RewardOnline_array = np.append(RewardOnline_array, AverageRewardOnline)
    # STDOnline_array = np.append(STDOnline_array, STDOnline)
    
    return Online_time, likelihood_online, time_per_iteration, AverageRewardOnline, STDOnline


def train(seed, TrainingSet_Array, Labels_Array, List_TimeBatch, max_epoch, nTraj, i):
    #seed
    List_TimeOnline = []
    List_RewardOnline = []
    List_STDOnline = []
    List_LikelihoodOnline = []
    List_TimeLikelihoodOnline = []

    #given seed, trajectories
    Time_array_online = np.empty((0))
    RewardOnline_array = np.empty((0))
    STDOnline_array = np.empty((0))
    Likelihood_online_list = []
    time_likelihood_online_list =[]

    TrainingSet_tot = TrainingSet_Array[seed]
    Labels_tot = Labels_Array[seed]
    TimeBatch = List_TimeBatch[0]
        
    pool = multiprocessing.Pool(processes=1)
    args = [(i, nTraj, TrainingSet_tot, Labels_tot, TimeBatch, seed)]
    givenSeed_training_results = pool.starmap(DifferentTrainingSet, args) 
    
    pool.close()
    pool.join()
    
    Time_array_online = np.append(Time_array_online, givenSeed_training_results[0][0]) 
    Likelihood_online_list.append(givenSeed_training_results[0][1])
    time_likelihood_online_list.append(givenSeed_training_results[0][2])
    RewardOnline_array = np.append(RewardOnline_array, givenSeed_training_results[0][3])
    STDOnline_array = np.append(STDOnline_array, givenSeed_training_results[0][4])
        
    List_TimeOnline.append(Time_array_online)
    List_RewardOnline.append(RewardOnline_array)
    List_STDOnline.append(STDOnline_array)
    List_LikelihoodOnline.append(Likelihood_online_list)
    List_TimeLikelihoodOnline.append(time_likelihood_online_list)
        
    return List_TimeOnline, List_RewardOnline, List_STDOnline, List_LikelihoodOnline, List_TimeLikelihoodOnline

Nseed = 10
results_online = []
for i in range(len(nTraj)):
    pool = MyPool(Nseed)
    args = [(seed, TrainingSet_Array, Labels_Array, List_TimeBatch, max_epoch, nTraj, i) for seed in range(Nseed)]
    partial_results = pool.starmap(train, args) 
    pool.close()
    pool.join()
    
    results_online.append(partial_results)

# %%

with open('Comparison/Online/results_online.npy', 'wb') as f:
    np.save(f, results_online)
    
