#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:42:58 2021

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

# %%
version = 'batch'

pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN_preinit')
pi_lo_batch = []
pi_b_batch = []
option_space = 2
for i in range(option_space):
    pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_preinit'.format(i)))
    pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_preinit'.format(i)))
    
pi_hi_online_eval = OnlineBW_HIL.NN_PI_HI.load('Models/Saved_Model_Online/pi_hi_online_NN_preinit')
pi_lo_online_eval = []
pi_b_online_eval = []
option_space = 2
for i in range(option_space):
    pi_lo_online_eval.append(OnlineBW_HIL.NN_PI_LO.load('Models/Saved_Model_Online/pi_lo_NN_{}_online_preinit'.format(i)))
    pi_b_online_eval.append(OnlineBW_HIL.NN_PI_B.load('Models/Saved_Model_Online/pi_b_NN_{}_online_preinit'.format(i)))

def evaluateHIL_fromHuman(seed, Folder, Rand_traj, NEpisodes, initial_state, pi_hi_batch_weights, pi_lo_batch_weights1, pi_lo_batch_weights2, pi_b_batch_weights1, pi_b_batch_weights2):
    
    option_space = 2
    termination_space = 2
    observation_space_size = 13
    action_size = 8
    
    pi_hi = BatchBW_HIL.NN_PI_HI(option_space, observation_space_size)
    pi_lo1 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_lo2 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_b1 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    pi_b2 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    
    pi_hi_batch = pi_hi.NN_model()
    pi_hi_batch.set_weights(pi_hi_batch_weights)
    
    pi_lo_batch = []
    pi_lo1_model = pi_lo1.NN_model()
    pi_lo1_model.set_weights(pi_lo_batch_weights1)
    pi_lo_batch.append(pi_lo1_model)
    pi_lo2_model = pi_lo2.NN_model()
    pi_lo2_model.set_weights(pi_lo_batch_weights2)
    pi_lo_batch.append(pi_lo2_model)
    
    pi_b_batch = []
    pi_b1_model = pi_b1.NN_model()
    pi_b1_model.set_weights(pi_b_batch_weights1)
    pi_b_batch.append(pi_b1_model)
    pi_b2_model = pi_b2.NN_model()
    pi_b2_model.set_weights(pi_b_batch_weights2)
    pi_b_batch.append(pi_b2_model)
    
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(seed, 3000, NEpisodes, initial_state[0:2], Folder, Rand_traj)
        
    return trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch
   
if version == 'batch': 
    NEpisodes = 100
    Nseed=20
    folder = 15
    traj = 9
    initial_state = np.array([0, -2.6, 0, 8]) #np.array([-5.3, 6.7, 1, 6])
    Ncpu = Nseed
    pool = MyPool(Ncpu)
    args = [(seed, folder, traj, NEpisodes, initial_state, pi_hi_batch.get_weights(), pi_lo_batch[0].get_weights(), pi_lo_batch[1].get_weights(), pi_b_batch[0].get_weights(), pi_b_batch[1].get_weights()) for seed in range(Nseed)]
    HIL_from_human_evaluation_results_batch = pool.starmap(evaluateHIL_fromHuman, args) 
    pool.close()
    pool.join()
    
if version == 'online': 
    NEpisodes = 100
    Nseed=20
    folder = 15
    traj = 9
    initial_state = np.array([0, -2.6, 0, 8]) #np.array([-5.3, 6.7, 1, 6])
    Ncpu = Nseed
    pool = MyPool(Ncpu)
    args = [(seed, folder, traj, NEpisodes, initial_state, pi_hi_online_eval.get_weights(), pi_lo_online_eval[0].get_weights(), pi_lo_online_eval[1].get_weights(), pi_b_online_eval[0].get_weights(), pi_b_online_eval[1].get_weights()) for seed in range(Nseed)]
    HIL_from_human_evaluation_results_online = pool.starmap(evaluateHIL_fromHuman, args) 
    pool.close()
    pool.join()