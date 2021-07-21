#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:57:33 2021

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
from mpl_toolkits.mplot3d import Axes3D

# %% Preprocessing_data with psi based on the coins clusters distribution  

def Show_DataSet(Folders, coins):
# =============================================================================
#     action_space = 'complete', which considers an heading direction every 45deg
#                    'simplified', which considers an action every 90
#     coins = 'distr_only' coins on the gaussians distributions
#             'full_coins' all the coins in the original experiment
# =============================================================================
    TrainingSet = np.empty((0,4))
    Labels = np.empty((0,2))
    Time = []
    Real_Time = []
    Trajectories = []
    Real_Traj = []
    Rotation = []
    Reward = []
    Real_reward = []
    for folder in Folders:
        for experiment in range(1,11):
            True_traj, True_time, Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, coin_direction_single_traj, reward_single_traj, real_reward = World.Foraging.ProcessData(folder, experiment, coins)
            Training_set_single_traj_together = np.concatenate((Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1), coin_direction_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
            TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
            Labels = np.append(Labels, Labels_single_traj, 0)
            Time.append(Time_single_traj)
            Trajectories.append(Training_set_single_traj_together)
            Rotation.append(Labels_single_traj)
            Reward.append(reward_single_traj)
            Real_reward.append(real_reward)
            Real_Traj.append(True_traj)
            Real_Time.append(True_time)
    
    return TrainingSet, Labels, Trajectories, Rotation, Time, Reward, Real_Traj, Real_reward, Real_Time