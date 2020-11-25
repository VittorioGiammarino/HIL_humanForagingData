#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as ptch
import csv
import World

# %% Preprocessing_data
Folders = [6, 7, 11, 12, 15]
size_input = 2
Training_set = np.empty((0,2))
Labels = np.empty((0,1))
Time = []
Trajectories = []
Rotation = []
for folder in Folders:
    for experiment in range(1,11):
        Training_set_single_traj, Labels_single_traj, Time_single_traj = World.Foraging.ProcessData(folder, experiment)
        Training_set = np.append(Training_set, Training_set_single_traj[0:-1], 0)
        Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
        Time.append(Time_single_traj)
        Trajectories.append(Training_set_single_traj)
        Rotation.append(Labels)

# %% Plot a random trajectory and reward Distribution
sigma1 = 5
circle1 = plt.Circle((60, 75), 2*sigma1, color='k',  fill=False)
sigma2 = 11
circle2 = plt.Circle((-15, -50), 2*sigma2, color='k',  fill=False)
sigma3 = 18
circle3 = plt.Circle((-50, 30), 2*sigma3, color='k',  fill=False)
sigma4 = 13
circle4 = plt.Circle((49, -40), 2*sigma4, color='k',  fill=False)
  
Rand_traj = np.random.randint(0,len(Time))

fig, ax = plt.subplots()
plot_data = plt.scatter(Trajectories[Rand_traj][:,0], Trajectories[Rand_traj][:,1], c=Time[Rand_traj], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plt.xlabel('x')
plt.ylabel('y')
plt.show()  

