#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import World
import BatchBW_HIL
import time

# %% Preprocessing_data
Folders = [6, 7, 11, 12, 15]
size_input = 2
TrainingSet = np.empty((0,2))
Labels = np.empty((0,1))
Time = []
Trajectories = []
Rotation = []
for folder in Folders:
    for experiment in range(1,11):
        Training_set_single_traj, Labels_single_traj, Time_single_traj = World.Foraging.ProcessData(folder, experiment)
        TrainingSet = np.append(TrainingSet, 0.1*(Training_set_single_traj[0:-1]), 0)
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

# %% Training
option_space = 2
M_step_epoch = 50
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet[0:2000,:], Labels[0:2000,:], option_space, M_step_epoch, size_batch, optimizer)
N=10 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time

# %%evaluation

for traj in range(10):
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(3000,1,np.array([0, 0]))

    # Plot result

    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(trajBatch[0][:,0], trajBatch[0][:,1], c=OptionsBatch[0][1:], marker='o', cmap='Set1')
    cbar = fig.colorbar(plot_data, ticks=[0, 1])
    cbar.ax.set_yticklabels(['option 1', 'option 2'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/FiguresBatch/Traj_VS_Options_traj{}.eps'.format(traj), format='eps')
    plt.show()  

    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(trajBatch[0][:,0], trajBatch[0][:,1], c=Time[0][0:len(trajBatch[0])], marker='o', cmap='cool')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/FiguresBatch/Traj_VS_Time_traj{}.eps'.format(traj), format='eps')
    plt.show()  

# %%
BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}'.format(i))