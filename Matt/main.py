#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
import BatchBW_HIL
import time

# %% Preprocessing_data
Folders = [6] #[6, 7, 11, 12, 15]
size_input = 2
TrainingSet = np.empty((0,3))
Labels = np.empty((0,1))
Time = []
Trajectories = []
Rotation = []
Reward = []
for folder in Folders:
    for experiment in range(1,11):
        Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, reward_single_traj = World.Foraging.ProcessData(folder, experiment)
        Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
        TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
        Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
        Time.append(Time_single_traj)
        Trajectories.append(Training_set_single_traj_together)
        Rotation.append(Labels)
        Reward.append(reward_single_traj)

# %% Plot a random trajectory and reward Distribution

coins_location = World.Foraging.CoinLocation(6, 1)

Rand_traj = 4 #np.random.randint(0,len(Time))

sigma1 = 0.5
circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)

fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(Trajectories[Rand_traj][:,0], Trajectories[Rand_traj][:,1], c=Time[Rand_traj][:-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_Time_traj{}.eps'.format(Rand_traj), format='eps')
plt.show()  

# %%
sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig3, ax3 = plt.subplots()
plot_data = plt.scatter(Trajectories[Rand_traj][:,0], Trajectories[Rand_traj][:,1], c=Trajectories[Rand_traj][:,2], marker='o', cmap='bwr')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig3.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['no coins', 'see coins'])
ax3.add_artist(circle1)
ax3.add_artist(circle2)
ax3.add_artist(circle3)
ax3.add_artist(circle4)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View_traj{}.eps'.format(Rand_traj), format='eps')
plt.show()  

# %% Training
option_space = 2
M_step_epoch = 50
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet[:,:], Labels[:,:], option_space, M_step_epoch, size_batch, optimizer)
N=20 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time

# %%evaluation

for traj in range(10):
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(3000,1,np.array([0, 0]))

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