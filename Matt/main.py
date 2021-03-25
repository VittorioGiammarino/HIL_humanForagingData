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
        Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, reward_single_traj = World.Foraging.ProcessData(folder, experiment, 'simplified')
        Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
        TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
        Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
        Time.append(Time_single_traj)
        Trajectories.append(Training_set_single_traj_together)
        Rotation.append(Labels_single_traj.reshape(len(Labels_single_traj),1))
        Reward.append(reward_single_traj)


# %% Plot a random trajectory and reward Distribution

coins_location = World.Foraging.CoinLocation(6, 1, 'full_coins')

Rand_traj = 9 #np.random.randint(0,len(Time))
size_data = 1000

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
plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Time[Rand_traj][0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
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
plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Trajectories[Rand_traj][0:size_data,2], marker='o', cmap='bwr')
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

# %% train_pi_hi
T_set = Trajectories[Rand_traj][0:size_data,:]
pi_hi = BatchBW_HIL.NN_PI_HI(2, 3)
pi_hi_model = pi_hi.PreTraining(T_set)

options_predictions = pi_hi_model.predict(T_set)

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
plot_data = plt.scatter(T_set[:,0], T_set[:,1], c=np.argmax(options_predictions,1), marker='o', cmap='bwr')
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

# %% train_pi_lo

options = np.argmax(options_predictions,1).reshape(len(T_set))
op1 = np.where(options == 0)
op2 = np.where(options == 1)
Heading_set = Rotation[Rand_traj][:]
action_space = len(np.unique(Heading_set))

#option1
T_set_op1 = np.concatenate((T_set[op1,:], T_set[op1,:], T_set[op1,:], T_set[op1,:]), 1)[0,:,:]
Labels_op1 = np.concatenate((Heading_set[op1,:], Heading_set[op1,:],Heading_set[op1,:],Heading_set[op1,:]), 1)[0,:,:]
pi_lo_op1 = BatchBW_HIL.NN_PI_LO(action_space,3)
pi_lo_model = pi_lo_op1.PreTraining(T_set_op1, Labels_op1, 200)

#option2
T_set_op2 = np.concatenate((T_set[op2,:],T_set[op2,:],T_set[op2,:],T_set[op2,:]), 1)[0,:,:]
Labels_op2 = np.concatenate((Heading_set[op2,:],Heading_set[op2,:],Heading_set[op2,:],Heading_set[op2,:]), 1)[0,:,:]
pi_lo_op2 = BatchBW_HIL.NN_PI_LO(action_space,3)
pi_lo_model = pi_lo_op2.PreTraining(T_set_op2, Labels_op2, 200)



# %% Training
Likelihood_batch_list = []
T_set = Trajectories[Rand_traj][0:size_data,:]
Heading_set = Rotation[Rand_traj][0:size_data]
option_space = 2
M_step_epoch = 50
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(T_set, Heading_set, option_space, M_step_epoch, size_batch, optimizer, options_predictions)
N=2 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time
Likelihood_batch_list.append(likelihood)

# %%
likelihood_temp = Agent_BatchHIL.likelihood_approximation()

# %%evaluation

for traj in range(1):
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(3000,1,T_set[0,0:2])

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
    plot_data = plt.scatter(trajBatch[0][:,0], trajBatch[0][:,1], c=OptionsBatch[0][1:], marker='o', cmap='bwr')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
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
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
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