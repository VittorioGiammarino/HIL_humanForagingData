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
from utils_main import Show_DataSet
from sklearn.preprocessing import OneHotEncoder

# %% Preprocessing_data with psi based on the coins clusters distribution  
Folders = [6] #[6, 7, 11, 12, 15]
size_data = 3100
Rand_traj = 4

TrainingSet, Labels, Trajectories, Rotation, Time, Reward = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'distr_only')
# _,_,_,_,_,_ = Show_DataSet(Folders, size_data, Rand_traj, 'simplified', 'distr_only')
# _,_,_,_,_,_  = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'full_coins')
# _,_,_,_,_,_  = Show_DataSet(Folders, size_data, Rand_traj, 'simplified', 'full_coins')


# %% one-hot encoding
# encode psi
psi = TrainingSet[:,2].reshape(len(TrainingSet[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)

# encode closest coin direction
closest_coin_direction = TrainingSet[:,3].reshape(len(TrainingSet[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)

coordinates = TrainingSet[:,0:2].reshape(len(TrainingSet[:,0:2]),2)

TrainingSet = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)

# %% train_pi_hi
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)

# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)

coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)

T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)


option_space = 2
size_input = T_set.shape[1]
# T_set =TrainingSet
pi_hi = BatchBW_HIL.NN_PI_HI(option_space, size_input)
pi_hi_model = pi_hi.PreTraining(T_set)

options_predictions = pi_hi_model.predict(T_set)


# %% train_pi_lo

# options = np.argmax(options_predictions,1).reshape(len(T_set))
# op1 = np.where(options == 0)
# op2 = np.where(options == 1)
# Heading_set = Rotation[Rand_traj][:]
# Heading_set = Labels
# action_space = len(np.unique(Heading_set))

# #option1
# T_set_op1 = np.concatenate((T_set[op1,:], T_set[op1,:], T_set[op1,:], T_set[op1,:]), 1)[0,:,:]
# Labels_op1 = np.concatenate((Heading_set[op1,:], Heading_set[op1,:],Heading_set[op1,:],Heading_set[op1,:]), 1)[0,:,:]
# # T_set_op1 = T_set[op1, :][0,:,:]
# # Labels_op1 = Heading_set[op1, :] [0,:,:]
# pi_lo_op1 = BatchBW_HIL.NN_PI_LO(action_space,3)
# pi_lo_model = pi_lo_op1.PreTraining(T_set_op1, Labels_op1, 1000)

# #option2
# T_set_op2 = np.concatenate((T_set[op2,:],T_set[op2,:],T_set[op2,:],T_set[op2,:]), 1)[0,:,:]
# Labels_op2 = np.concatenate((Heading_set[op2,:],Heading_set[op2,:],Heading_set[op2,:],Heading_set[op2,:]), 1)[0,:,:]
# # T_set_op2 = T_set[op2, :][0,:,:]
# # Labels_op2 = Heading_set[op2, :][0,:,:]
# pi_lo_op2 = BatchBW_HIL.NN_PI_LO(action_space,3)
# pi_lo_model = pi_lo_op2.PreTraining(T_set_op2, Labels_op2, 1000)



# %% Training single trajectory
Likelihood_batch_list = []
Rand_traj = 1
size_data = len(Trajectories[Rand_traj])-1
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[Rand_traj][0:size_data]
option_space = 2
M_step_epoch = 10
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-1)
# Agent_BatchHIL = BatchBW_HIL.BatchHIL_param_simplified(T_set, Heading_set, M_step_epoch, size_batch, optimizer) 
Agent_BatchHIL = BatchBW_HIL.BatchHIL(T_set, Heading_set, option_space, M_step_epoch, size_batch, optimizer, options_predictions)
N=2 #number of iterations for the BW algorithm
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
Likelihood_batch_list.append(likelihood)

# for Rand_traj in range(1,len(Trajectories)):
    
#     size_data = len(Trajectories[Rand_traj])-1
#     T_set = Trajectories[Rand_traj][0:size_data,:]
#     # encode psi
#     psi = T_set[:,2].reshape(len(T_set[:,2]),1)
#     onehot_encoder = OneHotEncoder(sparse=False)
#     onehot_encoded_psi = onehot_encoder.fit_transform(psi)
#     # encode closest coin direction
#     closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
#     onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
#     coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
#     T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
#     Heading_set = Rotation[Rand_traj][0:size_data]
#     # Agent_BatchHIL = BatchBW_HIL.BatchHIL_param_simplified(T_set, Heading_set, M_step_epoch, size_batch, optimizer) 
#     Agent_BatchHIL = BatchBW_HIL.BatchHIL(T_set, Heading_set, option_space, M_step_epoch, size_batch, optimizer, options_predictions, NN_init = 'from_network', NN_options = pi_hi_batch, NN_low = pi_lo_batch, NN_termination = pi_b_batch)
#     N=2 #number of iterations for the BW algorithm
#     pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
#     Likelihood_batch_list.append(likelihood)
    

# %%evaluation
coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')
for traj in range(1):
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(3000, 1, np.array([5, 5]), Folders[0], Rand_traj)

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
    #plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[0, 1])
    cbar.ax.set_yticklabels(['option 1', 'option 2'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/FiguresBatch/Traj_VS_Options_traj{}_new3.eps'.format(traj), format='eps')
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
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/FiguresBatch/Traj_VS_Time_traj{}_new3.eps'.format(traj), format='eps')
    plt.show()  


# %%
BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}'.format(i))
