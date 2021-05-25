#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:24:02 2021

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
import Deep_Qlearning
from utils_main import Show_DataSet
from utils_main import Show_Training
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
import multiprocessing.pool

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
Folders = [6, 7, 11, 12, 15]
size_data = 3100
Rand_traj = 2

TrainingSet, Labels, Trajectories, Rotation, Time, _ = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'distr_only', 'no plot')
# _,_,_,_,_,_ = Show_DataSet(Folders, size_data, Rand_traj, 'simplified', 'distr_only')
_,_,_,_,_, Reward_eval_human  = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'full_coins', 'no plot')
# _,_,_,_,_,_  = Show_DataSet(Folders, size_data, Rand_traj, 'simplified', 'full_coins')
_, _, _, _, _, Reward_training_human = Show_Training(Folders, size_data, Rand_traj, 'complete', 'full_coins', 'no plot')


# %% Plot human expert

# Plot Human Day 1
episodes = np.arange(0,len(Reward_eval_human))
plt.plot(episodes, np.ones(len(episodes))*np.mean(Reward_eval_human),'k--')
plt.plot(episodes, Reward_eval_human,'g', label = 'human agent evaluation (mean = {})'.format(np.mean(Reward_eval_human)))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Human agent performance Day 2')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/Neurips/Human_Reward.eps', format='eps')
plt.show() 

# Plot Human Day 2
episodes = np.arange(0,len(Reward_training_human))
plt.plot(episodes, np.ones(len(episodes))*np.mean(Reward_training_human),'k--')
plt.plot(episodes, Reward_training_human,'g', label = 'human agent evaluation (mean = {})'.format(np.mean(Reward_training_human)))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Human agent performance Day 1')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/Neurips/Human_Reward_training.eps', format='eps')
plt.show() 

best_traj = np.argmax(Reward_eval_human)
best_subject = int(best_traj/10)
traj_within_folder = best_traj%10 
coins_location = World.Foraging.CoinLocation(Folders[best_subject], traj_within_folder+1, 'full_coins')

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
plot_data = plt.scatter(Trajectories[best_traj][0:size_data,0], Trajectories[best_traj][0:size_data,1], c=Time[best_traj][0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Human traj, Reward {}'.format(Reward_eval_human[best_traj]))
plt.savefig('Figures/Neurips/Best_human_traj.eps', format='eps')
plt.show()  

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
ax.add_artist(circle4) #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/Neurips/Coins_Only.eps', format='eps')
plt.show() 

# %% DQN from scratch

NEpisodes = 500
Subject = Folders[best_subject]
Rand_traj = traj_within_folder
seed = 36
reset = 'standard'
initial_state = Trajectories[best_traj][0,:]

agent_NN_Q_learning_buffer = Deep_Qlearning.Q_learning_NN(seed, Subject, Rand_traj)
reward_per_episode, traj, network_weights = agent_NN_Q_learning_buffer.Training_buffer(NEpisodes, seed, reset, initial_state)

# %% Save DQN from scratch
DQ_learning_results = [reward_per_episode, traj, network_weights]
with open('RL_algorithms/DeepQ_Learning/Results/Neurips_Q_learning_results_deeper.npy', 'wb') as f:
    np.save(f, DQ_learning_results)


#%% Evaluation DQN agent

# %%Learning from human
# in the following given the comparison human flat RL we try to do imitation learning 
# preprocess and encode data set

size_data = len(Trajectories[best_traj])-1
T_set = Trajectories[best_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[best_traj][0:size_data]


# %% Behavioral Cloning

observation_space_size = T_set.shape[1]
action_size = len(np.unique(Heading_set))

model_BC_from_human = keras.Sequential([             
        keras.layers.Dense(512, activation='relu', input_shape=(observation_space_size,),
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                           bias_initializer=keras.initializers.Zeros()),                                
        keras.layers.Dense(action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2)),
        keras.layers.Softmax()
                         ])              

model_BC_from_human.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

model_BC_from_human.fit(T_set, Heading_set, epochs=200)



