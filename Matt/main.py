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
import csv

# %% Training_set_preprocessing
with open("4_walls_coins_task/FONC_06_DeID/fMRI/runNumber2_position.txt") as f:
    data_raw = f.readlines()
    
for i in range(len(data_raw)):
    row = data_raw[i][1:]
    row = row.replace(']', ',')
    data_raw[i] = row

agent_data = csv.reader(data_raw)
Training_set = np.empty((0,2))
time = np.empty((0,1))

for row in agent_data:
    Training_set = np.append(Training_set, np.array([[np.round(float(row[0])), np.round(float(row[2]))]]), 0)
    time = np.append(time, float(row[3]))
    
State_space, Training_set_index = np.unique(Training_set, return_index=True, axis = 0)    
Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
time_cleaned = time[np.sort(Training_set_index)]

# %% Labels_preprocessing

action_space = 8
action_range = 360/action_space

# determine ganularity of the action space
actions = np.zeros(1)
actions_rad = np.zeros(1)
actions_slots = (action_range/2)*np.ones(1)

for i in range(action_space):
    step = action_range
    step_rad = np.divide((step)*np.pi,180)
    actions = np.append(actions, actions[i]+step)
    actions_rad = np.append(actions_rad, actions_rad[i]+step_rad)
    actions_slots = np.append(actions_slots, actions_slots[i]+step)

Labels = np.empty((0,1))
for i in range(len(Training_set_cleaned)-1):
    state = Training_set_cleaned[i,:]
    state_next = Training_set_cleaned[i+1,:]
    Transition = np.zeros((action_space,2))
    Transition[0,0] = state[0] + 1
    Transition[0,1] = state[1] + 0
    Transition[1,0] = state[0] + 1
    Transition[1,1] = state[1] + 1
    Transition[2,0] = state[0] + 0
    Transition[2,1] = state[1] + 1
    Transition[3,0] = state[0] - 1
    Transition[3,1] = state[1] + 1
    Transition[4,0] = state[0] - 1
    Transition[4,1] = state[1] + 0
    Transition[5,0] = state[0] - 1
    Transition[5,1] = state[1] - 1
    Transition[6,0] = state[0] + 0
    Transition[6,1] = state[1] - 1
    Transition[7,0] = state[0] + 1
    Transition[7,1] = state[1] - 1
    index_x = np.where(state_next[0] == Transition[:,0])[0]
    index_y = np.where(state_next[1] == Transition[:,1])[0]
    index = np.intersect1d(index_y,index_x)
    if index.size == 0:
        index = 8
    Labels = np.append(Labels, actions[index])

# %% Simulate dynamics

sim = np.empty((0,2))
init = Training_set_cleaned[0,:]
sim = np.append(sim, init.reshape(1,2), 0)
for i in range(len(Labels)):
    Action = Labels[i]
    Transition = np.zeros((action_space+1,2))
    Transition[0,0] = sim[i,0] + 1
    Transition[0,1] = sim[i,1] + 0
    Transition[1,0] = sim[i,0] + 1
    Transition[1,1] = sim[i,1] + 1
    Transition[2,0] = sim[i,0] + 0
    Transition[2,1] = sim[i,1] + 1
    Transition[3,0] = sim[i,0] - 1
    Transition[3,1] = sim[i,1] + 1
    Transition[4,0] = sim[i,0] - 1
    Transition[4,1] = sim[i,1] + 0
    Transition[5,0] = sim[i,0] - 1
    Transition[5,1] = sim[i,1] - 1
    Transition[6,0] = sim[i,0] + 0
    Transition[6,1] = sim[i,1] - 1
    Transition[7,0] = sim[i,0] + 1
    Transition[7,1] = sim[i,1] - 1
    Transition[8,:] = Training_set_cleaned[i,:]
    index = np.where(Action == actions)[0]
    next_state = Transition[index,:]
    sim = np.append(sim,next_state,0)
  
# %%

fig = plt.figure()
plot_data = plt.scatter(Training_set_cleaned[:,0], Training_set_cleaned[:,1], c=time_cleaned, marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()    


fig = plt.figure()
plot_data = plt.scatter(sim[:,0], sim[:,1], c=time_cleaned, marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()  

