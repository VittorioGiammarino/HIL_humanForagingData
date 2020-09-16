#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:44:03 2020

@author: vittorio
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import HierarchicalImitationLearning as hil
import Simulation as sim
import BehavioralCloning as bc

folder_name = 'WILD_P03_water_4'
action_space = 8
Labels_dict_degrees, Labels_dict_rad, data_manip = hil.PreprocessData(folder_name, action_space)
water_locations = hil.WaterLocation(folder_name)

#%%
option_space=4
lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)   

NN_options, NN_actions, NN_termination = hil.Triple.load(lambdas.numpy()[0], eta.numpy(), 'Hard_coded_policy')

# %% Initialization for learning from HC Policy

option_spaceHC=4
muHC = np.array([1/4, 1/4, 1/4, 1/4])
#time = Expert_trajs[traj][:,1]
time_init = 0
time_end = 1000
sampling = 0.2
samples = time_end/sampling
timeHC = np.linspace(time_init,time_end,int(samples))
speed_value = 600
speedHC = speed_value*np.ones(int(samples))#Expert_trajs[traj][:,6]
tol = 100
nTrajs = 100
max_epochHC = int(samples)
zeta = 0.001
size_input = 2

#%% Simulate policy learnt

Trained_triple = hil.Triple(NN_options, NN_actions, NN_termination)
Trajs=1
[trajBW, controlBW, OptionBW, 
 TerminationBW, flagBW] = sim.HierarchicalPolicySim(Trained_triple, zeta, muHC, 2500, Trajs, option_spaceHC, size_input, 
                                                    Labels_dict_rad, speedHC/1000, timeHC)    

# %%

xBW = np.empty((0))
yBW = np.empty((0))
actionBW = np.empty((0))
optionsBW = np.empty((0))
terminationsBW = np.empty((0))
for j in range(len(trajBW)):
    for i in range(len(trajBW[j])-401):
        xBW = np.append(xBW, trajBW[j][i][0])
        yBW = np.append(yBW, trajBW[j][i][1])
        actionBW = np.append(actionBW, controlBW[j][i])
        optionsBW = np.append(optionsBW, OptionBW[j][i])
        terminationsBW = np.append(terminationsBW, TerminationBW[j][i])
 
length_traj = np.empty((0))
for j in range(len(trajBW)):
    length_traj = np.append(length_traj, len(trajBW[j][:]))
averageBW = np.divide(np.sum(length_traj),len(length_traj))
success_percentageBW = np.divide(np.sum(flagBW),len(length_traj))


traj=2

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=optionsBW[:], marker='o', cmap='Set1');
cbar = fig.colorbar(plot_action, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Option1', 'Option2', 'Option3', 'Option4'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHC_HIL/HILpolicy_options_HC_traj{}_ActionSpace{}.eps'.format(traj,action_space), format='eps')

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=actionBW[:], marker='o', cmap='Set1');
cbar = fig.colorbar(plot_action, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHC_HIL/HILpolicy_actions_HC_traj{}_ActionSpace{}.eps'.format(traj,action_space), format='eps')

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=terminationsBW[:], marker='o', cmap='copper');
cbar = fig.colorbar(plot_action, ticks=[0, 1])
cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHC_HIL/HILpolicy_termination_HC_traj{}_ActionSpace{}.eps'.format(traj,action_space), format='eps')

    