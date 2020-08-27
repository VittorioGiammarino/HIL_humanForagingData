#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:09:39 2020

@author: vittorio
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import HierarchicalImitationLearning as hil
import Simulation as sim
import BehavioralCloning as bc

# %% Load Data
folder_name = 'WILD_P03_water_4'
action_space = 8
Labels_dict_degrees, Labels_dict_rad, data_manip = hil.PreprocessData(folder_name, action_space)

water_locations = hil.WaterLocation(folder_name)

# %% Plot entire data set

fig = plt.figure()
plot_data = plt.scatter(data_manip[:,2], data_manip[:,3], c=data_manip[:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_{}_wholeData.eps'.format(folder_name), format='eps')
plt.show()

# %%
folder_name2 = 'WILD_P03_water_8'
Labels_dict_degrees,Labels_dict_rad,data_manip2 = hil.PreprocessData(folder_name2, action_space)

fig = plt.figure()
plot_data = plt.scatter(data_manip2[:,2], data_manip2[:,3], c=data_manip2[:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_{}_wholeData.eps'.format(folder_name2), format='eps')
plt.show()

# %%
Expert_trajs = hil.BatchTrainingSet_velocity(data_manip)
Expert_trajs_generated = hil.EvaluateActionSpaceDiscretization(Expert_trajs)

# %% Plot Expert's Data

traj = 0

fig = plt.figure()
plot_traj = plt.scatter(Expert_trajs[traj][:,2], Expert_trajs[traj][:,3], c=Expert_trajs[traj][:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_traj, ticks=[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
cbar.ax.set_yticklabels(['t = 0','t = 50', 't = 100', 't = 150', 't = 200', 't = 250', 't = 300', 't = 350', 't = 400', 't = 450', 't = 500'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')
plt.show()

# %% Plot Expert's Data generated by discretization action space

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=3800)

fig = plt.figure()
plot_traj = plt.scatter(Expert_trajs_generated[traj][:,2], Expert_trajs_generated[traj][:,3], c=Expert_trajs_generated[traj][:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_traj, ticks=[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
cbar.ax.set_yticklabels(['t = 0','t = 50', 't = 100', 't = 150', 't = 200', 't = 250', 't = 300', 't = 350', 't = 400', 't = 450', 't = 500'])
plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/ExpertGenerated_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')
plt.show()

def animation_frame(i, x, o):
    plot_traj.set_offsets(x[0:i,:])
    plot_traj.set_sizes(10*np.ones(i))
    plot_traj.set_array(o[0:i])
    return plot_traj

animation = anim.FuncAnimation(fig, func = animation_frame, frames=Expert_trajs_generated[traj][:,1].shape[0], fargs=(Expert_trajs_generated[traj][:,2:4], Expert_trajs_generated[traj][:,1]))
animation.save('Videos/VideosExpert/ExpertPolicy_{}_traj{}_ActionSpace{}.mp4'.format(folder_name,traj,action_space), writer=writer)

# %%

data_manip_tot = np.append(data_manip,data_manip2,0)

# %%
# TrainingSet = (Expert_trajs_generated[traj][:,2:4])/1000
# labels = Expert_trajs_generated[traj][:,5]
TrainingSet = (data_manip_tot[:,2:4])/1000
labels = data_manip_tot[:,5]
time = Expert_trajs[traj][:,1]
speed = Expert_trajs[traj][:,6]/1000

# %% Behavioral Cloning

size_input = TrainingSet.shape[1]
model1 = bc.NN1(action_space, size_input)
model2 = bc.NN2(action_space, size_input)
model3 = bc.NN3(action_space, size_input)

# train the models
# model1.fit(TrainingSet, labels, epochs=10000)
# encoded = tf.keras.utils.to_categorical(labels)
# model2.fit(TrainingSet, encoded, epochs=300)
# model3.fit(TrainingSet, encoded, epochs=100)

# model1.save('Variables_saved/BehavioralCloning/model1_action{}'.format(action_space))

# %%
model_BC = tf.keras.models.load_model('Variables_saved/BehavioralCloning/model1_action{}'.format(action_space))

# %% Initialization
option_space = 3
# action space
termination_space = 2
size_input = TrainingSet.shape[1]

NN_options = hil.NN_options(option_space, size_input)
NN_actions = hil.NN_actions(action_space, size_input)
NN_termination = hil.NN_termination(termination_space, size_input)

N=3 #Iterations
zeta = 0.1 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env = 'human_foraging'
max_epoch = len(time)

ED = hil.Experiment_design(labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, 
                           Triple, LAMBDAS, ETA, env, max_epoch, speed, time)

lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)   
    
# %%
# x,u = sim.VideoFlatPolicy(model_BC,ED.max_epoch,ED.size_input,Labels_dict_rad, ED.speed, ED.time)

# fig = plt.figure()                                                    
# plot_action = plt.scatter(x[:,0]*1000, x[:,1]*1000, c=ED.time, marker='o', cmap='cool');
# cbar = fig.colorbar(plot_action, ticks=[10, 50, 130])
# cbar.ax.set_yticklabels(['10sec', '50sec', '130sec'])
# plot_water = plt.plot(water_locations[:,0], water_locations[:,1], 'bx')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('Figures/FiguresBC/BCpolicy_{}and{}_ActionSpace{}.eps'.format(folder_name,folder_name2,action_space), format='eps')

# def animation_frame_BC(i, x, o):
#     plot_action.set_offsets(x[0:i,:]*1000)
#     plot_action.set_sizes(10*np.ones(i))
#     plot_action.set_array(o[0:i])
#     return plot_action

# animation = anim.FuncAnimation(fig, func = animation_frame_BC, frames=x.shape[0], fargs=(x, ED.time))
# animation.save('Videos/VideosBC/BCpolicy_{}and{}_ActionSpace{}.mp4'.format(folder_name,folder_name2,action_space), writer=writer)

# %% HMM order estimation

Likelihood = np.empty(0) 
# %%
Model_orders = [2]

for d in Model_orders:
    Likelihood = np.append(Likelihood, -hil.HMM_order_estimation(d, ED))
    
# with open('Variables_saved/likelihood.npy', 'wb') as f:
#     np.save(f,[Likelihood, Model_orders])


# %% Plot Figure
 	
with open('Variables_saved/likelihood.npy', 'rb') as f:
    Likelihood, Model_orders = np.load(f)

fig = plt.figure()
plot_action = plt.plot(Model_orders, Likelihood,'o--');
plt.xlabel('Model Order')
plt.ylabel('Lower bound for the Likelihood')
plt.savefig('Figures/FiguresHIL/Likelihood_over_order.eps', format='eps')
plt.show()    
    
# %%

NN_Termination, NN_Actions, NN_Options = hil.BaumWelch(ED, lambdas, eta)


# %%
Trained_triple = hil.Triple(NN_Options, NN_Actions, NN_Termination)
Trajs=1
[trajBW, controlBW, OptionBW, 
 TerminationBW, flagBW] = sim.HierarchicalPolicySim(Trained_triple, ED.zeta, ED.mu, ED.max_epoch, Trajs, ED.option_space, ED.size_input, 
                                                    Labels_dict_rad, ED.speed, ED.time)    
# %%

xBW = np.empty((0))
yBW = np.empty((0))
actionBW = np.empty((0))
optionsBW = np.empty((0))
terminationsBW = np.empty((0))
for j in range(len(trajBW)):
    for i in range(len(trajBW[j])-1):
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

# %%

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=optionsBW[:], marker='o', cmap='Set1');
cbar = fig.colorbar(plot_action, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Option1', 'Option2', 'Option3'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHIL/HILpolicy_options_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=actionBW[:], marker='o', cmap='Set1');
cbar = fig.colorbar(plot_action, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHIL/HILpolicy_actions_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')

fig = plt.figure()                                                    
plot_action = plt.scatter(xBW[:]*1000, yBW[:]*1000, c=terminationsBW[:], marker='o', cmap='copper');
cbar = fig.colorbar(plot_action, ticks=[0, 1])
cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHIL/HILpolicy_termination_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')
                                                    
# %%

x, u, o, b = sim.VideoHierarchicalPolicy(Trained_triple, ED.zeta, ED.mu, ED.max_epoch, ED.option_space, ED.size_input, 
                                                    Labels_dict_rad, ED.speed, ED.time)
    
# %%

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=3800)

fig = plt.figure()
ax1 = plt.subplot(311)
plot_action = plt.scatter(x[:,0]*1000, x[:,1]*1000, c=o[1:], marker='x', cmap='Set1');
cbar = fig.colorbar(plot_action, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Option1', 'Option2', 'Option3'])
plt.xlabel('x')
plt.ylabel('y')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(312, sharex=ax1)
plot_option = plt.scatter(x[0:-1,0]*1000, x[0:-1,1]*1000, c=u, marker='x', cmap='Set1');
cbar = fig.colorbar(plot_option, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
plt.xlabel('x')
plt.ylabel('y')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(313, sharex=ax1)
plot_termination = plt.scatter(x[:,0]*1000, x[:,1]*1000, c=b, marker='x', cmap='copper');
cbar = fig.colorbar(plot_termination, ticks=[0, 1])
cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresHIL/HILpolicy_subfigs_{}_traj{}_ActionSpace{}.eps'.format(folder_name,traj,action_space), format='eps')
plt.show()
    
#%% Animation

def animation_frame_Hierarchical(i, x, o, u, b):
    plot_action.set_offsets(x[0:i,:]*1000)
    plot_action.set_sizes(10*np.ones(i))
    plot_action.set_array(o[0:i])
    plot_option.set_offsets(x[0:i,:]*1000)
    plot_option.set_sizes(10*np.ones(i))
    plot_option.set_array(u[0:i])
    plot_termination.set_offsets(x[0:i,:]*1000)
    plot_termination.set_sizes(10*np.ones(i))
    plot_termination.set_array(b[0:i])
    return plot_action, plot_option, plot_termination,

animation = anim.FuncAnimation(fig, func = animation_frame_Hierarchical, frames=b.shape[0], fargs=(x, o, u, b))
animation.save('Videos/VideosHIL/HILpolicy_subfigs_{}_traj{}_ActionSpace{}.mp4'.format(folder_name,traj,action_space), writer=writer)    
    
    
    
    
    
    
    
    
    
    
    
    

   