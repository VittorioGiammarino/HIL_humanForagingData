#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:09:39 2020

@author: vittorio
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# %% Download Data
folder_name = 'WILD_P03_crow_2'
f = open('{}/moving_actor.csv'.format(folder_name))
moving_actor = csv.reader(f)

data = np.empty((0,5))
data_manip = np.empty((0,5))

for row in moving_actor:
    try:
        temp = np.empty((1,5))
        if float(row[6])<0:
            row[6] = 360 + float(row[6])
        temp[0,:] = [row[0],row[1],row[3],row[4],row[6]]
        data = np.append(data, temp, 0)
        data_manip = np.append(data_manip, temp, 0)
    except:
        print('the data type is not a float')
        
        
# %%

fig = plt.figure()
plot_data = plt.scatter(data[:,2], data[:,3], c=data[:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_{}_wholeData.eps'.format(folder_name), format='eps')
plt.show()

# %% Action space definition

# determine ganularity of the action space
action_space = 100
action_range = 360/action_space

actions = np.zeros(1)
actions_slots = (action_range/2)*np.ones(1)

for i in range(action_space):
    step = action_range
    actions = np.append(actions, actions[i]+step)
    actions_slots = np.append(actions_slots, actions_slots[i]+step)

for i in range(len(data_manip)):
    index = np.amin(np.where(data_manip[i,4]<actions_slots))
    data_manip[i,4] = actions[index]

# %% Divide data in trajectories

Expert_trajs = [np.empty((0,5)) for _ in range(int(data_manip[-1,0]))]

for i in range(len(data_manip)):
    temp = np.empty((1,5))
    temp[0,:] = data_manip[i,:]
    Expert_trajs[int(data_manip[i,0])-1] = np.append(Expert_trajs[int(data_manip[i,0])-1][:][:],temp,0)

# %% Compute velocity
for i in range(len(Expert_trajs)):
    vel = np.zeros((len(Expert_trajs[i]),1))
    for j in range(len(Expert_trajs[i])-1):
        vel_x = (Expert_trajs[i][j+1][2]-Expert_trajs[i][j][2])/(Expert_trajs[i][j+1][1]-Expert_trajs[i][j][1])
        vel_y = (Expert_trajs[i][j+1][3]-Expert_trajs[i][j][3])/(Expert_trajs[i][j+1][1]-Expert_trajs[i][j][1])
        vel[j+1,0] = np.sqrt((vel_x)**2+(vel_y)**2)
       
    Expert_trajs[i] = np.append(Expert_trajs[i],vel,1)
    
    
# %% Plot Expert's Data

traj = 0

fig = plt.figure()
plot_traj = plt.scatter(Expert_trajs[traj][:,2], Expert_trajs[traj][:,3], c=Expert_trajs[traj][:,1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_traj, ticks=[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
cbar.ax.set_yticklabels(['t = 0','t = 50', 't = 100', 't = 150', 't = 200', 't = 250', 't = 300', 't = 350', 't = 400', 't = 450', 't = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_{}_traj{}.eps'.format(folder_name,traj), format='eps')
plt.show()




 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

   