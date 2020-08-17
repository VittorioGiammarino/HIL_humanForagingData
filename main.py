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
f = open('WILD_P03_crow_2/moving_actor.csv')
moving_actor = csv.reader(f)

data = np.empty((0,5))

for row in moving_actor:
    try:
        temp = np.empty((1,5))
        temp[0,:] = [row[0],row[1],row[3],row[4],row[6]]
        data = np.append(data, temp, 0)
    except:
        print('the data type is not a float')

# %% Divide data in trajectories

Expert_trajs = [np.empty((0,5)) for _ in range(int(data[-1,0]))]

for i in range(len(data)):
    temp = np.empty((1,5))
    temp[0,:] = data[i,:]
    Expert_trajs[int(data[i,0])-1] = np.append(Expert_trajs[int(data[i,0])-1][:][:],temp,0)

# %% Compute velocity
for i in range(len(Expert_trajs)):
    vel = np.zeros((len(Expert_trajs[i]),1))
    for j in range(len(Expert_trajs[i])-1):
        vel_x = (Expert_trajs[i][j+1][2]-Expert_trajs[i][j][2])/(Expert_trajs[i][j+1][1]-Expert_trajs[i][j][1])
        vel_y = (Expert_trajs[i][j+1][3]-Expert_trajs[i][j][3])/(Expert_trajs[i][j+1][1]-Expert_trajs[i][j][1])
        vel[j+1,0] = np.sqrt((vel_x)**2+(vel_y)**2)
       
    Expert_trajs[i] = np.append(Expert_trajs[i],vel,1)
    
    


# %% Plot Expert's Data

fig = plt.figure()
plt.scatter(Expert_trajs[0][:,2], Expert_trajs[0][:,3], c=Expert_trajs[0][:,1], marker='o', cmap='winter')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Figures/FiguresExpert/Expert_state_action_distribution.eps', format='eps')
plt.show()


   