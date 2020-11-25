#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:13:05 2020

@author: vittorio
"""

import numpy as np
import csv

class Foraging:
    def TransitionCheck4Labels(state,state_next):
        Transition = np.zeros((8,2))
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
        
        return index
    
    def StateTransition(original_actions, original_data):
        sim = np.empty((0,2))
        init = original_data[0,:]
        sim = np.append(sim, init.reshape(1,2), 0)
        for i in range(len(original_actions)):
            index = original_actions[i]
            Transition = np.zeros((9,2))
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
            Transition[8,:] = original_data[i,:]
            next_state = Transition[int(index),:]
            sim = np.append(sim,next_state.reshape(1,2),0)
            
        return sim
    
    
    def ProcessData(Folder, experiment):
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_position.txt".format(Folder,experiment)) as f:
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

        Labels = np.empty((0,1))
        for i in range(len(Training_set_cleaned)-1):
            index = Foraging.TransitionCheck4Labels(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
            Labels = np.append(Labels, index)

        # % Simulate dynamics
        Simulated_states = Foraging.StateTransition(Labels, Training_set_cleaned)
        
        return Simulated_states, Labels, time_cleaned