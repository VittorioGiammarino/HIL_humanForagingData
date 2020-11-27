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
    
class Simulation_NN:
    def __init__(self, pi_hi, pi_lo, pi_b):
        option_space = len(pi_lo)
        self.option_space = option_space
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        self.zeta = 0.0001
        self.pi_hi = pi_hi
        self.pi_lo = pi_lo
        self.pi_b = pi_b      

    def Transition(state,action):
        Transition = np.zeros((9,2))
        Transition[0,0] = state[0,0] + 0.1
        Transition[0,1] = state[0,1] + 0
        Transition[1,0] = state[0,0] + 0.1
        Transition[1,1] = state[0,1] + 0.1
        Transition[2,0] = state[0,0] + 0
        Transition[2,1] = state[0,1] + 0.1
        Transition[3,0] = state[0,0] - 0.1
        Transition[3,1] = state[0,1] + 0.1
        Transition[4,0] = state[0,0] - 0.1
        Transition[4,1] = state[0,1] + 0
        Transition[5,0] = state[0,0] - 0.1
        Transition[5,1] = state[0,1] - 0.1
        Transition[6,0] = state[0,0] + 0
        Transition[6,1] = state[0,1] - 0.1
        Transition[7,0] = state[0,0] + 0.1
        Transition[7,1] = state[0,1] - 0.1
        Transition[8,:] = state
        state_plus1 = Transition[int(action),:]
        
        return state_plus1          
                
    def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories, initial_state):
        traj = [[None]*1 for _ in range(number_of_trajectories)]
        control = [[None]*1 for _ in range(number_of_trajectories)]
        Option = [[None]*1 for _ in range(number_of_trajectories)]
        Termination = [[None]*1 for _ in range(number_of_trajectories)]
    
        for t in range(0,number_of_trajectories):       
            x = np.empty((0,2))
            x = np.append(x, initial_state.reshape(1,2), 0)
            u_tot = np.empty((0,0))
            o_tot = np.empty((0,0),int)
            b_tot = np.empty((0,0),int)
        
            # Initial Option
            prob_o = self.mu
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[0]):
                prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled))
            o_tot = np.append(o_tot,o)
        
            # Termination
            state = x[0,:]
            state = state.reshape(1,2)
            prob_b = self.pi_b[o](state).numpy()
            prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
            for i in range(1,prob_b_rescaled.shape[1]):
                prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
            draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
            b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
            b_tot = np.append(b_tot,b)
            if b == 1:
                b_bool = True
            else:
                b_bool = False
        
            o_prob_tilde = np.empty((1,self.option_space))
            if b_bool == True:
                o_prob_tilde = self.pi_hi(state).numpy()
            else:
                o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
        
            for k in range(0,max_epoch_per_traj):
                state = x[k,:]
                state = state.reshape((1,2))
                # draw action
                prob_u = self.pi_lo[o](state).numpy()
                prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                for i in range(1,prob_u_rescaled.shape[1]):
                    prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
                # given action, draw next state
                state_plus1 = Simulation_NN.Transition(state, u)
                state_plus1 = state_plus1.reshape(1,2)
                x = np.append(x, state_plus1, 0)
                u_tot = np.append(u_tot,u)
                        
                # Select Termination
                # Termination
                prob_b = self.pi_b[o](state_plus1).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state_plus1).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
            
        
            traj[t] = x
            control[t]=u_tot
            Option[t]=o_tot
            Termination[t]=b_tot

            return traj, control, Option, Termination
            