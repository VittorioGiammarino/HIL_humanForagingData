#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:57:58 2021

@author: vittorio
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import csv

class Foraging:    
    def GeneratePsi(Simulated_states, coin_location):      
        reward = 0
        see_coin_array = np.empty((0))
        coin_direction_array = np.empty((0))
        for i in range(len(Simulated_states)):
            see_coin = 0
            dist_from_coins = np.linalg.norm(coin_location-Simulated_states[i,:],2,1)
            l=0
            if np.min(dist_from_coins)<=8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-Simulated_states[i,0], closer_coin_position[1]-Simulated_states[i,1]])
                angle = np.arctan2(closer_coin_relative_position[1], closer_coin_relative_position[0])
                coin_direction = angle
            else:
                coin_direction = 0                
                
            for k in range(len(dist_from_coins)):
                if np.min(dist_from_coins)>8:
                    break
                if dist_from_coins[k]<=8:
                    see_coin = 1
                if dist_from_coins[k]<=2:
                    coin_location = np.delete(coin_location, l, 0)
                    reward = reward+1
                else:
                    l=l+1
                    
            see_coin_array = np.append(see_coin_array, see_coin)                 
            coin_direction_array = np.append(coin_direction_array, coin_direction)
                       
        return see_coin_array, coin_direction_array, reward
    
    def CoinLocation(Folder, experiment, version = 'distr_only'):
        N_coins = 325
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_coin_location.txt".format(Folder,experiment)) as f:
            coin_location_raw = f.readlines()
            
        if len(coin_location_raw) > N_coins:
            start_counting = len(coin_location_raw) - N_coins
            coin_location_raw = coin_location_raw[start_counting:]

        for i in range(len(coin_location_raw)):
            row = coin_location_raw[i][7:-2]
            row = row.replace('[', ',')
            coin_location_raw[i] = row
            
        coin_location_data = csv.reader(coin_location_raw)
        coin_location = np.empty((0,2))
        for row in coin_location_data:
            if len(row)==3:
                coin_location = np.append(coin_location, np.array([[float(row[0]), float(row[2])]]),0)
            else:
                coin_location = np.append(coin_location, np.array([[float(row[1]), float(row[3])]]),0) 
        
        if version == 'distr_only': 
            bool_distribution = np.empty((4))
            j=0
            for i in range(len(coin_location)):
                bool_distribution[0] = (coin_location[j,0]-60)**2 + (coin_location[j,1]-75)**2 <= (2*5)**2
                bool_distribution[1] = (coin_location[j,0]+15)**2 + (coin_location[j,1]+50)**2 <= (2*11)**2
                bool_distribution[2] = (coin_location[j,0]+50)**2 + (coin_location[j,1]-30)**2 <= (2*18)**2
                bool_distribution[3] = (coin_location[j,0]-49)**2 + (coin_location[j,1]+40)**2 <= (2*13)**2
                
                if np.sum(bool_distribution)==0:
                    coin_location = np.delete(coin_location, j, 0) 
                else:
                    j = j+1                
                
        return coin_location
    
    def Get_action(x, x_plus_1):
        
        relative_position = np.array([x_plus_1[0]-x[0], x_plus_1[1]-x[1]])
        angle = np.arctan2(relative_position[1], relative_position[0])
        step = np.linalg.norm(relative_position,2,0)
        
        return angle, step
        
    def ProcessData(Folder, experiment, coins = 'full_coins'):
        coin_location = Foraging.CoinLocation(Folder, experiment, coins)
        
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_position.txt".format(Folder,experiment)) as f:
            data_raw = f.readlines()
    
        for i in range(len(data_raw)):
            row = data_raw[i][1:]
            row = row.replace(']', ',')
            data_raw[i] = row

        agent_data = csv.reader(data_raw)
        Training_set = np.empty((0,2))
        True_set = np.empty((0,2))
        time = np.empty((0,1))

        for row in agent_data:
            True_set = np.append(True_set, np.array([[float(row[0]), float(row[2])]]), 0)
            Training_set = np.append(Training_set, np.array([[np.round(float(row[0]),2), np.round(float(row[2]),2)]]), 0)
            time = np.append(time, float(row[3]))    
            
        State_space, Training_set_index = np.unique(Training_set, return_index=True, axis = 0)    
        Training_set_cleaned = Training_set[np.sort(Training_set_index),:]
        time_cleaned = time[np.sort(Training_set_index)]
        True_Training_set = True_set
        
        Labels = np.empty((0,2))
        for i in range(len(Training_set_cleaned)-1):
            direction, step_length = Foraging.Get_action(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:]) 
            Labels = np.append(Labels, np.array([[direction, 0.1*step_length]]), 0)
            
        see_coin_array, coin_direction_array, reward = Foraging.GeneratePsi(Training_set_cleaned, coin_location)
        
        with open("4_walls_coins_task/FONC_{}_DeID/fMRI/runNumber{}_master.txt".format(Folder, experiment)) as f:
            data_raw = f.readlines()
            
        Real_reward = len(data_raw)
              
        return True_Training_set, time, 0.1*Training_set_cleaned, Labels, time_cleaned, see_coin_array, coin_direction_array, reward, Real_reward
            
    
    class env:
        def __init__(self,  Folder, expert_traj, init_state = np.array([0,0,0,0]), version_coins = 'full_coins'):
            self.state = init_state
            self.Folder = Folder
            self.expert_traj = expert_traj
            self.version_coins = version_coins
            self.coin_location = 0.1*Foraging.CoinLocation(Folder, expert_traj+1, self.version_coins)
            self.observation_space = np.array([len(self.state)])
              
        class action_space:
            def high():
                return np.array([np.pi])
            def low():
                return np.array([-np.pi])
            def shape():
                return (1,)
            
        def reset(self, version = 'standard', init_state = np.array([0,0,0,0])):
            if version == 'standard':
                self.state = init_state
                self.coin_location = 0.1*Foraging.CoinLocation(self.Folder, self.expert_traj+1, self.version_coins)
            else:
                state = 0.1*np.random.randint(-100,100,2)
                init_state = np.concatenate((state, np.array([0,0])))
                self.state = init_state
                self.coin_location = 0.1*Foraging.CoinLocation(self.Folder, self.expert_traj+1, self.version_coins)
                
            return self.state
                
        def seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
    
        def step(self, action):
            
            r=0
            state_partial = self.state[0:2]
            # given action, draw next state
            angle = action[0]
            step = 0.017
            state_plus1_partial = np.zeros((2,))
            state_plus1_partial[0] = state_partial[0] + step*np.cos(angle)
            state_plus1_partial[1] = state_partial[1] + step*np.sin(angle)
                
            if state_plus1_partial[0]>10 or state_plus1_partial[0]<-10:
                state_plus1_partial[0] = state_partial[0] 

            if state_plus1_partial[1]>10 or state_plus1_partial[1]<-10:
                state_plus1_partial[1] = state_partial[1]                 
                    
            # Update psi and reward and closest coin direction
            dist_from_coins = np.linalg.norm(self.coin_location-state_plus1_partial,2,1)
            l=0
            psi = 0
                
            if np.min(dist_from_coins)<=0.8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = self.coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-state_plus1_partial[0],closer_coin_position[1]-state_plus1_partial[1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])
                coin_direction = angle
            else:
                coin_direction = 0  
            
            for p in range(len(dist_from_coins)):
                if np.min(dist_from_coins)>0.8:
                    break
                if dist_from_coins[p]<=0.8:
                    psi = 1
                if dist_from_coins[p]<=0.2:
                    self.coin_location = np.delete(self.coin_location, l, 0)
                    r = r+1
                else:
                    l=l+1
                    
            state_plus1 = np.concatenate((state_plus1_partial, [psi], [coin_direction]))
            self.state = state_plus1
            
            return state_plus1, r, False, False