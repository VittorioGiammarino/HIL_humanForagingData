#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:19:08 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
from mpl_toolkits.mplot3d import Axes3D

# %% Preprocessing_data with psi based on the coins clusters distribution  

def Show_DataSet(Folders, size_data, Rand_traj, action_space, coins, plot='plot'):
# =============================================================================
#     action_space = 'complete', which considers an heading direction every 45deg
#                    'simplified', which considers an action every 
#     coins = 'distr_only' coins on the gaussians distributions
#             'full_coins' all the coins in the original experiment
# =============================================================================
    TrainingSet = np.empty((0,4))
    Labels = np.empty((0,1))
    Time = []
    Trajectories = []
    Rotation = []
    Reward = []
    Real_reward = []
    for folder in Folders:
        for experiment in range(1,11):
            Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, coin_direction_single_traj, reward_single_traj, real_reward = World.Foraging.ProcessData(folder, experiment, action_space, coins)
            Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1), coin_direction_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
            TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
            Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
            Time.append(Time_single_traj)
            Trajectories.append(Training_set_single_traj_together)
            Rotation.append(Labels_single_traj.reshape(len(Labels_single_traj),1))
            Reward.append(reward_single_traj)
            Real_reward.append(real_reward)

    if plot == 'plot':
        coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins') #np.random.randint(0,len(Time))
        True_traj, True_time = World.Foraging.TrueData(Folders[0], Rand_traj+1)
        
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
        plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Time[Rand_traj][0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
        cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Processed traj, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_Time_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
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
        ax.add_artist(circle4)
        plot_data = plt.scatter(0.1*True_traj[0:size_data,0], 0.1*True_traj[0:size_data,1], c=True_time[0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
        cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('True traj, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/ExpertTRUE_Traj_VS_Time_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()  
        
        sigma1 = 0.5
        circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
        sigma2 = 1.1
        circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
        sigma3 = 1.8
        circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
        sigma4 = 1.3
        circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
        fig3, ax3 = plt.subplots()
        plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Trajectories[Rand_traj][0:size_data,2], marker='o', cmap='bwr')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
        cbar = fig3.colorbar(plot_data, ticks=[0, 1])
        cbar.ax.set_yticklabels(['no coins', 'see coins'])
        ax3.add_artist(circle1)
        ax3.add_artist(circle2)
        ax3.add_artist(circle3)
        ax3.add_artist(circle4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Options, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()  
        
        
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        
        for c, m, z in [('blue', 'o', 0.0), ('red', 'o', 1.0)]:
            psi = np.where(z==Trajectories[Rand_traj][0:size_data,2])[0]
            ax3d.scatter(Trajectories[Rand_traj][psi,0], Trajectories[Rand_traj][psi,1], Trajectories[Rand_traj][psi,2], c=c, marker=m)
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk') 
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('psi')
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View3D_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()
    
    return TrainingSet, Labels, Trajectories, Rotation, Time, Reward, Real_reward

def Show_Training(Folders, size_data, Rand_traj, action_space, coins, plot='plot'):
# =============================================================================
#     action_space = 'complete', which considers an heading direction every 45deg
#                    'simplified', which considers an action every 
#     coins = 'distr_only' coins on the gaussians distributions
#             'full_coins' all the coins in the original experiment
# =============================================================================
    TrainingSet = np.empty((0,4))
    Labels = np.empty((0,1))
    Time = []
    Trajectories = []
    Rotation = []
    Reward = []
    Real_reward = []
    for folder in Folders:
        for experiment in range(1,11):
            Training_set_single_traj, Labels_single_traj, Time_single_traj, psi_single_traj, coin_direction_single_traj, reward_single_traj, real_reward = World.Foraging_Training.ProcessData(folder, experiment, action_space, coins)
            Training_set_single_traj_together = np.concatenate((0.1*Training_set_single_traj[0:-1,:], psi_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1), coin_direction_single_traj[0:-1].reshape(len(psi_single_traj[0:-1]),1)),1)
            TrainingSet = np.append(TrainingSet, Training_set_single_traj_together, 0)
            Labels = np.append(Labels, Labels_single_traj.reshape(len(Labels_single_traj),1), 0)
            Time.append(Time_single_traj)
            Trajectories.append(Training_set_single_traj_together)
            Rotation.append(Labels_single_traj.reshape(len(Labels_single_traj),1))
            Reward.append(reward_single_traj)
            Real_reward.append(real_reward)

    if plot == 'plot':
        coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins') #np.random.randint(0,len(Time))
        True_traj, True_time = World.Foraging.TrueData(Folders[0], Rand_traj+1)
        
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
        plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Time[Rand_traj][0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
        cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Processed traj, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_Time_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
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
        ax.add_artist(circle4)
        plot_data = plt.scatter(0.1*True_traj[0:size_data,0], 0.1*True_traj[0:size_data,1], c=True_time[0:size_data], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
        cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
        cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('True traj, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/ExpertTRUE_Traj_VS_Time_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()  
        
        sigma1 = 0.5
        circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
        sigma2 = 1.1
        circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
        sigma3 = 1.8
        circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
        sigma4 = 1.3
        circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
        fig3, ax3 = plt.subplots()
        plot_data = plt.scatter(Trajectories[Rand_traj][0:size_data,0], Trajectories[Rand_traj][0:size_data,1], c=Trajectories[Rand_traj][0:size_data,2], marker='o', cmap='bwr')
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
        cbar = fig3.colorbar(plot_data, ticks=[0, 1])
        cbar.ax.set_yticklabels(['no coins', 'see coins'])
        ax3.add_artist(circle1)
        ax3.add_artist(circle2)
        ax3.add_artist(circle3)
        ax3.add_artist(circle4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Options, action space {}, coins {}'.format(action_space, coins))
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()  
        
        
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        
        for c, m, z in [('blue', 'o', 0.0), ('red', 'o', 1.0)]:
            psi = np.where(z==Trajectories[Rand_traj][0:size_data,2])[0]
            ax3d.scatter(Trajectories[Rand_traj][psi,0], Trajectories[Rand_traj][psi,1], Trajectories[Rand_traj][psi,2], c=c, marker=m)
        plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk') 
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('psi')
        plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View3D_traj{}_{}_{}.eps'.format(Rand_traj, action_space, coins), format='eps')
        plt.show()
    
    return TrainingSet, Labels, Trajectories, Rotation, Time, Reward, Real_reward
