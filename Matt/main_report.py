#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:26:21 2021

@author: vittorio
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
from tensorflow import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
import BatchBW_HIL
from utils_main import Show_DataSet
from utils_main import Show_Training
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
import multiprocessing.pool

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
Folders = [6] #[6, 7, 11, 12, 15]
size_data = 3100
Rand_traj = 2

TrainingSet, Labels, Trajectories, Rotation, Time, _ = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'distr_only', 'plot')
# _,_,_,_,_,_ = Show_DataSet(Folders, size_data, Rand_traj, 'simplified', 'distr_only')
_,_,_,_,_, Reward_eval_human  = Show_DataSet(Folders, size_data, Rand_traj, 'complete', 'full_coins', 'plot')
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
plt.savefig('Figures/FiguresExpert/Human_Reward.eps', format='eps')
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
plt.savefig('Figures/FiguresExpert/Human_Reward_training.eps', format='eps')
plt.show() 

coins_location = World.Foraging.CoinLocation(Folders[0], Rand_traj+1, 'full_coins')

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
plt.title('Best Human traj, Reward {}'.format(Reward_eval_human[Rand_traj]))
plt.savefig('Figures/FiguresExpert/Best_human_traj.eps', format='eps')
plt.show()  

# %%
coins_location = World.Foraging.CoinLocation(Folders[0], Rand_traj+1, 'full_coins')

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
plt.savefig('Figures/FiguresExpert/Coins_Only.eps', format='eps')
plt.show()  

# %% 
Folders = [6]
with open('RL_algorithms/DeepQ_Learning/Results/Q_learning_evaluation_results__deeper.npy', 'rb') as f:
    DQN_Evaluation = np.load(f, allow_pickle=True).tolist()

averageDQN = []

for i in range(len(DQN_Evaluation)):
    averageDQN.append(np.mean(DQN_Evaluation[i][0]))
best_index_agent = np.argmax(averageDQN)

best_reward_index=np.argmax(DQN_Evaluation[best_index_agent][0])
best_episode=DQN_Evaluation[best_index_agent][1][best_reward_index]

episodes = np.arange(0,len(DQN_Evaluation[best_index_agent][0]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(DQN_Evaluation[best_index_agent][0]),'k--')
plt.plot(episodes, DQN_Evaluation[best_index_agent][0],'g', label = 'DQN agent (mean = {})'.format(np.mean(DQN_Evaluation[best_index_agent][0])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evaluation DQN')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/FiguresDQN/DQN_evaluation.eps', format='eps')
plt.show() 

coins_location = World.Foraging.CoinLocation(Folders[0], Rand_traj+1, 'full_coins') #np.random.randint(0,len(Time))

time = np.linspace(0,480,3001)  
sigma1 = 0.5
circle1 = ptch.Circle((6.0, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(best_episode[:,0], best_episode[:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best DQN Traj, reward {}'.format(DQN_Evaluation[best_index_agent][0][best_reward_index]))
plt.savefig('Figures/FiguresDQN/DQN_Traj_example.eps', format='eps')
plt.show() 

# %%
picked_agent = 0

with open('RL_algorithms/DeepQ_Learning/Results/Q_learning_results_deeper.npy', 'rb') as f:
    Mixture_of_DQN = np.load(f, allow_pickle=True).tolist()

N_agents = 10

best_reward = np.zeros(N_agents)
best_episode = [[None]*1 for _ in range(N_agents)]
best_agent = np.zeros(N_agents)
best_net = [[None]*1 for _ in range(N_agents)]

for i in range(len(Mixture_of_DQN)):
    for j in range(len(Mixture_of_DQN[i][0])):
        temp = Mixture_of_DQN[i][0][j]
        k = np.argmin(best_reward)
        if temp>best_reward[k]:
            best_reward[k]=Mixture_of_DQN[i][0][j]
            best_episode[k]=Mixture_of_DQN[i][1][j]
            best_agent[k]=i
            best_net[k]=Mixture_of_DQN[i][2][j]

picked_agent = np.argmax(best_reward)

episodes = np.arange(0,len(Mixture_of_DQN[0][0]))
z = np.polyfit(episodes, Mixture_of_DQN[int(best_agent[picked_agent])][0], 3)
p = np.poly1d(z)
plt.plot(episodes,p(episodes),'k--', label='training trend')
plt.plot(episodes, Mixture_of_DQN[int(best_agent[picked_agent])][0],'g', label = 'DQN agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training DQN')
plt.legend()
plt.savefig('Figures/FiguresDQN/DQN_training_trend.eps', format='eps')
plt.show() 

#%% Learning from a human Expert Behavioral Cloning
    
with open('Results_main/BC_from_human_evaluation_results.npy', 'rb') as f:
    BC_from_human_evaluation_results = np.load(f, allow_pickle=True).tolist()

averageBC = []

for i in range(len(BC_from_human_evaluation_results)):
    averageBC.append(np.mean(BC_from_human_evaluation_results[i][0]))
best_index_agentBC = np.argmax(averageBC)
best_reward_indexBC=np.argmax(BC_from_human_evaluation_results[best_index_agentBC][0])
best_episodeBC=BC_from_human_evaluation_results[best_index_agentBC][1][best_reward_indexBC]


time = np.linspace(0,480,3001)  
sigma1 = 0.5
circle1 = ptch.Circle((6.0, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(best_episodeBC[:,0], best_episodeBC[:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best BC Traj, reward {}'.format(best_reward[0]))
plt.savefig('Figures/FiguresBC/BC_from_human_Evaluation.eps', format='eps')
plt.show() 

episodes = np.arange(0,len(BC_from_human_evaluation_results[best_index_agentBC][0]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(BC_from_human_evaluation_results[best_index_agentBC][0]),'k--')
plt.plot(episodes, BC_from_human_evaluation_results[best_index_agentBC][0], 'g', label = 'BC (mean = {})'.format(np.mean(BC_from_human_evaluation_results[best_index_agentBC][0])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evaluation BC')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/FiguresBC/BC_evaluation_trend.eps', format='eps')
plt.show() 

# %% HIL-preinitialized from human
    
with open('Results_main/HIL_from_human_evaluation_results.npy', 'rb') as f:
    HIL_from_human_evaluation_results = np.load(f, allow_pickle=True).tolist()

averageHIL = []

for i in range(len(HIL_from_human_evaluation_results)):
    averageHIL.append(np.mean(HIL_from_human_evaluation_results[i][6]))
best_index_agentHIL = np.argmax(averageHIL)
best_reward_indexHIL=np.argmax(HIL_from_human_evaluation_results[best_index_agentHIL][6])
best_episodeHIL=HIL_from_human_evaluation_results[best_index_agentHIL][0][best_reward_indexHIL]

coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')

# Plot result
sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(best_episodeHIL[:,0], best_episodeHIL[:,1], c=HIL_from_human_evaluation_results[best_index_agentHIL][2][best_reward_indexHIL][1:], marker='o', cmap='bwr')
#plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['option 1', 'option 2'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
plt.xlim([-10, 10])
plt.ylim([-10, 11])
plt.xlabel('x')
plt.ylabel('y')
plt.title('HIL agent, reward {}'.format(HIL_from_human_evaluation_results[best_index_agentHIL][6][best_reward_indexHIL]))
plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Options_traj_reward{}.eps'.format(HIL_from_human_evaluation_results[best_index_agentHIL][6][best_reward_indexHIL]), format='eps')
plt.show()  

sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(best_episodeHIL[:,0], best_episodeHIL[:,1], c=Time[0][0:len(best_episode[0])], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
plt.xlim([-10, 10])
plt.ylim([-10, 11])
plt.xlabel('x')
plt.ylabel('y')
plt.title('HIL agent, reward {}'.format(HIL_from_human_evaluation_results[best_index_agentHIL][6][best_reward_indexHIL]))
plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Time_traj_reward{}.eps'.format(HIL_from_human_evaluation_results[best_index_agentHIL][6][best_reward_indexHIL]), format='eps')
plt.show()  

episodes = np.arange(0,len(HIL_from_human_evaluation_results[best_index_agentHIL][6]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(HIL_from_human_evaluation_results[best_index_agentHIL][6]),'k--')
plt.plot(episodes, HIL_from_human_evaluation_results[best_index_agentHIL][6],'g', label = 'HIL (mean = {})'.format(np.mean(HIL_from_human_evaluation_results[best_index_agentHIL][6])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evaluation HIL')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/FiguresBatch/HIL_evaluation_trend.eps', format='eps')
plt.show() 


# %% HIL-random initialization from human

with open('Results_main/HIL_from_human_evaluation_results_random_init.npy', 'rb') as f:
    HIL_from_human_evaluation_results_random_init = np.load(f, allow_pickle=True).tolist()

averageHIL_random_init = []

for i in range(len(HIL_from_human_evaluation_results_random_init)):
    averageHIL_random_init.append(np.mean(HIL_from_human_evaluation_results_random_init[i][6]))
best_index_agentHIL_random_init = np.argmax(averageHIL_random_init)
best_reward_indexHIL_random_init=np.argmax(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6])
best_episodeHIL_random_init=HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][0][best_reward_indexHIL_random_init]


coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')
best_index = np.argmax(best_reward)

# Plot result
sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(best_episodeHIL_random_init[:,0], best_episodeHIL_random_init[:,1], c=HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][2][best_reward_indexHIL_random_init][1:], marker='o', cmap='bwr')
#plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['option 1', 'option 2'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
plt.xlim([-10, 10])
plt.ylim([-10, 11])
plt.xlabel('x')
plt.ylabel('y')
plt.title('HIL agent random init, reward {}'.format(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6][best_reward_indexHIL_random_init]))
plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Options_traj_new_random_init_reward{}.eps'.format(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6][best_reward_indexHIL_random_init]), format='eps')
plt.show()  

sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(best_episodeHIL_random_init[:,0], best_episodeHIL_random_init[:,1], c=time, marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
plt.xlim([-10, 10])
plt.ylim([-10, 11])
plt.xlabel('x')
plt.ylabel('y')
plt.title('HIL agent random init, reward {}'.format(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6][best_reward_indexHIL_random_init]))
plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Time_traj_new_random_init_reward{}.eps'.format(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6][best_reward_indexHIL_random_init]), format='eps')
plt.show()  

episodes = np.arange(0,len(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6]),'k--')
plt.plot(episodes, HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6],'g', label = 'HIL random init (mean {})'.format(np.mean(HIL_from_human_evaluation_results_random_init[best_index_agentHIL_random_init][6])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evaluation HIL random init')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/FiguresBatch/HIL_evaluation_trend_random_init.eps', format='eps')
plt.show() 


# %% Training Option Critic starting from HIL preinit

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_results_second_attempt.npy', 'rb') as f:
    DeepSoftOC_learning_results = np.load(f, allow_pickle=True).tolist()
    
N_agents = 10

best_reward = np.zeros(N_agents)
best_traj = [[None]*1 for _ in range(N_agents)]
best_option = [[None]*1 for _ in range(N_agents)]
best_temrination = [[None]*1 for _ in range(N_agents)]
best_agent = np.zeros(N_agents)
best_pi_hi = [[None]*1 for _ in range(N_agents)]
best_pi_lo = [[None]*1 for _ in range(N_agents)]
best_pi_b = [[None]*1 for _ in range(N_agents)]
seeds = [31] #[13, 29, 33, 36, 27, 21, 15, 31]

for i in range(len(DeepSoftOC_learning_results)):
    for j in range(len(DeepSoftOC_learning_results[i][0])):
        temp = DeepSoftOC_learning_results[i][0][j]
        k = np.argmin(best_reward)
        if temp>best_reward[k]:
            best_reward[k]=DeepSoftOC_learning_results[i][0][j]
            best_traj[k]=DeepSoftOC_learning_results[i][1][j]
            best_option[k]=DeepSoftOC_learning_results[i][2][j]
            best_temrination[k]=DeepSoftOC_learning_results[i][3][j]
            best_pi_hi[k] = DeepSoftOC_learning_results[i][4][j]
            best_pi_lo[k] = DeepSoftOC_learning_results[i][5][j]
            best_pi_b[k] = DeepSoftOC_learning_results[i][6][j]
            best_agent[k]=j

coins_location = World.Foraging.CoinLocation(6, 2+1, 'full_coins') #np.random.randint(0,len(Time))

best_index = np.argmax(best_reward)
n_episode = best_index
time = np.linspace(0,480,3001)  
 
sigma1 = 0.5
circle1 = ptch.Circle((6.0, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
plot_data = plt.scatter(best_traj[n_episode][:,0], best_traj[n_episode][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traj, OC n{}, reward {}'.format(best_agent[n_episode], best_reward[n_episode]))
plt.savefig('Figures/FiguresOC/OC_Traj{}_reward{}.eps'.format(best_agent[n_episode], best_reward[n_episode]), format='eps')
plt.show()      

sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(best_traj[n_episode][:,0], best_traj[n_episode][:,1], c=best_option[n_episode][:], marker='o', cmap='bwr')
#plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['option 1', 'option 2'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title('OC agent, reward {}'.format(best_reward[n_episode]))
plt.savefig('Figures/FiguresOC/OC_Traj_VS_Options_traj_reward{}.eps'.format(best_reward[n_episode]), format='eps')
plt.show()  

episodes = np.arange(0,len(DeepSoftOC_learning_results[0][0]))
z = np.polyfit(episodes, DeepSoftOC_learning_results[0][0],4)
p = np.poly1d(z)
plt.plot(episodes, p(episodes),'k--', label='Evaluation Trend')
plt.plot(episodes, DeepSoftOC_learning_results[0][0], 'g', label = 'OC agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training OC')
plt.legend()
plt.savefig('Figures/FiguresOC/OC_training_trend.eps', format='eps')
plt.show() 

# %% Evaluating Option Critic

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_evaluation.npy', 'rb') as f:
    DeepSoftOC_learning_evaluation = np.load(f, allow_pickle=True).tolist()
 
average_reward = []
for i in range(len(DeepSoftOC_learning_evaluation)):
    average_reward.append(np.mean(DeepSoftOC_learning_evaluation[i][3]))
best_index = np.argmax(average_reward)
best_traj_index = np.argmax(DeepSoftOC_learning_evaluation[best_index][3])


coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')

# Plot result
sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(DeepSoftOC_learning_evaluation[best_index][0][best_traj_index][:,0], DeepSoftOC_learning_evaluation[best_index][0][best_traj_index][:,1], c=DeepSoftOC_learning_evaluation[best_index][1][best_traj_index][:], marker='o', cmap='bwr')
#plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['option 1', 'option 2'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title('OC agent, reward {}'.format(DeepSoftOC_learning_evaluation[best_index][3][best_traj_index]))
plt.savefig('Figures/FiguresOC/OC_Traj_VS_Options_traj_reward{}.eps'.format(DeepSoftOC_learning_evaluation[best_index][3][best_traj_index]), format='eps')
plt.show()  

# Plot result
sigma1 = 0.5
circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
sigma2 = 1.1
circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
sigma3 = 1.8
circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
sigma4 = 1.3
circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
fig, ax2 = plt.subplots()
plot_data = plt.scatter(DeepSoftOC_learning_evaluation[best_index][0][best_traj_index][:,0], DeepSoftOC_learning_evaluation[best_index][0][best_traj_index][:,1], c=time, marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
ax2.add_artist(circle1)
ax2.add_artist(circle2)
ax2.add_artist(circle3)
ax2.add_artist(circle4)
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title('OC agent, reward {}'.format(DeepSoftOC_learning_evaluation[best_index][3][best_traj_index]))
plt.savefig('Figures/FiguresOC/OC_Traj_VS_Time_traj_reward{}.eps'.format(DeepSoftOC_learning_evaluation[best_index][3][best_traj_index]), format='eps')
plt.show()  


episodes = np.arange(0,len(DeepSoftOC_learning_evaluation[best_index][3]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(DeepSoftOC_learning_evaluation[best_index][3]),'k--')
plt.plot(episodes, DeepSoftOC_learning_evaluation[best_index][3],'g', label = 'OC (mean = {})'.format(np.mean(DeepSoftOC_learning_evaluation[best_index][3])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evaluation OC')
plt.legend()
plt.ylim([0, 300])
plt.savefig('Figures/FiguresOC/OC_evaluation_trend.eps', format='eps')
plt.show() 

# %% Summarizing plot
with open('RL_algorithms/DeepQ_Learning/Results/Q_learning_evaluation_results__deeper.npy', 'rb') as f:
    DQN_Evaluation = np.load(f, allow_pickle=True).tolist()
    
with open('Results_main/HIL_from_human_evaluation_results.npy', 'rb') as f:
    HIL_from_human_evaluation_results = np.load(f, allow_pickle=True).tolist()

averageDQN = []
averageHIL = []

for i in range(len(DQN_Evaluation)):
    averageDQN.append(np.mean(DQN_Evaluation[i][0]))
best_index_agentDQN = np.argmax(averageDQN)
best_reward_indexDQN=np.argmax(DQN_Evaluation[best_index_agentDQN][0])
best_episodeDQN=DQN_Evaluation[best_index_agent][1][best_reward_indexDQN]

for i in range(len(HIL_from_human_evaluation_results)):
    averageHIL.append(np.mean(HIL_from_human_evaluation_results[i][6]))
best_index_agentHIL = np.argmax(averageHIL)
best_reward_indexHIL=np.argmax(HIL_from_human_evaluation_results[best_index_agentHIL][6])
best_episodeHIL=HIL_from_human_evaluation_results[best_index_agentHIL][0][best_reward_indexDQN]    


episodes = np.arange(0,len(DeepSoftOC_learning_evaluation[best_index][3]))
plt.plot(episodes,np.ones(len(episodes))*np.mean(DeepSoftOC_learning_evaluation[best_index][3]),'k--')
plt.plot(episodes, DeepSoftOC_learning_evaluation[best_index][3],'g', label = 'OC+HIL (mean = {})'.format(np.mean(DeepSoftOC_learning_evaluation[best_index][3])))
plt.plot(episodes,np.ones(len(episodes))*np.mean(DQN_Evaluation[best_index_agentDQN][0]),'r--')
plt.plot(episodes, DQN_Evaluation[best_index_agentDQN][0],'b', label = 'DQN (mean = {})'.format(np.mean(DQN_Evaluation[best_index_agentDQN][0])))
plt.plot(episodes,np.ones(len(episodes))*np.mean(HIL_from_human_evaluation_results[best_index_agentHIL][6]),'y--')
plt.plot(episodes, HIL_from_human_evaluation_results[best_index_agentHIL][6],'m', label = 'HIL (mean = {})'.format(np.mean(HIL_from_human_evaluation_results[best_index_agentHIL][6])))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('comparison')
plt.legend()
plt.ylim([0, 350])
plt.savefig('Figures/comparison.eps', format='eps')
plt.show() 

episodes = np.arange(0,len(Reward_eval_human))
plt.plot(episodes, np.ones(len(episodes))*np.mean(Reward_eval_human),'k--')
plt.plot(episodes, Reward_eval_human,'g', label = 'Human (mean = {})'.format(np.mean(Reward_eval_human)))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Human agent performance Day 2')
plt.legend()
plt.ylim([0, 350])
plt.savefig('Figures/FiguresExpert/Human_Reward_final_comparison.eps', format='eps')
plt.show() 
