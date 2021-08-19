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
import OnlineBW_HIL
import OnlineBW_HIL_Neural
from utils_main import Show_DataSet
from utils_main import Show_Training
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
import multiprocessing.pool

# %% Preprocessing_data from humans with psi based on the coins clusters distribution  
Folders = [6, 7, 11, 12, 15]

TrainingSet, Labels, Trajectories, Rotation, Time, _, _, _, _ = Show_DataSet(Folders, 'complete', 'distr_only')
_,_,_,_,_, Reward_eval_human, Real_Traj_eval_human, Real_Reward_eval_human, Real_Time_eval_human = Show_DataSet(Folders, 'complete', 'full_coins')
_, _, _, _, _, Reward_training_human, Real_Traj_training_human, Real_Reward_training_human, Real_Time_training_human = Show_Training(Folders, 'complete', 'full_coins')

# %% Plot human expert

threshold = np.mean(Real_Reward_eval_human)
performance_training = np.mean(Real_Reward_training_human)
Rand_traj = 2
# Rand_traj = np.argmax(Real_Reward_eval_human)
size_data = len(Trajectories[Rand_traj])

nTraj = Rand_traj%10
folder = Folders[int(Rand_traj/10)]
coins_location = World.Foraging.CoinLocation(folder, nTraj+1, 'full_coins')

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
plt.title('Human Agent, Reward {}'.format(Reward_eval_human[Rand_traj]))
plt.savefig('Figures/FiguresExpert/Processed_human_traj.eps', format='eps')
plt.show()  

time = np.linspace(0,480,len(Real_Traj_eval_human[Rand_traj][:,0])) 

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
plot_data = plt.scatter(0.1*Real_Traj_eval_human[Rand_traj][:,0], 0.1*Real_Traj_eval_human[Rand_traj][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Human Agent, Reward {}'.format(Real_Reward_eval_human[Rand_traj]))
plt.savefig('Figures/FiguresExpert/Real_human_traj{}.eps'.format(Rand_traj), format='eps')
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
plot_data = plt.scatter(Trajectories[2][:,0], Trajectories[2][:,1], c=Trajectories[2][:,2], marker='o', cmap='bwr')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
cbar = fig3.colorbar(plot_data, ticks=[0, 1])
cbar.ax.set_yticklabels(['no coins', 'see coins'])
ax3.add_artist(circle1)
ax3.add_artist(circle2)
ax3.add_artist(circle3)
ax3.add_artist(circle4)
plt.xlabel('x')
plt.ylabel('y')
# plt.title('Options, action space {}, coins {}'.format(action_space, coins))
plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
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
Rand_traj = 2

with open('RL_algorithms/DeepQ_Learning/Results/Q_learning_evaluation_results__deeper.npy', 'rb') as f:
    DQN_Evaluation = np.load(f, allow_pickle=True).tolist()

averageDQN = []
MaxDQN = []

for i in range(len(DQN_Evaluation)):
    averageDQN.append(np.mean(DQN_Evaluation[i][0]))
    MaxDQN.append(np.max(DQN_Evaluation[i][0]))
best_index_agent = np.argmax(MaxDQN)

best_reward_index=np.argmax(DQN_Evaluation[best_index_agent][0])
best_episode=DQN_Evaluation[best_index_agent][1][best_reward_index]


coins_location = World.Foraging.CoinLocation(Folders[0], Rand_traj+1, 'full_coins') #np.random.randint(0,len(Time))

time = np.linspace(0,480,len(best_episode[:,0]))  
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
plt.title('DQN Agent, reward {}'.format(DQN_Evaluation[best_index_agent][0][best_reward_index]))
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
z = np.polyfit(episodes, Mixture_of_DQN[int(best_agent[picked_agent])][0], 100)
p = np.poly1d(z)
plt.plot(episodes,p(episodes),'g', label = 'DQN agent')
#plt.plot(episodes, Mixture_of_DQN[int(best_agent[picked_agent])][0],'g', label = 'DQN agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training DQN')
plt.legend()
plt.savefig('Figures/FiguresDQN/DQN_training_trend_smoothed.eps', format='eps')
plt.show() 


plt.plot(episodes, Mixture_of_DQN[int(best_agent[picked_agent])][0],'g', label = 'DQN agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training DQN')
plt.legend()
plt.savefig('Figures/FiguresDQN/DQN_training_trend_unsmoothed.eps', format='eps')
plt.show() 

#%% Learning from a human Expert
# %% Behavioral Cloning
size_data = len(Trajectories[Rand_traj])-1
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[Rand_traj][0:size_data]
observation_space_size = T_set.shape[1]
action_size = len(np.unique(Heading_set))

model_BC_from_human = keras.Sequential([             
        keras.layers.Dense(512, activation='relu', input_shape=(observation_space_size,),
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                           bias_initializer=keras.initializers.Zeros()),                                
        keras.layers.Dense(action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2)),
        keras.layers.Softmax()
                         ])              

model_BC_from_human.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

model_BC_from_human.fit(T_set, Heading_set, epochs=200)

# %%

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def EvaluationBC(Folder, Rand_traj, NEpisodes, initial_state, seed, model):

    reward_per_episode =[]
    traj = [[None]*1 for _ in range(NEpisodes)]
    control = [[None]*1 for _ in range(NEpisodes)]
    np.random.seed(seed)
    observation_space_size = 13
    env = World.Foraging.env(Folder, Rand_traj, init_state = initial_state)

    for i_episode in range(NEpisodes):
        x = np.empty((0, observation_space_size))
        u = np.empty((0, 1))
        current_state = env.reset('standard', init_state = initial_state)
        coordinates = current_state[0:2]
        psi = current_state[2]
        psi_encoded = np.zeros(2) #psi dimension = 2
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(9) # coin dir dimension = 9
        coin_dir = current_state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))            
        cum_reward = 0 
        x = np.append(x, current_state_encoded.reshape(1, observation_space_size), 0)
        
        for t in range(3000):
            # draw action
            prob_u = model(current_state_encoded.reshape(1, observation_space_size)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            action = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                               
            obs, reward = env.step(action)
            current_state = obs
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(2)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(9)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))   
            x = np.append(x, current_state_encoded.reshape(1, observation_space_size), 0)
            u = np.append(u, [[action]], 0)
            cum_reward = cum_reward + reward
                                
        print("Episode {}: cumulative reward = {}".format(i_episode, cum_reward))
        reward_per_episode.append(cum_reward)
        traj[i_episode] = x    
        control[i_episode] = u
    
    return  reward_per_episode, traj, control

def evaluateBC_fromHuman(seed, Folder, Rand_traj, NEpisodes, initial_state, weights):
    
    observation_space_size = 13
    action_size = 8
    model = keras.Sequential([             
            keras.layers.Dense(512, activation='relu', input_shape=(observation_space_size,),
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                           bias_initializer=keras.initializers.Zeros()),                                
            keras.layers.Dense(action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2)),
            keras.layers.Softmax()
                         ])   
    model.set_weights(weights)
    reward_per_episode, traj, network_weights = EvaluationBC(Folder, Rand_traj, NEpisodes, initial_state, seed, model)
    return reward_per_episode, traj, network_weights
    
NEpisodes = 100
Nseed=40
initial_state = Trajectories[Rand_traj][0,:]
environment = World.Foraging.env(Folders[0], Rand_traj, init_state = initial_state)

initial_state = np.array([0, -2.6, 0, 8])
Ncpu = Nseed
pool = MyPool(Ncpu)
# args = [(seed, Folders[0], Rand_traj, NEpisodes, initial_state, model_BC_from_human.get_weights()) for seed in range(Nseed)]
# BC_from_human_evaluation_results = pool.starmap(evaluateBC_fromHuman, args) 
# BC_from_human_evaluation_results = evaluate(0, Folders[0], Rand_traj, NEpisodes, initial_state, model_BC_from_human.get_weights())

pool.close()
pool.join()

# %%

# with open('4_walls_coins_task/BC_from_human_evaluation_results.npy', 'wb') as f:
#     np.save(f, BC_from_human_evaluation_results)
    
with open('Results_main/BC_from_human_evaluation_results.npy', 'rb') as f:
    BC_from_human_evaluation_results = np.load(f, allow_pickle=True).tolist()

averageBC = []
MaxBC = []

for i in range(len(BC_from_human_evaluation_results)):
    averageBC.append(np.mean(BC_from_human_evaluation_results[i][0]))
    MaxBC.append(np.max(BC_from_human_evaluation_results[i][0]))
best_index_agentBC = np.argmax(MaxBC)
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
plt.title('BC Agent, reward {}'.format(BC_from_human_evaluation_results[best_index_agentBC][0][best_reward_indexBC]))
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

# %% train_pi_hi
# Need to run but in practice it is unused

T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)

# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)

coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)

T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)


option_space = 2
size_input = T_set.shape[1]
# T_set =TrainingSet
pi_hi = BatchBW_HIL.NN_PI_HI(option_space, size_input)
pi_hi_model = pi_hi.PreTraining(T_set)

options_predictions = pi_hi_model.predict(T_set)


# %% Training single trajectory pre initialized batch
Likelihood_batch_list = []
size_data = len(Trajectories[Rand_traj])-1
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[Rand_traj][0:size_data]
option_space = 2
M_step_epoch = 10
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-1)
# Agent_BatchHIL = BatchBW_HIL.BatchHIL_param_simplified(T_set, Heading_set, M_step_epoch, size_batch, optimizer) 
Agent_BatchHIL = BatchBW_HIL.BatchHIL(T_set, Heading_set, option_space, M_step_epoch, size_batch, optimizer, options_predictions)
N=10 #number of iterations for the BW algorithm
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
Likelihood_batch_list.append(likelihood)

# %evaluation
# coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')
# seed = 0
# BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
# [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(seed, 3000, 1, initial_state[0:2], Folders[0], Rand_traj)

#%%
BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN_preinit_new')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_preinit_new'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_preinit_new'.format(i))
#%
# %% Online BW for HIL
Rand_traj = 2
size_data = len(Trajectories[Rand_traj])-1
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[Rand_traj][0:size_data]
option_space = 2
M_step_epoch = 10
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-1)
# Agent_BatchHIL = BatchBW_HIL.BatchHIL_param_simplified(T_set, Heading_set, M_step_epoch, size_batch, optimizer) 
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(T_set, Heading_set, option_space, M_step_epoch, optimizer)
T_min=100 #number of iterations for the BW algorithm
pi_hi_online, pi_lo_online, pi_b_online =  Agent_OnlineHIL.Online_Baum_Welch_together(T_min)

# %% 
OnlineBW_HIL.NN_PI_HI.save(pi_hi_online, 'Models/Saved_Model_Online/pi_hi_online_NN_preinit')
for i in range(option_space):
    OnlineBW_HIL.NN_PI_LO.save(pi_lo_online[i], 'Models/Saved_Model_Online/pi_lo_NN_{}_online_preinit'.format(i))
    OnlineBW_HIL.NN_PI_B.save(pi_b_online[i], 'Models/Saved_Model_Online/pi_b_NN_{}_online_preinit'.format(i))

# %%

# pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN_preinit_new')
# pi_lo_batch = []
# pi_b_batch = []
# option_space = 2
# for i in range(option_space):
#     pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_preinit_new'.format(i)))
#     pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_preinit_new'.format(i)))

pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN_preinit')
pi_lo_batch = []
pi_b_batch = []
option_space = 2
for i in range(option_space):
    pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_preinit'.format(i)))
    pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_preinit'.format(i)))

def evaluateHIL_fromHuman(seed, Folder, Rand_traj, NEpisodes, initial_state, pi_hi_batch_weights, pi_lo_batch_weights1, pi_lo_batch_weights2, pi_b_batch_weights1, pi_b_batch_weights2):
    
    option_space = 2
    termination_space = 2
    observation_space_size = 13
    action_size = 8
    
    pi_hi = BatchBW_HIL.NN_PI_HI(option_space, observation_space_size)
    pi_lo1 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_lo2 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_b1 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    pi_b2 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    
    pi_hi_batch = pi_hi.NN_model()
    pi_hi_batch.set_weights(pi_hi_batch_weights)
    
    pi_lo_batch = []
    pi_lo1_model = pi_lo1.NN_model()
    pi_lo1_model.set_weights(pi_lo_batch_weights1)
    pi_lo_batch.append(pi_lo1_model)
    pi_lo2_model = pi_lo2.NN_model()
    pi_lo2_model.set_weights(pi_lo_batch_weights2)
    pi_lo_batch.append(pi_lo2_model)
    
    pi_b_batch = []
    pi_b1_model = pi_b1.NN_model()
    pi_b1_model.set_weights(pi_b_batch_weights1)
    pi_b_batch.append(pi_b1_model)
    pi_b2_model = pi_b2.NN_model()
    pi_b2_model.set_weights(pi_b_batch_weights2)
    pi_b_batch.append(pi_b2_model)
    
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(seed, 6000, NEpisodes, initial_state[0:2], Folder, Rand_traj)
        
    return trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch
    

# initial_state = np.array([0, -2.6, 0, 8])#Trajectories[Rand_traj][0,:]
# folder = 6
# nTraj = 2

NEpisodes = 10
Nseed = multiprocessing.cpu_count()
initial_state = Trajectories[Rand_traj][0,:]
Ncpu = Nseed
pool = MyPool(Ncpu)
args = [(seed, folder, nTraj, NEpisodes, initial_state, pi_hi_batch.get_weights(), pi_lo_batch[0].get_weights(), pi_lo_batch[1].get_weights(), pi_b_batch[0].get_weights(), pi_b_batch[1].get_weights()) for seed in range(Nseed)]
HIL_from_human_evaluation_results = pool.starmap(evaluateHIL_fromHuman, args) 
pool.close()
pool.join()



# %%

# with open('4_walls_coins_task/HIL_from_human_evaluation_results.npy', 'wb') as f:
#     np.save(f, HIL_from_human_evaluation_results)
    
# with open('Results_main/HIL_from_human_evaluation_results.npy', 'rb') as f:
#     HIL_from_human_evaluation_results = np.load(f, allow_pickle=True).tolist()

averageHIL = []

for i in range(len(HIL_from_human_evaluation_results)):
    averageHIL.append(np.mean(HIL_from_human_evaluation_results[i][6]))
best_index_agentHIL = np.argmax(averageHIL)
best_reward_indexHIL=np.argmax(HIL_from_human_evaluation_results[best_index_agentHIL][6])
best_episodeHIL=HIL_from_human_evaluation_results[best_index_agentHIL][0][best_reward_indexHIL]

coins_location = World.Foraging.CoinLocation(folder, nTraj+1, 'full_coins')
time = np.linspace(0,480,len(best_episodeHIL)) 

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
plot_data = plt.scatter(best_episodeHIL[:,0], best_episodeHIL[:,1], c=time, marker='o', cmap='cool')
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

# %% Training single trajectory random initialization
Likelihood_batch_list = []
Rand_traj = 2
size_data = len(Trajectories[Rand_traj])-1
T_set = Trajectories[Rand_traj][0:size_data,:]
# encode psi
psi = T_set[:,2].reshape(len(T_set[:,2]),1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_psi = onehot_encoder.fit_transform(psi)
# encode closest coin direction
closest_coin_direction = T_set[:,3].reshape(len(T_set[:,3]),1)
onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
T_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
Heading_set = Rotation[Rand_traj][0:size_data]
option_space = 2
M_step_epoch = 10
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-1)
# Agent_BatchHIL = BatchBW_HIL.BatchHIL_param_simplified(T_set, Heading_set, M_step_epoch, size_batch, optimizer) 
Agent_BatchHIL = BatchBW_HIL.BatchHIL(T_set, Heading_set, option_space, M_step_epoch, size_batch, optimizer, options_predictions, NN_init='random')
N=10 #number of iterations for the BW algorithm
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood = Agent_BatchHIL.Baum_Welch(N)
Likelihood_batch_list.append(likelihood)

BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN_randominit')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_randominit'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_randominit'.format(i))

# %%evaluation
# coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')
# seed = 0
# BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
# [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(seed, 3000, 1, initial_state[0:2], Folders[0], Rand_traj)

pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN_randominit')
pi_lo_batch = []
pi_b_batch = []
option_space = 2
for i in range(option_space):
    pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_randominit'.format(i)))
    pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_randominit'.format(i)))


def evaluateHIL_fromHuman(seed, Folder, Rand_traj, NEpisodes, initial_state, pi_hi_batch_weights, pi_lo_batch_weights1, pi_lo_batch_weights2, pi_b_batch_weights1, pi_b_batch_weights2):
    
    option_space = 2
    termination_space = 2
    observation_space_size = 13
    action_size = 8
    
    pi_hi = BatchBW_HIL.NN_PI_HI(option_space, observation_space_size)
    pi_lo1 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_lo2 = BatchBW_HIL.NN_PI_LO(action_size, observation_space_size)
    pi_b1 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    pi_b2 = BatchBW_HIL.NN_PI_B(termination_space, observation_space_size)
    
    pi_hi_batch = pi_hi.NN_model()
    pi_hi_batch.set_weights(pi_hi_batch_weights)
    
    pi_lo_batch = []
    pi_lo1_model = pi_lo1.NN_model()
    pi_lo1_model.set_weights(pi_lo_batch_weights1)
    pi_lo_batch.append(pi_lo1_model)
    pi_lo2_model = pi_lo2.NN_model()
    pi_lo2_model.set_weights(pi_lo_batch_weights2)
    pi_lo_batch.append(pi_lo2_model)
    
    pi_b_batch = []
    pi_b1_model = pi_b1.NN_model()
    pi_b1_model.set_weights(pi_b_batch_weights1)
    pi_b_batch.append(pi_b1_model)
    pi_b2_model = pi_b2.NN_model()
    pi_b2_model.set_weights(pi_b_batch_weights2)
    pi_b_batch.append(pi_b2_model)
    
    BatchSim = World.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(seed, 3000, NEpisodes, initial_state[0:2], Folder, Rand_traj)
        
    return trajBatch, controlBatch, OptionsBatch, TerminationBatch, psiBatch, coin_directionBatch, rewardBatch
    
NEpisodes = 100
Nseed=40
initial_state = Trajectories[Rand_traj][0,:]
Ncpu = Nseed
pool = MyPool(Ncpu)
args = [(seed, Folders[0], Rand_traj, NEpisodes, initial_state, pi_hi_batch.get_weights(), pi_lo_batch[0].get_weights(), pi_lo_batch[1].get_weights(), pi_b_batch[0].get_weights(), pi_b_batch[1].get_weights()) for seed in range(Nseed)]
HIL_from_human_evaluation_results_random_init = pool.starmap(evaluateHIL_fromHuman, args) 
pool.close()
pool.join()   

with open('Results_main/HIL_from_human_evaluation_results_random_init.npy', 'wb') as f:
    np.save(f, HIL_from_human_evaluation_results_random_init)

# %%


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


# %%
# BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN')
# for i in range(option_space):
#     BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i))
#     BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}'.format(i))

# %% Training Option Critic

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_results_4th_attempt.npy', 'rb') as f:
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
time = np.linspace(0,480,len(best_traj[n_episode][:,0]))  
 
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
z = np.polyfit(episodes, DeepSoftOC_learning_results[0][0],10)
p = np.poly1d(z)
plt.plot(episodes, p(episodes),'k--', label='Evaluation Trend')
#plt.plot(episodes, DeepSoftOC_learning_results[0][0], 'g', label = 'OC agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training OC')
plt.legend()
plt.savefig('Figures/FiguresOC/OC_training_trend.eps', format='eps')
plt.show() 

# %% Evaluating Option Critic

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_evaluation_4th_attempt_8000_steps.npy', 'rb') as f:
    DeepSoftOC_learning_evaluation = np.load(f, allow_pickle=True).tolist()
 
average_reward = []
for i in range(len(DeepSoftOC_learning_evaluation)):
    average_reward.append(np.mean(DeepSoftOC_learning_evaluation[i][3]))
best_index = np.argmax(average_reward)
best_traj_index = np.argmax(DeepSoftOC_learning_evaluation[best_index][3])


coins_location = World.Foraging.CoinLocation(6, Rand_traj+1, 'full_coins')
time = np.linspace(0,480,len(DeepSoftOC_learning_evaluation[best_index][0][best_traj_index][:,0]))  

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
