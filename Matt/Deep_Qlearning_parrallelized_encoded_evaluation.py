#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:14:41 2021

@author: vittorio
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import World
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from tensorflow import keras
import multiprocessing
import multiprocessing.pool

with open('4_walls_coins_task/Q_learning_results_encoded.npy', 'rb') as f:
    Mixture_of_DQN_encoded = np.load(f, allow_pickle=True).tolist()
    
# %%

N_agents = 5

best_reward = np.zeros(N_agents)
best_episode = [[None]*1 for _ in range(N_agents)]
best_agent = np.zeros(N_agents)
best_net = [[None]*1 for _ in range(N_agents)]

for i in range(len(Mixture_of_DQN_encoded)):
    for j in range(len(Mixture_of_DQN_encoded[i][0])):
        temp = Mixture_of_DQN_encoded[i][0][j]
        
        for k in range(N_agents):
            if temp>best_reward[k]:
                best_reward[k]=Mixture_of_DQN_encoded[i][0][j]
                best_episode[k]=Mixture_of_DQN_encoded[i][1][j]
                best_agent[k]=i
                best_net[k]=Mixture_of_DQN_encoded[i][2][j]
                break
 
# %%

coins_location = World.Foraging.CoinLocation(6, 2+1, 'full_coins') #np.random.randint(0,len(Time))

n_episode = 0
time = np.linspace(0,500,3001)  
 
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
plot_data = plt.scatter(best_episode[n_episode][:,0], best_episode[n_episode][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traj, DQN n{}, reward {}'.format(best_agent[n_episode], best_reward[n_episode]))
plt.savefig('Figures/FiguresDQN/DQN_Traj{}_reward{}_encoded.eps'.format(best_agent[n_episode], best_reward[n_episode]), format='eps')
plt.show()      


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

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]

        return states, actions, rewards, states_     
        
        
class Q_learning_NN_encoded:
    def __init__(self, seed, Folder, expert_traj, import_net=False, weights = 0):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size = self.coordinates + self.view + self.closest_coin_dir # first 2 are fore coordinates, second 2 for view, and 9 for closest coin direction
        self.Q_network = Q_learning_NN_encoded.NN_model(self)
        if import_net:
            self.Q_network.set_weights(weights)

    def NN_model(self):
        model = keras.Sequential([             
                keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                                
                keras.layers.Dense(self.env.action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2))
                                 ])              
        
        model.compile(optimizer='adam',
                               loss=tf.keras.losses.MeanSquaredError(),
                               metrics=['accuracy'])
        
        return model
    
    
    def Evaluation(self, NEpisodes, initial_state, seed):

        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        control = [[None]*1 for _ in range(NEpisodes)]
        np.random.seed(seed)
        

        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            u = np.empty((0, 1))
            current_state = self.env.reset('standard', init_state=initial_state)
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(self.view)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(self.closest_coin_dir)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))            
            cum_reward = 0 
            x = np.append(x, current_state_encoded.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                mean = np.argmax(self.Q_network(current_state_encoded.reshape(1,self.observation_space_size)))
                
                action = int(np.random.normal(mean, 1.5, 1))%(self.env.action_size-1)
                                   
                obs, reward = self.env.step(action)
                current_state = obs
                coordinates = current_state[0:2]
                psi = current_state[2]
                psi_encoded = np.zeros(self.view)
                psi_encoded[int(psi)]=1
                coin_dir_encoded = np.zeros(self.closest_coin_dir)
                coin_dir = current_state[3]
                coin_dir_encoded[int(coin_dir)]=1
                current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))   
                x = np.append(x, current_state_encoded.reshape(1, self.observation_space_size), 0)
                u = np.append(u, [[action]], 0)
                cum_reward = cum_reward + reward
                                    
            print("Episode {}: cumulative reward = {}".format(i_episode, cum_reward))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x    
            control[i_episode] = u

        
        return  reward_per_episode, traj, control
    

def evaluate(seed, Folders, Rand_traj, NEpisodes, initial_state, net_weights):
    
    agent_NN_Q_learning = Q_learning_NN_encoded(seed, Folders, Rand_traj, import_net = True, weights = net_weights)
    reward_per_episode, traj, network_weights = agent_NN_Q_learning.Evaluation(NEpisodes, initial_state, seed)
    
    return reward_per_episode, traj, network_weights

NEpisodes = 100
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2
Nseed=int(40/N_agents)

reset = 'standard'
initial_state = np.array([0, -2.6, 0, 8])
Ncpu = len(best_net)*Nseed
pool = MyPool(Ncpu)
args = [(seed, Folders, Rand_traj, NEpisodes, initial_state, net_weights) for net_weights in best_net for seed in range(Nseed)]
Q_learning_evaluation_results_encoded = pool.starmap(evaluate, args) 
pool.close()
pool.join()

# %%
with open('4_walls_coins_task/Q_learning_evaluation_results_encoded.npy', 'wb') as f:
    np.save(f, Q_learning_evaluation_results_encoded)