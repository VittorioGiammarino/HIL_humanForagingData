#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:44:57 2021

@author: vittorio
"""

import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import World

# %%
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
        
        
class Q_learning_NN:
    def __init__(self, seed, Folder, expert_traj, import_net=False, weights = 0):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.Q_network = Q_learning_NN.NN_model(self)
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
    
    def Training(self, NEpisodes):
        
        gamma = 0.9 
        epsilon = 0.1
        reward_per_episode =[]
        batch_size = 1
        alpha = 0.1
        
        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            cum_reward = 0 
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                epsilon = max(epsilon*0.99, 0.01)
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)      
                
                obs, reward = self.env.step(action)
                new_state = obs
                
                future_optimal_value = np.max(self.Q_network(new_state.reshape(1,len(new_state))),1)
                learned_value = reward + gamma*future_optimal_value
                
                old_value = self.Q_network(current_state.reshape(1,len(current_state))).numpy()
                y = old_value
                y[np.arange(batch_size),action] = (1-alpha)*old_value[np.arange(batch_size),action] + alpha*learned_value
            
                self.Q_network.fit(current_state.reshape(1,len(current_state)), y, epochs=1, verbose = 0)
                
                current_state = new_state
                cum_reward = cum_reward + reward
                                
                    
            print("Episode reward {}".format(cum_reward))
            reward_per_episode.append(cum_reward)

                
            # end = time.time()
            # if end-start > maxTime:
            #     break
                
        
        mean = np.mean(reward_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,i_episode+1,i_episode+1),reward_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,i_episode+1,i_episode+1), mean*np.ones(i_episode+1), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Training NN Q Learning')
        plt.savefig('cartpole_Tain_NN_Qlearning.eps', format="eps")  
        
        return reward_per_episode
    
    
    def Training_buffer(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        network_weights = [[None]*1 for _ in range(NEpisodes)]
        batch_size = 256
        Buffer = ReplayBuffer(30000, self.observation_space_size)
        
        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            current_state = self.env.reset(reset, initial_state)
            cum_reward = 0 
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)       
                
                obs, reward = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state)
                current_state = new_state
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                cum_reward = cum_reward + reward
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state = Buffer.sample_buffer(batch_size)
                
                    future_optimal_value = np.max(self.Q_network(new_state),1)
                    learned_value = reward + gamma*future_optimal_value
                
                    y = self.Q_network(state).numpy()
                    y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                    self.Q_network.fit(state, y, epochs=1, verbose = 0)
                                
                                
                    
            print("Episode {}: cumulative reward = {} (seed = {})".format(i_episode, cum_reward, seed))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x
            network_weights[i_episode] = self.Q_network.get_weights()
            
        return reward_per_episode, traj, network_weights 
    
    def Training_buffer_Double(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        network_weights = [[None]*1 for _ in range(NEpisodes)]
        batch_size = 256
        Buffer = ReplayBuffer(50000, self.observation_space_size)
        
        Target_Q_net = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=5),
                               bias_initializer=keras.initializers.Zeros()),                                
            keras.layers.Dense(self.env.action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=6))
                                 ])              
        
        Target_Q_net.set_weights(self.Q_network.get_weights())
        
        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            current_state = self.env.reset(reset, initial_state)
            cum_reward = 0 
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)       
                     
                
                obs, reward = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state)
                current_state = new_state
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                cum_reward = cum_reward + reward
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state = Buffer.sample_buffer(batch_size)
                                                        
                    future_optimal_value = np.max(Target_Q_net(new_state),1)
                    learned_value = reward + gamma*future_optimal_value 
            
                    y = self.Q_network(state).numpy()
                    y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
            
                    self.Q_network.fit(state, y, epochs=1, verbose = 0)
                    
                    if np.mod(t,250)==0:
                        Target_Q_net.set_weights(self.Q_network.get_weights())
  

            print("Episode {}: cumulative reward = {} (seed = {})".format(i_episode, cum_reward, seed))
            network_weights[i_episode] = self.Q_network.get_weights()
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x

        
        return reward_per_episode, traj, network_weights 
        
    def Evaluation(self, NEpisodes, initial_state, seed):

        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        control = [[None]*1 for _ in range(NEpisodes)]
        np.random.seed(seed)
        

        for i_episode in range(NEpisodes):
            x = np.empty((0, self.observation_space_size))
            u = np.empty((0, 1))
            current_state = self.env.reset('standard', init_state=initial_state)
            cum_reward = 0 
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            for t in range(3000):
                # env.render()
                mean = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                
                action = int(np.random.normal(mean, 1.5, 1))%(self.env.action_size-1)
                                   
                obs, reward = self.env.step(action)
                current_state = obs
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                u = np.append(u, [[action]], 0)
                cum_reward = cum_reward + reward
                                    
            print("Episode {}: cumulative reward = {}".format(i_episode, cum_reward))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x    
            control[i_episode] = u

        
        return  reward_per_episode, traj, control
        

# NEpisodes = 20
# Folders = 6 #[6, 7, 11, 12, 15]
# Rand_traj = 4
# seed = 0

# agent_NN_Q_learning_buffer = Q_learning_NN(seed, Folders, Rand_traj)
# reward_per_episode, traj, network_weights = agent_NN_Q_learning_buffer.Training_buffer(NEpisodes, seed)



# #%%


# coins_location = World.Foraging.CoinLocation(Folders, Rand_traj+1, 'full_coins') #np.random.randint(0,len(Time))

# n_episode = 9   
# time = np.linspace(0,500,3001)  
 
# sigma1 = 0.5
# circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
# sigma2 = 1.1
# circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
# sigma3 = 1.8
# circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
# sigma4 = 1.3
# circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
# fig, ax = plt.subplots()
# ax.add_artist(circle1)
# ax.add_artist(circle2)
# ax.add_artist(circle3)
# ax.add_artist(circle4)
# plot_data = plt.scatter(traj[19][:,0], traj[19][:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
# plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
# cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
# cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('DQN trial')
# plt.savefig('Figures/FiguresDQN/simple_trial.eps', format='eps')
# plt.show()      
