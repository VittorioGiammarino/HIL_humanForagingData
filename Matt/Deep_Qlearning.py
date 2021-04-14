#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:44:57 2021

@author: vittorio
"""

import matplotlib.pyplot as plt
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
    def __init__(self, seed, Folder, expert_traj):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.Q_network = Q_learning_NN.NN_model(self)

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
    
    
    def Training_buffer(self, NEpisodes):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        batch_size = 256
        Buffer = ReplayBuffer(10000, self.observation_space_size)
        
        for i_episode in range(NEpisodes):
            current_state = self.env.reset('random')
            cum_reward = 0 
            
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
                cum_reward = cum_reward + reward
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state = Buffer.sample_buffer(batch_size)
                
                    future_optimal_value = np.max(self.Q_network(new_state),1)
                    learned_value = reward + gamma*future_optimal_value
                
                    y = self.Q_network(state).numpy()
                    y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                    self.Q_network.fit(state, y, epochs=1, verbose = 0)
                                
                                
                    
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

    def Training_buffer_Double(self, NEpisodes):
        
        gamma = 0.99 
        epsilon = 0.5
        reward_per_episode =[]
        batch_size = 256
        Buffer = ReplayBuffer(10000, self.observation_space_size)
        
        Double_Q_net = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(self.observation_space_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=5),
                               bias_initializer=keras.initializers.Zeros()),                                
            keras.layers.Dense(self.env.action_space.n, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=6))
                                 ])              
        
        Double_Q_net.compile(optimizer='adam',
                             loss=tf.keras.losses.MeanSquaredError(),
                             metrics=['accuracy'])
        
        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            cum_reward = 0
            
            for t in range(3000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                b = np.argmax(Double_Q_net(current_state.reshape(1,len(current_state))))
                
                if np.mod(t,50)==0:
                    epsilon = epsilon/2
            
                if np.random.random() <= epsilon: 
                    action = np.random.randint(0,self.env.action_size)       
                    
                if np.random.random() <= epsilon: 
                    b = np.random.randint(0,self.env.action_size)    
                
                obs, reward = self.env.step(action)
                new_state = obs
                Buffer.store_transition(current_state, action, reward, new_state)
                current_state = new_state
                cum_reward = cum_reward + reward
                
                if Buffer.mem_cntr>batch_size:
                    state, action, reward, new_state = Buffer.sample_buffer(batch_size)
                    
                    if np.random.random()<=0.5:
                
                        future_optimal_value = np.max(Double_Q_net(new_state),1)
                        learned_value = reward + gamma*future_optimal_value 
                
                        y = self.Q_network(state).numpy()
                        y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                        self.Q_network.fit(state, y, epochs=1, verbose = 0)
                        
                    else:
                        future_optimal_value = np.max(self.Q_network(new_state),1)
                        learned_value = reward + gamma*future_optimal_value
                        y = Double_Q_net(state).numpy()
                        y[np.arange(batch_size),action] = learned_value[np.arange(batch_size)]
                
                        Double_Q_net.fit(state, y, epochs=1, verbose = 0)                        
                                
                  

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
        
    def Evaluation(self, NEpisodes):

        steps_per_episode =[]

        for i_episode in range(NEpisodes):
            current_state = self.env.reset()
            
            for t in range(5000):
                # env.render()
                action = np.argmax(self.Q_network(current_state.reshape(1,len(current_state))))
                   
                obs, reward, done, _ = self.env.step(action)
                current_state = obs
                    
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    steps_per_episode.append(t+1)
                    break
                
        
        mean = np.mean(steps_per_episode)
        
        plt.figure()
        plt.plot(np.linspace(1,NEpisodes,NEpisodes),steps_per_episode, 'g', label='Policy')
        plt.plot(np.linspace(1,NEpisodes,NEpisodes), mean*np.ones(NEpisodes), 'k--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Steps lasted')
        plt.legend()
        plt.title('Evaluation NN Q Learning')
        plt.savefig('cartpole_Eval_NN_Qlearning.eps', format="eps")  
        
        return steps_per_episode
        

NEpisodes = 5
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 4

rp_list = []
tabular_training_list = []
tabular_eval_list = []
NN_training_list = []
NN_eval_list = []
NN_buffer_training_list = []
NN_buffer_eval_list = []

for seed in range(1):
    # rp_list.append(random_policy(seed, NEpisodes))
    # agent_Q_learning_tabular = Q_learning_tabular(seed)
    # tabular_training_list.append(agent_Q_learning_tabular.Training(NEpisodes))
    # tabular_eval_list.append(agent_Q_learning_tabular.Evaluation(NEpisodes))
    # agent_NN_Q_learning = Q_learning_NN(seed)
    # NN_training_list.append(agent_NN_Q_learning.Training(NEpisodes))
    # NN_eval_list.append(agent_NN_Q_learning.Evaluation(NEpisodes))
    agent_NN_Q_learning_buffer = Q_learning_NN(seed, Folders, Rand_traj)
    NN_buffer_training_list.append(agent_NN_Q_learning_buffer.Training_buffer(NEpisodes))
    NN_buffer_eval_list.append(agent_NN_Q_learning_buffer.Evaluation(NEpisodes))
    # agent_NN_Double_Q_learning_buffer = Q_learning_NN(seed)
    # agent_NN_Double_Q_learning_buffer.Training_buffer_Double(NEpisodes)
    # agent_NN_Double_Q_learning_buffer.Evaluation(NEpisodes)


#%%

rp = np.asarray(rp_list)
tabular_training = np.asarray(tabular_training_list)
tabular_eval = np.asarray(tabular_eval_list)
NN_training = np.asarray(NN_training_list)
NN_eval = np.asarray(NN_eval_list)
NN_buffer_training = np.asarray(NN_buffer_training_list)
NN_buffer_eval = np.asarray(NN_buffer_eval_list)


# Training
ave_tab_train = np.mean(tabular_training,0)
std_tab_train = np.std(tabular_training,0)
ave_NN_train = np.mean(NN_training,0)
std_NN_train = np.std(NN_training,0)
ave_NN_buffer_train = np.mean(NN_buffer_training,0)
std_NN_buffer_train = np.std(NN_buffer_training,0)


fig, ax = plt.subplots()
ax.plot(np.arange(len(ave_tab_train)), ave_tab_train, label='Tabular Q-learning', c='b')
ax.fill_between(np.arange(len(ave_tab_train)), ave_tab_train-std_tab_train, ave_tab_train+std_tab_train, alpha=0.1, facecolor='b')
ax.plot(np.arange(len(ave_NN_train)), ave_NN_train, label='Deep Q-learning', c='g')
ax.fill_between(np.arange(len(ave_NN_train)), ave_NN_train-std_NN_train, ave_NN_train+std_NN_train, alpha=0.1, facecolor='g')
ax.plot(np.arange(len(ave_NN_buffer_train)), ave_NN_buffer_train, label='Deep Q-learning w/ buffer', c='k')
ax.fill_between(np.arange(len(ave_NN_buffer_train)), ave_NN_buffer_train-std_NN_buffer_train, ave_NN_buffer_train+std_NN_buffer_train, alpha=0.1, facecolor='k')
ax.legend(facecolor = '#d8dcd6')
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Reward')
ax.set_title('Training')
plt.savefig('Training_comparison.png', format='png')


# Evaluation
ave_rp = np.mean(rp,0)
std_rp = np.std(rp,0)
ave_tabular_eval = np.mean(tabular_eval,0)
std_tabular_eval = np.std(tabular_eval,0)
ave_NN_eval = np.mean(NN_eval,0)
std_NN_eval = np.std(NN_eval,0)
ave_NN_buffer_eval = np.mean(NN_buffer_eval,0)
std_NN_buffer_eval = np.std(NN_buffer_eval,0)

fig, ax = plt.subplots()
ax.plot(np.arange(len(ave_rp)), ave_rp, label='random Policy', c='r')
ax.fill_between(np.arange(len(ave_rp)), ave_rp-std_rp, ave_rp+std_rp, alpha=0.1, facecolor='r')
ax.plot(np.arange(len(ave_tabular_eval)), ave_tabular_eval, label='Tabular Q-learning', c='b')
ax.fill_between(np.arange(len(ave_tabular_eval)), ave_tabular_eval-std_tabular_eval, ave_tabular_eval+std_tabular_eval, alpha=0.1, facecolor='b')
ax.plot(np.arange(len(ave_NN_eval)), ave_NN_eval, label='Deep Q-learning', c='g')
ax.fill_between(np.arange(len(ave_NN_eval)), ave_NN_eval-std_NN_eval, ave_NN_eval+std_NN_eval, alpha=0.1, facecolor='g')
ax.plot(np.arange(len(ave_NN_buffer_eval)), ave_NN_buffer_eval, label='Deep Q-learning w/ buffer', c='k')
ax.fill_between(np.arange(len(ave_NN_buffer_eval)), ave_NN_buffer_eval-std_NN_buffer_eval, ave_NN_buffer_eval+std_NN_buffer_eval, alpha=0.1, facecolor='k')
ax.legend(facecolor = '#d8dcd6')
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Reward')
ax.set_title('Evaluation')
plt.savefig('Evaluation_comparison.png', format='png')
