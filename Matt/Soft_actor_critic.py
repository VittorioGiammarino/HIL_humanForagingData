#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:03:14 2021

@author: vittorio
"""


""

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb
import World
import multiprocessing
import BatchBW_HIL
import multiprocessing.pool
import matplotlib.pyplot as plt
import matplotlib.patches as ptch


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

# %%
class ReplayBuffer():
    def __init__(self, max_size, input_dims, input_dims_encoded, seed):
        np.random.seed(seed)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.state_memory_encoded = np.zeros((self.mem_size, input_dims_encoded), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory_encoded = np.zeros((self.mem_size, input_dims_encoded), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, state_encoded, action, reward, state_, state_encoded_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state_memory_encoded[index] = state_encoded
        self.new_state_memory[index] = state_
        self.new_state_memory_encoded[index] = state_encoded_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_encoded = self.state_memory_encoded[batch]
        states_ = self.new_state_memory[batch]
        states_encoded_ = self.new_state_memory_encoded[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        
        return states, states_encoded, actions, rewards, states_, states_encoded_
    
def NN_model(input_size, output_size, init_seed):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=init_seed),
                               bias_initializer=keras.initializers.Zeros()),         
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=init_seed),
                               bias_initializer=keras.initializers.Zeros()),                           
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2))
                             ])         
    
    return model

class soft_Actor_Critic_with_DQN:
    def __init__(self, seed, Folder, expert_traj):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.zeta = 0
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
                    
        self.Buffer = ReplayBuffer(1000000, self.observation_space_size, self.observation_space_size_encoded, seed)
        pi_lo_class = BatchBW_HIL.NN_PI_LO(self.env.action_size, self.observation_space_size_encoded)
        self.pi_lo_batch = pi_lo_class.NN_model()
        self.Q_net = NN_model(self.observation_space_size, self.env.action_size, 1)
        self.DQ_net = NN_model(self.observation_space_size, self.env.action_size, 2)
        self.V_net = NN_model(self.observation_space_size, 1, 3)
        self.V_net_target = NN_model(self.observation_space_size, 1, 4)
        self.optimizer_actor = keras.optimizers.Adamax(learning_rate=1e-5)
        self.optimizer_critic = keras.optimizers.Adamax(learning_rate=1e-2)

    
    def Training(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        gamma = 0.99 
        tau = 0.005
        reward_per_episode =[]
        batch_size = 512
        np.random.seed(seed)
        traj = [[None]*1 for _ in range(NEpisodes)]
        pi_lo = [[None]*1 for _ in range(NEpisodes)]
        
        for i_episode in range(NEpisodes):
            
            T = i_episode + 1
            x = np.empty((0, self.observation_space_size))

            current_state = self.env.reset(reset, initial_state)
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(self.view)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(self.closest_coin_dir)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
            cum_reward = 0 
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            
            # draw action
            prob_u = self.pi_lo_batch(current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            current_action = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
            for t in range(3000):
                
                obs, reward = self.env.step(current_action)
                new_state = obs
                
                coordinates = new_state[0:2]
                psi = new_state[2]
                psi_encoded = np.zeros(self.view)
                psi_encoded[int(psi)]=1
                coin_dir_encoded = np.zeros(self.closest_coin_dir)
                coin_dir = new_state[3]
                coin_dir_encoded[int(coin_dir)]=1
                new_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
                
                self.Buffer.store_transition(current_state, current_state_encoded, current_action, reward, new_state, new_state_encoded)
                
                          
                # training here TODO
                if self.Buffer.mem_cntr>batch_size and np.mod(T,5)==0:
                    
                    states, states_encoded, actions, rewards, new_states, new_states_encoded = self.Buffer.sample_buffer(batch_size)

                    Q_new_states_values = self.Q_net(new_states)
                    DQ_new_states_values = self.DQ_net(new_states)
                    y_v = kb.sum(self.pi_lo_batch(new_states_encoded)*Q_new_states_values,1)
                    y_vD = kb.sum(self.pi_lo_batch(new_states_encoded)*DQ_new_states_values,1)
                    y_vFinal = np.min(np.concatenate((y_v.numpy().reshape(batch_size,1), y_vD.numpy().reshape(batch_size,1)),1),1)
                    
                    weights_v = []
                    with tf.GradientTape() as tape:
                        weights_v.append(self.V_net.trainable_weights)                    
                        tape.watch(weights_v)
                        value = self.V_net(states)
                        value_target = y_vFinal - kb.sum(self.pi_lo_batch(states_encoded)*kb.log(kb.clip(self.pi_lo_batch(states_encoded),1e-10,1.0)),1)
                        loss_v = keras.losses.MSE(value, value_target)
            
                    grads_v = tape.gradient(loss_v,weights_v)

                    self.optimizer_critic.apply_gradients(zip(grads_v[0][:], self.V_net.trainable_weights))
                    v_weights = self.V_net.get_weights()
                    v_target_weights = self.V_net_target.get_weights()
                                        
                    if T==1:
                        self.V_net_target.set_weights(v_weights)

                        
                        
                    value_ = gamma*self.V_net_target(new_states)[:,0] #kb.sum(self.pi_lo_batch[option](new_states_encoded)*Q_new_states_values,1) 
                    y_j = (rewards + value_).numpy()
                    
                    m = actions.shape[0]
                    n = self.env.action_size
                    auxiliary_vector = np.zeros((m,n))
                    for k in range(m):
                        auxiliary_vector[k,actions[k]] = 1
                        
                    
                    with tf.GradientTape() as tape:
                        tape.watch(self.Q_net.trainable_weights)
                        y = kb.sum(auxiliary_vector*self.Q_net(states),1)
                        Q_net_Loss = keras.losses.MSE(y, y_j)
                    gradQ = tape.gradient(Q_net_Loss, self.Q_net.trainable_weights)
                    self.optimizer_critic.apply_gradients(zip(gradQ, self.Q_net.trainable_weights))
                    
                    with tf.GradientTape() as tape:
                        tape.watch(self.DQ_net.trainable_weights)
                        yD = kb.sum(auxiliary_vector*self.DQ_net(states),1)
                        DQ_net_Loss = keras.losses.MSE(yD, y_j)  
                    gradDQ = tape.gradient(DQ_net_Loss, self.DQ_net.trainable_weights)
                    self.optimizer_critic.apply_gradients(zip(gradDQ, self.DQ_net.trainable_weights))
                    
                    
                    Q_values = self.Q_net(states)
                    DQ_values = self.DQ_net(states)
                    Q_vFinal = np.min(np.concatenate((Q_values.numpy().reshape(batch_size,self.env.action_size,1), DQ_values.numpy().reshape(batch_size,self.env.action_size,1)),2),2)
                    
                    weights_lo = []
                    with tf.GradientTape() as tape:
                        weights_lo.append(self.pi_lo_batch.trainable_weights)                    
                        tape.watch(weights_lo)
                        loss_lo = tf.math.reduce_mean(self.pi_lo_batch(states_encoded)*kb.log(kb.clip(self.pi_lo_batch(states_encoded),1e-10,1.0))-auxiliary_vector*kb.log(kb.clip(self.pi_lo_batch(states_encoded),1e-10,1.0))*Q_vFinal)
            
                    grads_lo = tape.gradient(loss_lo,weights_lo)
                    self.optimizer_actor.apply_gradients(zip(grads_lo[0][:], self.pi_lo_batch.trainable_weights))
                    
                    
                    if T==1:
                        self.V_net_target.set_weights(v_weights)
                    else:
                        for i in range(len(v_target_weights)):
                            v_target_weights[i] = (1-tau)*v_target_weights[i]+tau*v_weights[i]
                        self.V_net_target.set_weights(v_target_weights)                 
                 
                    
                # draw next action
                prob_u = self.pi_lo_batch(new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                for i in range(1,prob_u_rescaled.shape[1]):
                    prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                temp = np.where(draw_u<=prob_u_rescaled)[1]
                if temp.size == 0:
                    draw_u = 1
                    new_action = np.argmax(prob_u)
                else:
                    new_action = np.amin(np.where(draw_u<prob_u_rescaled)[1])  
                
                current_state = new_state
                current_state_encoded = new_state_encoded
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                current_action = new_action
                cum_reward = cum_reward + reward                
             
            print("Episode {}: cumulative reward = {} (seed = {})".format(i_episode, cum_reward, seed))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x
            pi_lo[i_episode] = self.pi_lo_batch.get_weights()

        return reward_per_episode, traj, pi_lo
    

    
def train(NEpisodes, seed, Folders, Rand_traj):
    soft_critic = soft_Actor_Critic_with_DQN(seed, Folders, Rand_traj)
    reward_per_episode_SAC, traj_SAC, pi_lo = soft_critic.Training(NEpisodes, seed, reset = 'standard', initial_state = np.array([0, -2.6, 0, 8]))
    return reward_per_episode_SAC, traj_SAC, pi_lo


NEpisodes = 1000
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2
seeds = [31] #[13, 29, 33, 36, 27, 21, 15, 31]
SAC_learning_results = []


# Ncpu = 30
# pool = MyPool(Ncpu)
# args = [(NEpisodes, seed, Folders, Rand_traj) for seed in range(Ncpu)]
# SAC_learning_results = pool.starmap(train, args) 
# pool.close()
# pool.join()

for seed in seeds:

    soft_critic = soft_Actor_Critic_with_DQN(seed, Folders, Rand_traj)
    reward_per_episode_SAC, traj_SAC, pi_lo = soft_critic.Training(NEpisodes, seed, reset = 'standard', initial_state = np.array([0, -2.6, 0, 8]))
    
    SAC_learning_results.append([reward_per_episode_SAC, traj_SAC, pi_lo])
    
    
with open('RL_algorithms/Soft_Actor_Critic/Results/SAC_learning_results.npy', 'wb') as f:
    np.save(f, SAC_learning_results)