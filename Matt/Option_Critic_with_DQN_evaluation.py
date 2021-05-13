#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:04:09 2021

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
import BatchBW_HIL

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_results_second_attempt.npy', 'rb') as f:
    DeepSoftOC_learning_results = np.load(f, allow_pickle=True).tolist()
    
# %%

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


            
  
# %%

coins_location = World.Foraging.CoinLocation(6, 2+1, 'full_coins') #np.random.randint(0,len(Time))

best = np.argmax(best_reward)
n_episode = best
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
plt.savefig('Figures/FiguresOC/Traj_VS_Options_traj_reward{}.eps'.format(best_reward[n_episode]), format='eps')
plt.show()  

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
    
class Option_Critic_with_DQN_eval:
    def __init__(self, seed, Folder, expert_traj, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights):
        self.env = World.Foraging.env(Folder, expert_traj)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.option_space = option_space
        self.zeta = 0
        self.eta = 0.00001
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
                
        pi_hi = BatchBW_HIL.NN_PI_HI(self.option_space, self.observation_space_size_encoded)
        pi_hi_batch = pi_hi.NN_model()
        pi_hi_batch.set_weights(pi_hi_weights)
        
        pi_lo_batch = []
        pi_b_batch = []
        pi_lo_class = BatchBW_HIL.NN_PI_LO(self.env.action_size, self.observation_space_size_encoded)
        pi_b_class = BatchBW_HIL.NN_PI_B(2, self.observation_space_size_encoded)
        for i in range(self.option_space):
            pi_lo_batch.append(pi_lo_class.NN_model())
            pi_lo_batch[i].set_weights(pi_lo_weights[i])

            pi_b_batch.append(pi_b_class.NN_model())
            pi_b_batch[i].set_weights(pi_b_weights[i])
                        
        self.pi_hi_batch = pi_hi_batch
        self.pi_lo_batch = pi_lo_batch
        self.pi_b_batch = pi_b_batch


    def evaluateOC(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
 
        reward_per_episode =[]
        batch_size = 512
        np.random.seed(seed)
        traj = [[None]*1 for _ in range(NEpisodes)]
        Option = [[None]*1 for _ in range(NEpisodes)]
        Termination = [[None]*1 for _ in range(NEpisodes)]
        
        
        for i_episode in range(NEpisodes):
            
            o_tot = np.empty((0,0),int)
            b_tot = np.empty((0,0),int)
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
            
            # initial cost
            cost = 0
            
            # Initial Option
            prob_o = self.pi_hi_batch(current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            option = np.amin(np.where(draw_o<prob_o_rescaled))
            o_tot = np.append(o_tot,option)
            
            # draw action
            prob_u = self.pi_lo_batch[option](current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
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
                
                
                # Termination
                prob_b = self.pi_b_batch[option](new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                temp = np.where(draw_b<=prob_b_rescaled)[1]
                if temp.size == 0:
                    draw_b = 1
                    b = np.argmax(prob_b)
                else:
                    b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
                    
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False                
                 
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi_batch(new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,option] = 1 - self.zeta + self.zeta/self.option_space
                
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                option = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,option)
                
                # draw next action
                prob_u = self.pi_lo_batch[option](new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
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
             
                
            reward_per_episode.append(cum_reward)
            average = np.sum(np.array(reward_per_episode))/(i_episode+1)
            print("Episode {}: cumulative reward = {}, average = {} (seed = {})".format(i_episode, cum_reward, average, seed))
            traj[i_episode] = x
            Option[i_episode] = o_tot
            Termination[i_episode] = b_tot
        
        return traj, Option, Termination, reward_per_episode
 
    
episode = 496

best = 0 #np.argmax(best_reward) # best agent is number 0 which had a reward of 245 and was at iteration 428
   
NEpisodes = 100
option_space = 2
Folders = 6 #[6, 7, 11, 12, 15]
Rand_traj = 2
seeds = [13, 29, 33, 36, 27, 21, 15, 31]
DeepSoftOC_learning_evaluation = []

for seed in seeds:

    option_critic = Option_Critic_with_DQN_eval(seed, Folders, Rand_traj, option_space, best_pi_hi[best], best_pi_lo[best], best_pi_b[best])
    traj, Option, Termination, reward_per_episode  = option_critic.evaluateOC(NEpisodes, seed, reset = 'standard', initial_state = np.array([0, -2.6, 0, 8]))
    
    DeepSoftOC_learning_evaluation.append([traj, Option, Termination, reward_per_episode])
# %%

with open('RL_algorithms/Option_critic_with_DQN/Results/DeepSoftOC_learning_evaluation.npy', 'wb') as f:
    np.save(f, DeepSoftOC_learning_evaluation)