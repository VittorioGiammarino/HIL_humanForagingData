#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:40:29 2020

@author: vittorio
"""

import numpy as np
import HierarchicalImitationLearning as hil
#import gym


class env:
    
    def reset():
        #obs = np.array([0, 0])
        obs = np.array([6.72584, -8.25106])
        return obs
    
    def reset_random():
        x = np.random.uniform(-12000,12000)
        y = np.random.uniform(-12000,12000)
        obs = np.array([x,y])
        return obs
    
    def step(action,time,speed,k,state):
        obs_x = state[0,0] + (time[k]-time[k-1])*speed[k]*np.cos(action)
        obs_y = state[0,1] + (time[k]-time[k-1])*speed[k]*np.sin(action)
        obs = np.array([obs_x, obs_y])
        
        return obs
        
    

def FlatPolicySim(model, max_epoch, nTraj, size_input,  labels_dict_rad, speed, time):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)

    for episode in range(nTraj):
        done = False
        obs = env.reset()
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
    
        for k in range(1,max_epoch):
            state = obs.reshape((1,size_input))
            # draw action
            prob_u = model(obs.reshape((1,size_input))).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled[0,:]))
            u_tot = np.append(u_tot,u)
        
            # given action, draw next state
            action = labels_dict_rad[u]
            obs = env.step(action,time,speed,k,state)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
            if done == True:
                u_tot = np.append(u_tot,0.5)
                break
        
        traj[episode][:] = x
        control[episode][:] = u_tot
        flag = np.append(flag,done)
        
    return traj, control, flag

def VideoFlatPolicy(model, max_epoch, size_input, labels_dict_rad, speed, time):

    for episode in range(1):
        obs = env.reset()
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
        
        for k in range(1,max_epoch):
            state = obs.reshape((1,size_input))
            # draw action
            prob_u = model(obs.reshape((1,size_input))).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled[0,:]))
            u_tot = np.append(u_tot,u)
        
            # given action, draw next state
            action = labels_dict_rad[u]
            obs = env.step(action,time,speed,k,state)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
            
    return x, u_tot
    
def HierarchicalPolicySim(Triple, zeta, mu, max_epoch, nTraj, option_space, size_input, labels_dict_rad, speed, time):
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    Option = [[None]*1 for _ in range(nTraj)]
    Termination = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)
    
    for episode in range(nTraj):
        done = False
        obs = env.reset()
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        
        # Initial Option
        prob_o = mu
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[0]):
            prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled))
        o_tot = np.append(o_tot,o)
        
        # Termination
        state = obs.reshape((1,size_input))
        prob_b = Triple.NN_termination(np.append(state,[[o]], axis=1)).numpy()
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
        
        o_prob_tilde = np.empty((1,option_space))
        if b_bool == True:
            o_prob_tilde = Triple.NN_options(state).numpy()
        else:
            o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
            o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)
        
        for k in range(1,max_epoch):
            state = obs.reshape((1,size_input))
            # draw action
            prob_u = Triple.NN_actions(np.append(state,[[o]], axis=1)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            u_tot = np.append(u_tot,u)
            
            # given action, draw next state
            action = labels_dict_rad[u]
            obs = env.step(action,time,speed,k,state)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
            if done == True:
                u_tot = np.append(u_tot,0.5)
                break
            
            # Select Termination
            # Termination
            state_plus1 = obs.reshape((1,size_input))
            prob_b = Triple.NN_termination(np.append(state_plus1,[[o]], axis=1)).numpy()
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
        
            o_prob_tilde = np.empty((1,option_space))
            if b_bool == True:
                o_prob_tilde = Triple.NN_options(state_plus1).numpy()
            else:
                o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
                o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
            
        
        traj[episode][:] = x
        control[episode][:]=u_tot
        Option[episode][:]=o_tot
        Termination[episode][:]=b_tot
        flag = np.append(flag,done)
        
    return traj, control, Option, Termination, flag                
    
def VideoHierarchicalPolicy(Triple, zeta, mu, max_epoch, option_space, size_input, labels_dict_rad, speed, time):

    for episode in range(1):
        done = False
        obs = env.reset()
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        
        # Initial Option
        prob_o = mu
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[0]):
            prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled))
        o_tot = np.append(o_tot,o)
        
        # Termination
        state = obs.reshape((1,size_input))
        prob_b = Triple.NN_termination(np.append(state,[[o]], axis=1)).numpy()
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
        
        o_prob_tilde = np.empty((1,option_space))
        if b_bool == True:
            o_prob_tilde = Triple.NN_options(state).numpy()
        else:
            o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
            o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)
        
        for k in range(1,max_epoch):
            state = obs.reshape((1,size_input))
            # draw action
            prob_u = Triple.NN_actions(np.append(state,[[o]], axis=1)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            u_tot = np.append(u_tot,u)
            
            # given action, draw next state
            action = labels_dict_rad[u]
            obs = env.step(action,time,speed,k,state)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
            if done == True:
                u_tot = np.append(u_tot,0.5)
                break
            
            # Select Termination
            # Termination
            state_plus1 = obs.reshape((1,size_input))
            prob_b = Triple.NN_termination(np.append(state_plus1,[[o]], axis=1)).numpy()
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
        
            o_prob_tilde = np.empty((1,option_space))
            if b_bool == True:
                o_prob_tilde = Triple.NN_options(state_plus1).numpy()
            else:
                o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
                o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
            
    return x, u_tot, o_tot, b_tot

def HardCoded_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, labels_dict_rad, speed, time, tol, water_locations):
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    Option = [[None]*1 for _ in range(nTraj)]
    Termination = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)
    
    for episode in range(nTraj):
        obs = env.reset_random()
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        selected_water = 0

        # initial option
        prob_o = mu
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[0]):
            prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled))
        o_tot = np.append(o_tot,o)

        # Termination
        state = obs.reshape((1,size_input))
        prob_b = np.array([[1,0]])
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
        
        o_prob_tilde = np.empty((1,option_space))
        if b_bool == True:
            o_prob_tilde = mu
        else:
            o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
            o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)
        
        for k in range(1,max_epoch):
            state = obs.reshape((1,size_input))
            # draw action
            prob_u, selected_water = hil.HardCoded_policy.pi_lo(state, o, water_locations, action_space, tol, selected_water)
            prob_u = prob_u.reshape((1,action_space))
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            u_tot = np.append(u_tot,u)
            
            # given action, draw next state
            action = labels_dict_rad[u]
            obs = env.step(action,time,speed,k,state)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
                  
            # Select Termination
            # Termination
            state_plus1 = obs.reshape((1,size_input))
            prob_b = hil.HardCoded_policy.pi_b(state_plus1, o, water_locations, tol, selected_water)
            prob_b = prob_b.reshape((1,2))
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
        
            o_prob_tilde = np.empty((1,option_space))
            if b_bool == True:
                o_prob_tilde = hil.HardCoded_policy.pi_hi(state_plus1, water_locations, option_space)
                o_prob_tilde = o_prob_tilde.reshape((1,option_space))
            else:
                o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
                o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)

            
        
        traj[episode] = x
        control[episode]=u_tot
        Option[episode]=o_tot
        Termination[episode]=b_tot
        #flag = np.append(flag,done)
        
    return traj, control, Option, Termination             
    

