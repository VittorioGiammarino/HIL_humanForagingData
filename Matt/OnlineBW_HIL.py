#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:29:37 2020

@author: vittorio
"""

import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb
import time


class NN_PI_LO:
# =============================================================================
#     class for Neural network for pi_lo
# =============================================================================
    def __init__(self, action_space, size_input):
        self.action_space = action_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([             
                keras.layers.Dense(128, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                             
                keras.layers.Dense(self.action_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=1)),
                keras.layers.Softmax()
                                 ])              
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_lo.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model
    
    def PreTraining(self, TrainingSet, Labels, Nepoch):
        model = NN_PI_LO.NN_model(self)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=Nepoch)
        
        return model
        
            
class NN_PI_B:
# =============================================================================
#     class for Neural network for pi_b
# =============================================================================
    def __init__(self, termination_space, size_input):
        self.termination_space = termination_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.termination_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=3)),
                keras.layers.Softmax()
                                 ])               
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_b.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model    

    def PreTraining(self, TrainingSet, Labels):
        model = NN_PI_B.NN_model(self)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=200)
        
        return model    
            
class NN_PI_HI:
# =============================================================================
#     class for Neural Network for pi_hi
# =============================================================================
    def __init__(self, option_space, size_input):
        self.option_space = option_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(5, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=4),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.option_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=5)),
                keras.layers.Softmax()
                                ])                
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_hi.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)     
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model
    
    def PreTraining(self, TrainingSet):
        model = NN_PI_HI.NN_model(self)
        Labels = TrainingSet[:,3]
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=200)
        
        return model
    
class OnlineHIL:
    def __init__(self, TrainingSet, Labels, option_space, M_step_epoch, optimizer, NN_init = 'pre-train', NN_options = None, NN_low = None, NN_termination = None): 
        self.TrainingSet = TrainingSet
        self.Labels = Labels
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.action_space = 8 #len(np.unique(Labels))
        self.termination_space = 2
        self.zeta = 0.0001
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        
        # pi_hi net init
        pi_hi = NN_PI_HI(self.option_space, self.size_input)
        if NN_init == 'pre-train':
            pi_hi_model = pi_hi.PreTraining(self.TrainingSet)
            self.NN_options = pi_hi_model
        elif NN_init == 'from_network':
            self.NN_options = NN_options
        else:
            NN_options = pi_hi.NN_model()
            self.NN_options = NN_options
            
        # pi_lo and pi_b net init
        if NN_init == 'pre-train' and option_space==2:
            NN_low = []
            NN_termination = []
            pi_lo = NN_PI_LO(self.action_space, self.size_input)
            for options in range(self.option_space):
                NN_low.append(pi_lo.NN_model())
            self.NN_actions = NN_low
            
            pi_b = NN_PI_B(self.termination_space, self.size_input)
            Labels_b1 = TrainingSet[:,2]
            pi_b_model1 = pi_b.PreTraining(TrainingSet, Labels_b1)
            NN_termination.append(pi_b_model1)
            index_zero = np.where(Labels_b1 == 0)[0]
            Labels_b2 = np.zeros(len(TrainingSet[:,2]))
            Labels_b2[index_zero]=1
            pi_b_model2 = pi_b.PreTraining(TrainingSet, Labels_b2)
            NN_termination.append(pi_b_model2)            
            self.NN_termination = NN_termination
        elif NN_init == 'from_network':
            self.NN_actions = NN_low
            self.NN_termination = NN_termination  
        else:
            NN_low = []
            NN_termination = []
            pi_lo = NN_PI_LO(self.action_space, self.size_input)
            pi_b = NN_PI_B(self.termination_space, self.size_input)
            for options in range(self.option_space):
                NN_low.append(pi_lo.NN_model())
                NN_termination.append(pi_b.NN_model())
            self.NN_actions = NN_low
            self.NN_termination = NN_termination
        self.epochs = M_step_epoch
        self.optimizer = optimizer
        self.Lambda_Lb = 1
        self.Lambda_Lv = 0.1
        self.Lambda_DKL = 0.01        
        
    def FindStateIndex(self, value):
        stateSpace = np.unique(self.TrainingSet, axis=0)
        K = stateSpace.shape[0];
        stateIndex = 0
    
        for k in range(0,K):
            if np.sum(stateSpace[k,:]==value[0,:])==self.size_input:
                stateIndex = k
    
        return stateIndex
    
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = OnlineHIL.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
        else:
            o_prob_tilde = np.divide(zeta,option_space)
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state):
        Pi_lo = Pi_lo_parameterization(state)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state):
        Pi_b = Pi_b_parameterization(state)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(OnlineHIL.Pi_lo(a, Pi_lo_parameterization, state),0.0001,1)
        Pi_b_eval = np.clip(OnlineHIL.Pi_b(b, Pi_b_parameterization, state),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
        
    def TrainingSetID(self):
        TrainingSetID = np.empty((0,1))
        for i in range(len(self.TrainingSet)):
            ID = OnlineHIL.FindStateIndex(self,self.TrainingSet[i,:].reshape(1,self.size_input))
            TrainingSetID = np.append(TrainingSetID, [[ID]], axis=0)
            
        return TrainingSetID
    
      
    def Loss(self, phi, NN_termination, NN_options, NN_actions, t):
# =============================================================================
#         compute Loss function to minimize
# =============================================================================
        stateSpace = np.unique(self.TrainingSet, axis=0)
        loss = 0
        loss_pi_hi = 0
        loss_pi_b = 0
        loss_pi_lo = 0
        
        for at in range(self.action_space):
            for ot_past in range(self.option_space):
                for ot in range(self.option_space):
                    loss_pi_hi = loss_pi_hi - kb.sum(phi[ot_past,1,ot,:,at]*kb.log(kb.clip(NN_options(stateSpace,training=True)[:,ot],1e-10,1.0)))
                    for bt in range(self.termination_space):
                        if at==0:
                            loss_pi_lo = loss_pi_lo - kb.sum(phi[ot_past,bt,ot,:,:]*kb.log(kb.clip(NN_actions[ot](stateSpace,training=True)[:,:],1e-10,1.0)))
                        loss_pi_b = loss_pi_b - kb.sum(phi[ot_past,bt,ot,:,at]*kb.log(kb.clip(NN_termination[ot_past](stateSpace,training=True)[:,bt],1e-10,1.0)))
                                    
        loss = loss_pi_hi + loss_pi_lo + loss_pi_b
                
        
        return loss

    def OptimizeLoss(self, phi, t):
# =============================================================================
#         minimize Loss all toghether
# =============================================================================
        weights = []
        loss = 0
        T = len(self.TrainingSet)
        if t+1 == T:
            M_step_epochs = self.epochs
        else:
            M_step_epochs = self.epochs
                
        for epoch in range(M_step_epochs):
            #print('\nStart m-step for sample ', t,' iteration ', epoch+1)
        
            with tf.GradientTape() as tape:
                for i in range(self.option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = OnlineHIL.Loss(self, phi, self.NN_termination, self.NN_options, self.NN_actions, t)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*self.option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            #print('options loss:', float(loss))
        
        return loss        
     
    
    def AuxiliaryVector(self):
        N_samples = len(self.TrainingSet)
        auxiliary_vector = np.zeros((N_samples,self.action_space))
        for n in range(N_samples):
            for m in range(self.action_space):
                if int(self.Labels[n,0])==m:
                    auxiliary_vector[n,m]=1
                    
        return auxiliary_vector
        
    def Online_Baum_Welch_together(self, T_min):

        TrainingSetID = OnlineHIL.TrainingSetID(self)
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
                       
        zi = np.zeros((self.option_space, self.termination_space, self.option_space, 1))
        phi_h = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size,
                         self.action_space, self.termination_space, self.option_space,1))
        P_option_given_obs = np.zeros((self.option_space, 1))
        P_option_given_obs = self.mu.reshape((self.option_space, 1)) 
        phi = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space))
                                        
        for t in range(0,len(self.TrainingSet)):
            if t==0:
                eta=1
            else:
                eta=1/(t+1) 
                
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            zi_temp1 = np.ones((self.option_space, self.termination_space, self.option_space, 1))
            phi_h_temp = np.ones((self.option_space, self.termination_space, self.option_space, StateSpace_size,  self.action_space, 
                                  self.termination_space, self.option_space, 1))
            norm = np.zeros((len(self.mu)))
            P_option_given_obs_temp = np.zeros((self.option_space, 1))
            prod_term = np.ones((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space, 
                                 self.termination_space, self.option_space))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            
            for ot_past in range(self.option_space):
                for bt in range(self.termination_space):
                    for ot in range(self.option_space):
                        state = stateSpace[int(State),:].reshape(1,self.size_input)
                        action = int(Action)
                        zi_temp1[ot_past,bt,ot,0] = OnlineHIL.Pi_combined(ot, ot_past, action, bt, self.NN_options, 
                                                                          self.NN_actions[ot],  self.NN_termination[ot_past], state, self.zeta, 
                                                                          self.option_space)
                
                norm[ot_past] = P_option_given_obs[ot_past,0]*np.sum(zi_temp1[:,:,:,0],(1,2))[ot_past]
    
            zi_temp1[:,:,:,0] = np.divide(zi_temp1[:,:,:,0],np.sum(norm[:]))
            P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,0],1))*P_option_given_obs[:,0]),0) 
            
            zi = zi_temp1
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for bT in range(self.termination_space):
                                    for oT in range(self.option_space):
                                        prod_term[ot_past, bt, ot, st, at, bT, oT] = np.sum(zi[:,bT,oT,0]*np.sum(phi_h[ot_past,bt,ot,st,at,:,:,0],0))
                                        if at == int(Action) and st == int(State):
                                            phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (eta)*zi[ot_past,bt,ot,0]*P_option_given_obs[ot_past,0] 
                                            + (1-eta)*prod_term[ot_past,bt,ot,st,at,bT,oT]
                                        else:
                                            phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1-eta)*prod_term[ot_past,bt,ot,st,at,bT,oT]
                                    
            phi_h = phi_h_temp
            P_option_given_obs = P_option_given_obs_temp
            phi = np.sum(phi_h[:,:,:,:,:,:,:,0], (5,6))            
            
            #M-step 
            if t > T_min:
                loss = OnlineHIL.OptimizeLoss(self, phi, t)


        print('Maximization done, Total Loss:',float(loss))

                
        return self.NN_options, self.NN_actions, self.NN_termination
                
    def Online_Baum_Welch(self, T_min, StoppingTime):
        likelihood = OnlineHIL.likelihood_approximation(self)
        TrainingSetID = OnlineHIL.TrainingSetID(self)
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
        
        time_init = time.time()
        Time_list = [0]
        
        rho = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space, self.option_space, 1)) #rho filter initialiazation
        chi = np.zeros((self.option_space, 1)) #chi filter
        chi = self.mu.reshape((self.option_space, 1)) #chi filter initialization
        phi = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space))
        

        for t in range(0,len(self.TrainingSet)):
            
            if t==0:
                eta=0.5
            else:
                eta=0.5 
        
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            chi_temp_partial = np.zeros((self.option_space, self.termination_space, self.option_space, 1)) #store partial value of chi
            norm_chi = np.zeros((len(self.mu))) #store normalizing factor for chi
            chi_temp = np.zeros((self.option_space, 1)) #store final chi value temporary
            r_temp_partial = np.zeros((self.option_space, self.termination_space, self.option_space, 1)) #r numerator
            norm_r = np.zeros((len(self.mu),len(self.mu))) #normilizing factor for r
            rho_temp = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space,  
                                self.option_space, 1))
            prod_term = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space, 
                                 self.option_space, self.option_space))
            phi_temp = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space,  
                                self.option_space, 1))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            for oT_past in range(self.option_space):
                for oT in range(self.option_space):
                    for bT in range(self.termination_space):
                        state = stateSpace[int(State),:].reshape(1,self.size_input)
                        action = int(Action)                        
                        chi_temp_partial[oT_past,bT,oT,0] = OnlineHIL.Pi_combined(oT, oT_past, action, bT, self.NN_options, 
                                                                                  self.NN_actions[oT],  self.NN_termination[oT_past], state, self.zeta, 
                                                                                  self.option_space)
                        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(bT, oT, oT_past, self.NN_options, state, self.zeta, self.option_space),0.0001,1)
                        Pi_b_eval = np.clip(OnlineHIL.Pi_b(bT, self.NN_termination[oT_past], state),0.0001,1)
                        r_temp_partial[oT_past,bT,oT,0] = Pi_hi_eval*Pi_b_eval*chi[oT_past,0]
                        
                    norm_r[oT_past,oT] = chi[oT_past,0]*np.sum(r_temp_partial[:,:,:,0],(1))[oT_past,oT]
                norm_chi[oT_past] = chi[oT_past,0]*np.sum(chi_temp_partial[:,:,:,0],(1,2))[oT_past]
    
            chi_temp_partial[:,:,:,0] = np.divide(chi_temp_partial[:,:,:,0],np.sum(norm_chi[:]))
            chi_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(chi_temp_partial[:,:,:,0],1))*chi[:,0]),0) #next step chi
            norm_r = np.sum(norm_r,0)
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for oT in range(self.option_space):
                                    for oT_past in range(self.option_space):
                                        prod_term[ot_past, bt, ot, st, at, oT, oT_past] = rho[ot_past,bt,ot,st,at,oT_past,0]*np.sum(np.divide(r_temp_partial[oT_past,:,oT],norm_r[oT]))
                                        
                                    if at == int(Action) and st == int(State):
                                        rho_temp[ot_past,bt,ot,st,at,oT,0] = (eta)*np.divide(r_temp_partial[ot_past,bt,ot],norm_r[ot]) 
                                        + (1-eta)*np.sum(prod_term[ot_past,bt,ot,st,at,oT,:])
                                        phi_temp[ot_past,bt,ot,st,at,oT,0] = rho_temp[ot_past,bt,ot,st,at,oT,0]*chi_temp[oT,0] 
                                    else:
                                        rho_temp[ot_past,bt,ot,st,at,oT,0] = (1-eta)*np.sum(prod_term[ot_past,bt,ot,st,at,oT,:])
                                        phi_temp[ot_past,bt,ot,st,at,oT,0] = rho_temp[ot_past,bt,ot,st,at,oT,0]*chi_temp[oT,0] 
                                        
            chi = chi_temp
            rho = rho_temp
            phi = np.sum(phi_temp[:,:,:,:,:,:,0],5)
            
            #M-step 
            if t > T_min:
                loss = OnlineHIL.OptimizeLoss(self, phi, t)
                Time_list.append(time.time() - time_init)
                likelihood = np.append(likelihood, OnlineHIL.likelihood_approximation(self))
                
                if Time_list[-1] >= StoppingTime:
                    break
                  
        print('Maximization done, Total Loss:',float(loss))
                
        return self.NN_options, self.NN_actions, self.NN_termination, likelihood, Time_list              
                
                                        
                                        
                                        
                                        
                                        
                                        