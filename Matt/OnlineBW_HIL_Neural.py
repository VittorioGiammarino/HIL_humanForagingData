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


class ReplayBuffer():
    def __init__(self, max_size, input_dims, action_dims, seed):
        np.random.seed(seed)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action_memory_encoded = np.zeros((self.mem_size, action_dims), dtype=np.int32)

    def store_transition(self, state, action, action_encoded):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.action_memory_encoded[index] = action_encoded
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        actions_encoded = self.action_memory_encoded[batch]
        
        return states, actions, actions_encoded

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
        model.fit(TrainingSet, Labels, epochs=10)
        
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
   
    
def NN_model(input_size, output_size, seed_init):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=0.01, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),        
            keras.layers.Dense(256, activation='relu', 
                               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=0.01, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),                         
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=0.01, seed=seed_init))
                             ])       

    model.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])     
    
    return model
    

class OnlineHIL:
    def __init__(self, TrainingSet, Labels, option_space, M_step_epoch, optimizer, NN_init = 'pre-train', NN_options = None, NN_low = None, NN_termination = None): 
        self.TrainingSet_permanent = TrainingSet
        self.action_space = len(np.unique(Labels,axis=0))
        self.action_dictionary = np.unique(Labels, axis = 0)
        
        labels = np.zeros((len(Labels)))
        for i in range(len(Labels)):
            for j in range(self.action_space):
                if Labels[i] == self.action_dictionary[j]:
                    labels[i] = j
                    
        self.Labels = labels  
        
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.termination_space = 2
        self.zeta = 0
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        
        # pi_hi net init
        pi_hi = NN_PI_HI(self.option_space, self.size_input)
        if NN_init == 'pre-train':
            pi_hi_model = pi_hi.PreTraining(self.TrainingSet_permanent)
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
        self.phi_h_input_dim = 3*self.option_space+2*self.termination_space+self.size_input+self.action_space
        seed = 0
        self.phi_h = NN_model(self.phi_h_input_dim,1, seed)
        self.Buffer = ReplayBuffer(10*3000, self.size_input, self.action_space, seed)

            
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
        
    
    def Loss(self, states, actions_encoded, t):
# =============================================================================
#         compute Loss function to minimize
# =============================================================================
        batch_size = len(states)
        loss = 0
        loss_pi_hi = 0
        loss_pi_b = 0
        loss_pi_lo = 0
        
        
        for oT in range(self.option_space):
            for bT in range(self.termination_space):
                for ot_past in range(self.option_space):
                    for ot in range(self.option_space):
                        o_past = np.zeros((1,self.option_space))
                        o = np.zeros((1,self.option_space))
                        o_past[0,ot_past] = 1 
                        o[0,ot] = 1 
                        b_T = np.zeros((1,self.termination_space))
                        o_T = np.zeros((1,self.option_space))  
                        o_T[0,oT]=1
                        b_T[0,bT] = 1   
                        input_phi_h = np.concatenate((np.ones((batch_size,1))*o_past, np.ones((batch_size,1))*np.array([[0,1]]), np.ones((batch_size,1))*o, states, actions_encoded, np.ones((batch_size,1))*b_T, np.ones((batch_size,1))*o_T),1)
                        loss_pi_hi = loss_pi_hi - kb.sum(self.phi_h(input_phi_h)*kb.log(kb.clip(self.NN_options(states,training=True)[:,ot],1e-10,1.0)))
                        for bt in range(self.termination_space):
                            b = np.zeros((1,self.termination_space))
                            b[0,bt] = 1 
                            input_phi_h = np.concatenate((np.ones((batch_size,1))*o_past, np.ones((batch_size,1))*b, np.ones((batch_size,1))*o, states, actions_encoded, np.ones((batch_size,1))*b_T, np.ones((batch_size,1))*o_T),1)
                            loss_pi_lo = loss_pi_lo - kb.sum(self.phi_h(input_phi_h)*kb.sum(actions_encoded*kb.log(kb.clip(self.NN_actions[ot](states,training=True),1e-10,1.0)),1))
                            loss_pi_b = loss_pi_b - kb.sum(self.phi_h(input_phi_h)*kb.log(kb.clip(self.NN_termination[ot_past](states,training=True)[:,bt],1e-10,1.0)))
                                    
        loss = loss_pi_hi + loss_pi_lo + loss_pi_b
        
        
        return loss

    def OptimizeLoss(self, states, actions, t):
# =============================================================================
#         minimize Loss all toghether
# =============================================================================
        weights = []
        loss = 0
        T = len(self.TrainingSet_permanent)
        if t+1 == T:
            M_step_epochs = self.epochs
        else:
            M_step_epochs = self.epochs
                
        for epoch in range(M_step_epochs):
            # print('\nStart m-step for sample ', t,' iteration ', epoch+1)
        
            with tf.GradientTape() as tape:
                for i in range(self.option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = OnlineHIL.Loss(self, states, actions, t)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*self.option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            # print('options loss:', float(loss))
        
        return loss        
    
        
    def Online_Baum_Welch_together(self, T_min, batch_size):

        P_option_given_obs = np.zeros((self.option_space, 1))
        P_option_given_obs = self.mu.reshape((self.option_space, 1)) 
                                        
        for t in range(0,len(self.TrainingSet_permanent)):
            if t==0:
                eta=1
            else:
                eta=1/(t+1) 
                
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet_permanent))
                
            current_state = self.TrainingSet_permanent[t,:].reshape(1,self.size_input)
            current_action = self.Labels[t]
            current_action_encoded = np.zeros((1,self.action_space))
            current_action_encoded[0,int(current_action)] = 1
            
            self.Buffer.store_transition(current_state, current_action, current_action_encoded)
            
            if self.Buffer.mem_cntr>batch_size:
                
                states, actions, actions_encoded = self.Buffer.sample_buffer(batch_size)
                                
                #E-step
                zi_temp1 = np.ones((self.option_space, self.termination_space, self.option_space, 1))
                norm = np.zeros((len(self.mu)))
                P_option_given_obs_temp = np.zeros((self.option_space, 1))
    
            
                for ot_past in range(self.option_space):
                    for bt in range(self.termination_space):
                        for ot in range(self.option_space):
                            zi_temp1[ot_past,bt,ot,0] = OnlineHIL.Pi_combined(ot, ot_past, current_action, bt, self.NN_options, 
                                                                              self.NN_actions[ot],  self.NN_termination[ot_past], current_state, self.zeta, 
                                                                              self.option_space)
                    
                    norm[ot_past] = P_option_given_obs[ot_past,0]*np.sum(zi_temp1[:,:,:,0],(1,2))[ot_past]
        
                zi_temp1[:,:,:,0] = np.divide(zi_temp1[:,:,:,0],np.sum(norm[:]))
                P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,0],1))*P_option_given_obs[:,0]),0) 
                
                zi = zi_temp1
                
                input_phi_h_final = np.empty((0,self.phi_h_input_dim))
                output_phi_h_final = np.empty((0,1))
                for sample in range(batch_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for bT in range(self.termination_space):
                                    for oT in range(self.option_space):
                                        o_past = np.zeros((1,self.option_space))
                                        b = np.zeros((1,self.termination_space))
                                        o = np.zeros((1,self.option_space))
                                        o_past[0,ot_past] = 1 
                                        b[0,bt] = 1 
                                        o[0,ot] = 1 
                                        prod_term = 0
                                        for oTminus1 in range(self.option_space):
                                            partial_prod_term = 0
                                            for bTminus1 in range(self.termination_space):
                                                bT_minus1 = np.zeros((1,self.termination_space))
                                                oT_minus1 = np.zeros((1,self.option_space))  
                                                oT_minus1[0,oTminus1]=1
                                                bT_minus1[0,bTminus1] = 1
                                                input_phi_h = np.concatenate((o_past, b, o, states[sample,:].reshape(1,self.size_input), actions_encoded[sample,:].reshape(1,self.action_space), bT_minus1, oT_minus1),1)
                                                partial_prod_term = partial_prod_term + max(0,self.phi_h(input_phi_h))
                                                
                                            prod_term = prod_term + zi[oTminus1,bT,oT,0]*partial_prod_term
                                         
                                        b_T = np.zeros((1,self.termination_space))
                                        o_T = np.zeros((1,self.option_space))  
                                        o_T[0,oT]=1
                                        b_T[0,bT] = 1   
                                        input_phi_h_final = np.append(input_phi_h_final,np.concatenate((o_past, b, o, states[sample,:].reshape(1,self.size_input), actions_encoded[sample,:].reshape(1,self.action_space), b_T, o_T),1),0)
                                        if actions[sample] == int(current_action) and np.sum(states[sample,:] == current_state)==self.size_input:
                                            output_phi_h_final = np.append(output_phi_h_final,(eta)*zi[ot_past,bt,ot,0]*P_option_given_obs[ot_past,0] + (1-eta)*prod_term, 0)
                                        else:
                                            output_phi_h_final = np.append(output_phi_h_final,(1-eta)*prod_term, 0)
                 
                                        
                P_option_given_obs = P_option_given_obs_temp
                self.phi_h.fit(input_phi_h_final, output_phi_h_final, epochs=1, verbose = 0)
                
                #M-step 
                if t > T_min and np.mod(t,100)==0:
                    loss = OnlineHIL.OptimizeLoss(self, states, actions_encoded, t)
    
        print('Maximization done, Total Loss:',float(loss))
           
        return self.NN_options, self.NN_actions, self.NN_termination
                

                                        
                                        
                                        
                                        
                                        
                                        