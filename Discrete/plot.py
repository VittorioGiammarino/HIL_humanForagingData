#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:36:58 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

with open('results/HSAC_HIL_True_Foraging_0.npy', 'rb') as f:
    HSAC_HIL_mean = np.load(f, allow_pickle=True)
    
    

    
# %%

steps = np.linspace(0,1e6,len(HSAC_HIL_mean))

fig, ax = plt.subplots()
# plt.xscale('log')
# plt.xticks(Samples, labels=['100', '200', '500', '1k', '2k'])
clrs = sns.color_palette("husl", 5)
ax.plot(steps, HSAC_HIL_mean, label='HSAC+HIL', c=clrs[0])
ax.legend(loc=0, facecolor = '#d8dcd6')
ax.set_ylim([0,300])
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Discrete Foraging')
