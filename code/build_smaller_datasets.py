#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:23:35 2017

@author: EmileMathieu
"""

import numpy as np
import pandas as pd

#%% Transform to greyscale function

def rgb_to_greyscale(dataset):
    #Y = 0.299 R + 0.587 G + 0.114 B 
    columns_size = dataset.shape[1]
    if (columns_size % 3 != 0):
        raise ValueError('rgb_to_greyscale: column dimension must be a multiple of 3')
    channel_size = int(columns_size / 3)
    
    r = dataset[:,0:channel_size]
    g = dataset[:,channel_size:2*channel_size]
    b = dataset[:,2*channel_size:3*channel_size]
        
    return 0.299*r + 0.587*g + 0.114*b

#%% Load base dataset

X_full = pd.read_csv('../data/Xtr.csv', header=None)
X_full = X_full.as_matrix()
X_full = X_full[:, 0:-1]
X_full_grey = rgb_to_greyscale(X_full)

Y_full = pd.read_csv('../data/Ytr.csv')
Y_full = Y_full.as_matrix()
Y_full = Y_full[:,1]

#%% Transform to binary classification problem

class_0_indices = Y_full == 0
class_1_indices = Y_full == 1
class_0_1_indices = class_0_indices + class_1_indices
del class_0_indices,class_1_indices 

X_bin = X_full_grey[class_0_1_indices, :]
Y_bin = Y_full[class_0_1_indices]
Y_bin[Y_bin== 0] = -1
del class_0_1_indices

pd.DataFrame(Y_bin).to_csv('../data/Y_bin_01.csv',header=False, index=False)
pd.DataFrame(X_bin).to_csv('../data/X_bin_01.csv',header=False, index=False)

#%% Subsample multiclass problem: keep 100 samples (instead of 500) per class