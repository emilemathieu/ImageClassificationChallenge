# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:16:54 2017

@author: Thomas PESNEAU
"""
import os
path = "C:\\Users\\Thomas\\Desktop\\MVA 2016-2017\\2eme semestre\\Kernel methods for Machine learning\\Project\\kernel_challenge\\code\\unsupervised_feature_learning\\python"
os.chdir(path)
import pandas as pd
import numpy as np
import tools
#%% Define the parameters
rfSize = 6
nb_patches = 100000
nb_centroids = 1500
whitening = True
dim = [32,32,3]

#%% Load data
####### Load labeled set
X_train = pd.read_csv('../../../data/Xtr.csv',header=None)
X_train = X_train.as_matrix()
X_train = X_train[:, 0:-1]
## Rescale X_train
X_train = X_train + abs(np.min(X_train))
X_train = X_train * (255 / np.max(X_train))
X_train = np.round(X_train)
####### Load unlabeled set
X_test = pd.read_csv('../../../data/Xtr.csv',header=None)
X_test = X_test.as_matrix()
X_test = X_test[:, 0:-1]
## Rescale X_train
X_test = X_test + abs(np.min(X_test))
X_test = X_test * (255 / np.max(X_test))
X_test = np.round(X_test)

Y = pd.read_csv('../../../data/Ytr.csv',header=0)
Y = Y.as_matrix()
Y = Y[:,1]

#%% Extract features of the dataset
############ Extract random patches from unlabeled images
patches = tools.extract_random_patches(X_test,nb_patches,rfSize,dim)

patches = tools.pre_process(patches,eps=10)

if(whitening):
    patches,M,P = tools.whiten(patches,eps_zca=0.1)
#%%    
centroids = tools.Kmeans(patches,nb_centroids,nb_iter=50)
#%%
X_feat = tools.extract_features(X_train,centroids,rfSize,dim,M,P)
#%%
X_feat = tools.standard(X_feat)

#%% Export features
np.savetxt('X_unsupervised_features.csv',X_feat,delimiter=',')






