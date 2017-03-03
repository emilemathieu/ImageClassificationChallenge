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
nb_patches = 24000
nb_centroids = 96
whitening = True
dim = [32,32,3]
stride = 1
eps = 10

#%% Load data
######## Load labeled set
#X_train = pd.read_csv('../../../data/Xtr.csv',header=None)
#X_train = X_train.as_matrix()
#X_train = X_train[:, 0:-1]
### Rescale X_train
#X_train = X_train + abs(np.min(X_train))
#X_train = X_train * (255 / np.max(X_train))
#X_train = np.round(X_train)
######## Load unlabeled set
#X_test = pd.read_csv('../../../data/Xtr.csv',header=None)
#X_test = X_test.as_matrix()
#X_test = X_test[:, 0:-1]
### Rescale X_train
#X_test = X_test + abs(np.min(X_test))
#X_test = X_test * (255 / np.max(X_test))
#X_test = np.round(X_test)
#
#Y = pd.read_csv('../../../data/Ytr.csv',header=0)
#Y = Y.as_matrix()
#Y = Y[:,1]
#
##%% Extract features of the dataset
############# Extract random patches from unlabeled images
#patches = tools.extract_random_patches(X_test,nb_patches,rfSize,dim)
#%%
#patches = pd.read_csv('extractpatches.csv',header=None).as_matrix()
#patches =np.array(patches)
#patches = patches.astype(float)
##%%
#patches = tools.pre_process(patches,eps)
##%%
#if(whitening):
#    patches,M,P = tools.whiten(patches,eps_zca=0.1)
##%%    
#centroids = pd.read_csv('centroids.csv',header=None).as_matrix()
#centroids = np.array(centroids)
#centroids = centroids.astype(float)
##%%
#centroids = tools.Kmeans(patches,nb_centroids,nb_iter=50,centroids=centroids)
##%%
X_train = pd.read_csv('../../../data/Xtr.csv',header=None)
X_train = X_train.as_matrix()
X_train = X_train[:, 0:-1]
### Rescale X_train
X_train = X_train + abs(np.min(X_train))
X_train = X_train * (255 / np.max(X_train))
X_train = np.round(X_train)
centroids = pd.read_csv('centroids_learn.csv',header=None).as_matrix()
centroids = np.array(centroids)
centroids = centroids.astype(float)
M = pd.read_csv('M.csv',header=None).as_matrix()
M = np.array(M)
M = M.astype(float)
M = M.transpose()
P = pd.read_csv('P.csv',header=None).as_matrix()
P = np.array(P)
P = P.astype(float)
#%%
X_feat = tools.extract_features(X_train,centroids,rfSize,dim,stride,eps,M,P)

#%%
X_feat_s = tools.standard(X_feat)

#%% Export features
#np.savetxt('X_unsupervised_features.csv',X_feat,delimiter=',')






