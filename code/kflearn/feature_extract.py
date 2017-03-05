# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:16:54 2017

@author: Thomas PESNEAU
"""
#import os
#path = "C:\\Users\\Thomas\\Desktop\\MVA 2016-2017\\2eme semestre\\Kernel methods for Machine learning\\Project\\kernel_challenge\\code\\unsupervised_feature_learning\\python"
#os.chdir(path)
import pandas as pd
import numpy as np
import kflearn.tools as tools
from sklearn.cluster import KMeans
#%% Define the parameters
#rfSize = 6
#nb_patches = 100000
#nb_centroids = 100
#whitening = True
#dim = [32,32,3]
#stride = 1
#eps = 10
#
##%% Load data
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

#%% Extract features of the dataset
############ Extract random patches from unlabeled images
#patches = tools.extract_random_patches(X_test,nb_patches,rfSize,dim)
#
#patches = tools.pre_process(patches,eps)
#
#if(whitening):
#    patches,M,P = tools.whiten(patches,eps_zca=0.1)
##%%    
#centroids = tools.Kmeans(patches,nb_centroids,nb_iter=50)
##%%
#X_feat = tools.extract_features(X_train,centroids,rfSize,dim,stride,eps,M,P)
#
##%%
#X_feat = tools.standard(X_feat)

#%% Export features
#np.savetxt('X_unsupervised_features.csv',X_feat,delimiter=',')
#%% 
def FeatureLearning(X_train,X_test,rfSize,nb_patches,nb_centroids,nb_iter,whitening,dim,stride,eps,eps_zca):
    """
    Feature learning with K-means and feature extraction
    ----------------
    Parameters:
        X_train: numpy array nb_samples x nb_elements, training dataset
        X_test: numpy array, testing dataset
        rfSize: int, size of the patches
        nb_patches: int, number of patches to extract
        nb_centroids: int, number of centroids used in K-means
        nb_iter: int, number of iterations of K-means
        whitening: Boolean
        dim: list, dimension of the images in the dataset
        stride: int, step between each patches in feature extraction
        eps: float, normalization parameter for pre-processing
        eps_zca: float, normalization parameter for ZCA whitening
    Output:
        X_feature: numpy array nb_samples x nb_features
    """
    ## Rescale the data
    X_train = X_train + abs(np.min(X_train))
    X_train = X_train * (255 / np.max(X_train))
    X_train = np.round(X_train)
    patches = tools.extract_random_patches(X_train,nb_patches,rfSize,dim)
    patches = tools.pre_process(patches,eps)
    if(whitening):
        patches,M,P = tools.whiten(patches,eps_zca)
    #km = KMeans(n_clusters=nb_centroids).fit(patches)
    #centroids = km.cluster_centers_
    centroids = tools.Kmeans(patches,nb_centroids,nb_iter)
    if(whitening):
        X_feature = tools.extract_features(X_train,centroids,rfSize,dim,stride,eps,M,P)
    else:
        X_feature = tools.extract_features(X_train,centroids,rfSize,dim,stride,eps)
    X_feature = tools.standard(X_feature)
    return X_feature

#%% test
#nb_iter = 50
#eps_zca = 0.1
#X_feature = FeatureLearning(X_train,X_test,rfSize,nb_patches,nb_centroids,nb_iter,whitening,dim,stride,eps,eps_zca)





