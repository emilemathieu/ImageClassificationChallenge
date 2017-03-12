#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from importlib import reload

#%% Load dataset

IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

#X_full = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
#X_cnn_features = pd.read_csv('../data/Xtr_features_mycnn.csv', header=None).as_matrix()

#X_augmented = pd.read_csv('../data/augmented_X.csv',header=None).as_matrix()
#X_final = np.concatenate((X_full,X_augmented),axis=0)
X_train = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
X_test = pd.read_csv('../data/Xte.csv', header=None).as_matrix()[:, 0:-1]
#X_matlab_features = pd.read_csv('../data/X_features_kmeans.csv', header=None).as_matrix()

Y_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
#Y_augmented = pd.read_csv('../data/augmented_Y.csv').as_matrix()[:,1]
#Y_final = np.concatenate((Y_full, Y_augmented),axis=0)

#%% K-means feature learning
rfSize = 8
nb_patches = 40000
nb_centroids = 1000
nb_iter = 15
whitening = True
dim = [32,32,3]
stride = 1
eps = 10
eps_zca = 0.1
#%%
from mllib import tools
from mllib import kflearn
X_k = X_train
#X_k = np.concatenate((X_train,X_test),axis=0)
X_k = X_k + abs(np.min(X_k))
X_k = X_k * (255 / np.max(X_k))
X_k = np.round(X_k)
#%%
patches = tools.extract_random_patches(X_k,nb_patches,rfSize,dim)
#%%
#patches =  pd.read_csv('../data/patches_eval2.csv'.format(rfSize), header=None).as_matrix()
patches = patches[0:nb_patches,:]
#%% Patches pre processing
patches = tools.pre_process(patches,eps)
patches,M,P = tools.whiten(patches,eps_zca)
#%% run k means

centroids = kflearn.Kmeans(patches,nb_centroids,nb_iter)

#%%
#X_feat_2 = tools.extract_features(X_k[0:100],centroids,rfSize,dim,stride,eps,M,P)
#%%
X_feat = kflearn.extract_features(X_k,centroids,rfSize,dim,stride,eps,M,P)
X_feat_s = tools.standard(X_feat)

#%%
X_multi = X_feat_s
Y_multi = Y_full
N = len(Y_multi)

#%% Select classifiers

from sklearn.svm import SVC
from mllib import svm

classifiers = {
        'sklearn': SVC(kernel='linear', degree=2, C=10.),
        #'SMO OVO': svm.multiclass_ovo(C=10., kernel=svm.Kernel.linear(), tol=1.0, max_iter=5000),
               }

#%% Assess classifiers

import timeit
from sklearn.model_selection import train_test_split, KFold

scores_train = {classifier_name: [] for classifier_name, classifier in classifiers.items()}
scores_test = {classifier_name: [] for classifier_name, classifier in classifiers.items()}
times = {classifier_name: [] for classifier_name, classifier in classifiers.items()}

#X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.33)

for i, (train, test) in enumerate(KFold(n_splits=3, shuffle=True).split(range(len(Y_multi)))):
    X_train, X_test, y_train, y_test = X_multi[train], X_multi[test], Y_multi[train], Y_multi[test]

    for classifier_name, classifier in classifiers.items():
        print("%s - fit" % classifier_name)
        start = timeit.default_timer()
        classifier.fit(X_train, y_train)
        print("%s - predict\n" % classifier_name)
        scores_train[classifier_name].append(classifier.score(X_train, y_train))
        scores_test[classifier_name].append(classifier.score(X_test, y_test))
        times[classifier_name].append(timeit.default_timer() - start)

for classifier_name in classifiers:    
    print("\n%s -- score: training:%s & test:%s | time:%s" % 
          (classifier_name, np.mean(scores_train[classifier_name]), np.mean(scores_test[classifier_name]), round(np.mean(times[classifier_name]),2)))