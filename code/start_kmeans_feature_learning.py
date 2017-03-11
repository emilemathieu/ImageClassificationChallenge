#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:50:55 2017

@author: EmileMathieu
"""
import pandas as pd
import numpy as np
from mllib import svm
from mllib import tools
from mllib import kflearn

X_train = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
X_test = pd.read_csv('../data/Xte.csv', header=None).as_matrix()[:, 0:-1]

#################################
###      FEATURE LEARNING     ###
#################################
rfSize = 8
nb_patches = 40000
nb_centroids = 1000
nb_iter = 15
whitening = True
dim = [32,32,3]
stride = 1
eps = 10
eps_zca = 0.1

X_train += abs(np.min(X_train))
X_train *= (255 / np.max(X_train))
X_train = np.round(X_train)
X_test += abs(np.min(X_test))
X_test *= (255 / np.max(X_test))
X_test = np.round(X_test)

X_k = np.concatenate((X_train,X_test),axis=0)

patches = kflearn.extract_random_patches(X_k,nb_patches,rfSize,dim)
patches = patches[0:nb_patches,:]
patches = kflearn.pre_process(patches,eps)
patches,M,P = tools.whiten(patches,eps_zca)

centroids = kflearn.Kmeans(patches,nb_centroids,nb_iter)

X_train = kflearn.extract_features(X_train,centroids,rfSize,dim,stride,eps,M,P)
X_train = tools.standard(X_train)

X_test = kflearn.extract_features(X_test,centroids,rfSize,dim,stride,eps,M,P)
X_test = tools.standard(X_test)

#################################
###      CLASSIFICATION       ###
#################################

clf = svm.multiclass_ovo(C=10., kernel=svm.Kernel.linear(), tol=1.0, max_iter=5000)

#################################
###        ///////////        ###
#################################

Y_train = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]

clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction['Id'] = prediction['Id'] + 1
prediction['Prediction'] = prediction['Prediction'].astype(int)

prediction.to_csv('../data/Yte.csv',sep=',', header=True, index=False)