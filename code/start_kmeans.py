#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from mllib import svm, tools, kflearn

best_perf = len(sys.argv) > 1 and sys.argv[1] == 'best'

X_train = pd.read_csv('Xtr.csv', header=None).as_matrix()[:, 0:-1]
X_test = pd.read_csv('Xte.csv', header=None).as_matrix()[:, 0:-1]

#################################
###      FEATURE LEARNING     ###
#################################
rfSize = 8
if best_perf:
    print('You asked for best performance, warning it may take a while...')
    nb_patches = 400000
    nb_centroids = 4000
else:
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

Y_train = pd.read_csv('Ytr.csv').as_matrix()[:,1]

clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction['Id'] = prediction['Id'] + 1
prediction['Prediction'] = prediction['Prediction'].astype(int)

prediction.to_csv('Yte.csv',sep=',', header=True, index=False)