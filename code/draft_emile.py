#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

X_full = pd.read_csv('../data/Xtr.csv', header=None)
X_full = X_full.as_matrix()
X_full = X_full[:, 0:-1]

Y_full = pd.read_csv('../data/Ytr.csv')
Y_full = Y_full.as_matrix()
Y_full = Y_full[:,1]
#%% Transform to binary classification problem

class_0_indices = Y_full == 0
class_1_indices = Y_full == 1
class_0_1_indices = class_0_indices + class_1_indices
del class_0_indices,class_1_indices 

X = X_full[class_0_1_indices, :]
Y = Y_full[class_0_1_indices]
Y[Y== 0] = -1
del class_0_1_indices

#%% Transform to greyscale

def rgb_to_greyscale(dataset):
    #Y = 0.299 R + 0.587 G + 0.114 B 
    columns_size = dataset.shape[1]
    if (columns_size % 3 != 0):
        raise ValueError('rgb_to_greyscale: column dimension must be a multiple of 3')
    channel_size = columns_size / 3
    
    r = dataset[:,0:channel_size]
    g = dataset[:,channel_size:2*channel_size]
    b = dataset[:,2*channel_size:3*channel_size]
        
    return 0.299*r + 0.587*g + 0.114*b

X = rgb_to_greyscale(X)

import matplotlib.pyplot as plt
plt.imshow(X[0,:].reshape((32,32)), cmap='gray')

#%% Simple dataset

#from sklearn import datasets
#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

#%% Learn and test classifiers

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold

N = len(Y)
nb_splits = 10

svc_clf = SVC()
dummy_clf = DummyClassifier()

scores_SCV = np.zeros(nb_splits)
scores_dummy = np.zeros(nb_splits)

kf = KFold(n_splits=nb_splits, shuffle=True)

for i, (train, test) in enumerate(kf.split(range(N))):
    print(i)
    #print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    
    svc_clf.fit(X_train, y_train)
    scores_SCV[i] = svc_clf.score(X_test, y_test)
    
    dummy_clf.fit(X_train, y_train)
    scores_dummy[i] = dummy_clf.score(X_test, y_test)
    
print("Score SVC: %s" % scores_SCV.mean())
print("Score dummy: %s" %scores_dummy.mean())

#%% Submission

clf = dummy_clf

X_e = pd.read_csv('../data/Xte.csv', header=None)
X_e = X_e.as_matrix()
X_e = X_e[:, 0:-1]
X_e = rgb_to_greyscale(X_e)

prediction = dummy_clf.predict(X_e)
prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction.to_csv('../data/evaluation.csv', sep=',', header=True, index=False)