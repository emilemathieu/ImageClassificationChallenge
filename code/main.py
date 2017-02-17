#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime

#%% Transform to greyscale function

def rgb_to_greyscale(dataset):
    columns_size = dataset.shape[1]
    if (columns_size % 3 != 0):
        raise ValueError('rgb_to_greyscale: column dimension must be a multiple of 3')
    channel_size = int(columns_size / 3)
    
    r = dataset[:,0:channel_size]
    g = dataset[:,channel_size:2*channel_size]
    b = dataset[:,2*channel_size:3*channel_size]
    return 0.299*r + 0.587*g + 0.114*b

#%% Load dataset

IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

X_full = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
#X_cnn_features = pd.read_csv('../data/Xtr_features_cnn.csv', header=None).as_matrix()

Y_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]

#%%
#X_multi = X_cnn_features
X_multi = rgb_to_greyscale(X_full)
Y_multi = Y_full
N = len(Y_multi)
#%% Select classifiers

from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
import svm
from importlib import reload

kernel = svm.quadratic_kernel()

classifiers = {
               #'OVO-QP': svm.multiclass_1vs1(kernel=kernel),
               #'OVO-SMO quad': svm.multiclass_ovo(kernel=svm.quadratic_kernel()),
               #'OVA-SMO lin': svm.multiclass_ova(kernel=svm.linear_kernel(), algo='smo'),
               'sklearn lin 1': SVC(kernel='linear', C=1.0)
               }

#%% Assess classifiers

import timeit
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold

scores = {classifier_name: [] for classifier_name, classifier in classifiers.items()}
times = {classifier_name: 0 for classifier_name, classifier in classifiers.items()}

X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.33)

for classifier_name, classifier in classifiers.items():
    print("%s - fit" % classifier_name)
    start = timeit.default_timer()
    classifier.fit(X_train, y_train)
    print("%s - predict\n" % classifier_name)
    scores[classifier_name].append(classifier.score(X_test, y_test))
    times[classifier_name] = round(timeit.default_timer() - start, 2)

for classifier_name in classifiers:    
    print("\n%s -- score:%s | time:%s" % 
          (classifier_name, np.mean(scores[classifier_name]), times[classifier_name]))
