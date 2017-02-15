#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:14:55 2017

@author: EmileMathieu
"""

#%% Import libraries

import numpy as np
import pandas as pd

#%% Load binary dataset
X_bin = pd.read_csv('../data/X_bin_01.csv', header=None)
X_bin = X_bin.as_matrix()

Y_bin = pd.read_csv('../data/Y_bin_01.csv', header=None)
Y_bin = Y_bin.as_matrix()
Y_bin = Y_bin[:,0]

#%% Load small multiclass dataset
X_mul = pd.read_csv('../data/X_small_50.csv', header=None)
X_mul = X_mul.as_matrix()

Y_mul = pd.read_csv('../data/Y_small_50.csv', header=None)
Y_mul = Y_mul.as_matrix()
Y_mul = Y_mul[:,0]

#%% Select problem: binary or multiclass ?

#X = X_bin
#y = Y_bin
X = X_mul
y = Y_mul

#%% Select classifiers
from sklearn.svm import SVC, LinearSVC
import svm
from importlib import reload

#kernel = svm.rbf_kernel(0.5)
kernel = svm.linear_kernel()

classifiers = {'OVO': svm.multiclass_ovo(kernel=kernel, algo='smo'),
               'OVA': svm.multiclass_ova(kernel=kernel, algo='smo'),
               'klearn': SVC(kernel='linear')}

#classifiers = {'smo': svm.binary_classification_smo(kernel=kernel),
#               'qp': svm.binary_classification_qp(kernel=kernel),
#               'sklearn': SVC(kernel='linear')}

#%% Assess classifiers with KFold

import timeit
from sklearn.model_selection import KFold

N = len(y)
nb_splits = 2
kf = KFold(n_splits=nb_splits, shuffle=True)

scores = {classifier_name: np.zeros(nb_splits) for classifier_name, classifier in classifiers.items()}
times = {classifier_name: 0 for classifier_name, classifier in classifiers.items()}

for i, (train, test) in enumerate(kf.split(range(N))):
    print("KFold :%s/%s \n" % (i+1,nb_splits))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
    for classifier_name, classifier in classifiers.items():
        print("%s - fit" % classifier_name)
        start = timeit.default_timer()
        classifier.fit(X_train, y_train)
        print("%s - predict\n" % classifier_name)
        #classifier.predict(X_test)
        scores[classifier_name][i] = classifier.score(X_test, y_test)
        times[classifier_name] = round(timeit.default_timer() - start, 2)

for classifier_name in classifiers:
    print("\n%s -- score:%s | time:%s" % 
          (classifier_name, scores[classifier_name].mean(), times[classifier_name]))
