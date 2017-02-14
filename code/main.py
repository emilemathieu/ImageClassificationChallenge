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

#%% Keep multiclass problem
X_multi = rgb_to_greyscale(X_full)
Y_multi = Y_full

#%% Learn and test classifiers

from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from importlib import reload
import svm

N = len(Y_multi)
nb_splits = 2

#svc_clf = SVC(kernel='poly')
svc_clf = LinearSVC()
dummy_clf = DummyClassifier()
clf_multi_1vs1 = svm.multiclass_1vs1(svm.linear_kernel())
#clf_multi_1vsall = svm.multiclass_1vsall(svm.linear_kernel())
clf_multi_1vs1_smo = svm.multiclass_1vs1(svm.linear_kernel(), algo='smo')

scores_A = np.zeros(nb_splits)
scores_B = np.zeros(nb_splits)
scores_C = np.zeros(nb_splits)
scores_D = np.zeros(nb_splits)

kf = KFold(n_splits=nb_splits, shuffle=True)

for i, (train, test) in enumerate(kf.split(range(N))):
    print(i)
    X_train, X_test, y_train, y_test = X_multi[train], X_multi[test], Y_multi[train], Y_multi[test]
    
    print("A - fit")
    clf_multi_1vs1_smo.fit(X_train, y_train)
    print("A - predict")
    scores_A[i] = clf_multi_1vs1_smo.score(X_test, y_test)
    print(scores_A[i])
#    print("B - fit")
#    clf_multi_1vs1.fit(X_train, y_train)
#    print("B - predict")
#    scores_B[i] = clf_multi_1vs1.score(X_test, y_test)

    print("C - fit")
    svc_clf.fit(X_train, y_train)
    print("C - predict")
    scores_C[i] = svc_clf.score(X_test, y_test)
    print(scores_C[i])
    
    #print("D - fit")
    #dummy_clf.fit(X_train, y_train)
    #print("D - predict")
    #scores_D[i] = dummy_clf.score(X_test, y_test)
    
print("Score A: %s" % scores_A.mean())
#print("Score B: %s" %scores_B.mean())
print("Score C: %s" %scores_C.mean())
print("Score D: %s" %scores_D.mean())
#%% Submission

clf = svm.multiclass_1vs1(svm.linear_kernel())

clf.fit(X_multi, Y_multi)

X_e = pd.read_csv('../data/Xte.csv', header=None)
X_e = X_e.as_matrix()
X_e = X_e[:, 0:-1]
X_e = rgb_to_greyscale(X_e)

prediction = clf.predict(X_e)
prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction['Id'] = prediction['Id'] + 1
prediction.to_csv('../data/evaluation.csv', sep=',', header=True, index=False)