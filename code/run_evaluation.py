#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:50:55 2017

@author: EmileMathieu
"""
from mllib import svm
import pandas as pd
import datetime

#################################
###        CHOOSE ALGO        ###
#################################

clf = svm.multiclass_ovo(C=1000.0,kernel=svm.Kernel.rbf(gamma=1.0/50), tol=1.0)

#################################
###        ///////////        ###
#################################

#X_full = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
#X_multi = rgb_to_greyscale(X_full)
X_multi = pd.read_csv('../data/cnn/Xtr_features_mycnn_8.csv', header=None).as_matrix()
Y_multi = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]

N = len(Y_multi)

clf.fit(X_multi, Y_multi)

#X_e = pd.read_csv('../data/Xte.csv', header=None).as_matrix()[:, 0:-1]
#X_e = rgb_to_greyscale(X_e)
X_e = pd.read_csv('../data/cnn/Xte_features_mycnn_8.csv', header=None).as_matrix()


prediction = clf.predict(X_e)
prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction['Id'] = prediction['Id'] + 1
prediction['Prediction'] = prediction['Prediction'].astype(int)
prediction.to_csv('../data/evaluations/evaluation_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
                 sep=',', header=True, index=False)