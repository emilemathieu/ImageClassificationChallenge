#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:50:55 2017

@author: EmileMathieu
"""
import svm
import pandas as pd
import datetime

#################################
###        CHOOSE ALGO        ###
#################################

clf = svm.multiclass_ova(kernel=svm.quadratic_kernel(), algo='smo')

#################################
###        ///////////        ###
#################################

def rgb_to_greyscale(dataset):
    #Y = 0.299 R + 0.587 G + 0.114 B 
    columns_size = dataset.shape[1]
    if (columns_size % 3 != 0):
        raise ValueError('rgb_to_greyscale: column dimension must be a multiple of 3')
    channel_size = int(columns_size / 3)
    
    r = dataset[:,0:channel_size]
    g = dataset[:,channel_size:2*channel_size]
    b = dataset[:,2*channel_size:3*channel_size]
        
    return 0.299*r + 0.587*g + 0.114*b

X_full = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
Y_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]

X_multi = rgb_to_greyscale(X_full)
Y_multi = Y_full
N = len(Y_multi)


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
prediction['Prediction'] = prediction['Prediction'].astype(int)
prediction.to_csv('../data/evaluation_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
                 sep=',', header=True, index=False)