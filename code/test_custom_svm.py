#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:14:55 2017

@author: EmileMathieu
"""

if __name__ == '__main__':
    
    import svm
    
    #%% Load data
    import numpy as np
    from sklearn import datasets
    
    iris = datasets.load_iris()
    X = iris.data[0:100,:]
    y = iris.target[0:100]
    y[y == 0] = -1
    n = len(y)
    indices_suffle = np.random.permutation(n)
    X = X[indices_suffle, :]
    y = y[indices_suffle]
    
    X_train = X[0:50, :]
    y_train = y[0:50]
    X_test = X[50:100, :]
    y_test = y[50:100]
    
    
    #%%
    from sklearn.svm import SVC
    
    #kernel = svm.linear_kernel()
    kernel = svm.rbf_kernel(0.5)
    clf = svm.SVM(kernel=kernel)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
    clf_sktlrn = SVC(kernel='rbf')
    clf_sktlrn.fit(X_train, y_train)
    print(clf_sktlrn.score(X_test, y_test))
