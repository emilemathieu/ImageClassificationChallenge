#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:33:30 2017

@author: EmileMathieu
"""

#%%

import cvxopt
import numpy as np

MIN_SUPPORT_VECTOR_VALUE = 0.001

class SVM(object):
    """Linear Support Vector Classification.
    
        Parameters
        ----------
        C : float, optional (default=1.0)
            Penalty parameter C of the error term.
        
        kernel : callable, (R^d,R^d) -> R
            kernel function
    """

    def __init__(self, kernel, C=1.0):
        self._kernel = kernel
        self._C = C
        self._support_vectors = []
        self._support_vector_labels = []
        self._weights = []
        self._bias = 0.0          
            
    def fit(self, X, y):
        """
        Learn parameters of the model
        """
        lagrange_multipliers = self._compute_weights( X, y)
        
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_VALUE
            
        self._weights = lagrange_multipliers[support_vector_indices]
        print(self._weights)
        self._support_vectors = X[support_vector_indices]
        self._support_vector_labels = y[support_vector_indices]
        
        bias = np.mean(self._support_vector_labels - self.predict(self._support_vectors))
        self._bias = bias
        
    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K
    
    def _compute_weights(self, X, y):
        n_samples, n_features = X.shape
        self._n_samples = n_samples
        self._n_features = n_features
        
        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
        q = cvxopt.matrix(-1 * np.ones(n_samples), tc='d')
        G_1 = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_1 = cvxopt.matrix(np.zeros(n_samples))
        G_2 = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_2 = cvxopt.matrix(np.ones(n_samples) * self._C)
        G = cvxopt.matrix(np.vstack((G_1, G_2)), tc='d')
        h = cvxopt.matrix(np.vstack((h_1, h_2)), tc='d')
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(sol['x'])

    def predict(self, X):
        """
        Computes predictions for the given features X.
        """
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i, x in enumerate(X):
            result = self._bias
            for z_i, x_i, y_i in zip(self._weights,
                                     self._support_vectors,
                                     self._support_vector_labels):
                result += z_i * y_i * self._kernel(x_i, x)
            prediction[i] = np.sign(result).item()
        return prediction
    
    def score(self, X, y):
        """
        Computes average score for the given features X and labels y.
        """
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)
    
def linear_kernel():
        def f(x, y):
            return np.inner(x, y)
        return f
    
def rbf_kernel(sigma):
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f
