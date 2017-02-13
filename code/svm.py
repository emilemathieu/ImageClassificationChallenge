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

def linear_kernel():
        def f(x, y):
            return np.inner(x, y)
        return f

def rbf_kernel(sigma):
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f


class binary_classification(object):
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
        #print(self._weights)
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
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(sol['x'])

    def predict(self, X, need_real_value=False):
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
            if need_real_value:
                prediction[i] = result
            else:
                prediction[i] = np.sign(result).item()
        return prediction

    def score(self, X, y):
        """
        Computes average score for the given features X and labels y.
        """
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)
    

import itertools
from scipy.stats import mode

class multiclass_1vs1(object):
    def __init__(self, kernel, C=1.0):
        self._kernel = kernel
        self._C = C
        self._classifiers = []

    def fit(self, X, y):
        self._classes = set(y)
        self._pairs = list(itertools.combinations(self._classes, 2))
        self._nb_pairs = len(self._pairs)

        for pair_of_labels in self._pairs:
            print(pair_of_labels)
            SVM_binary = binary_classification(kernel=self._kernel, C=1.0)
            X_filtered, y_filtered = self._filter_dataset_by_labels(X, y, pair_of_labels)
            SVM_binary.fit(X_filtered, y_filtered)
            self._classifiers.append(SVM_binary)
    
    def _filter_dataset_by_labels(self, X, y, pair_of_labels):
        label_1, label_2 = pair_of_labels
        class_neg_indices = (y == label_1)
        class_pos_indices = (y == label_2)
        classes_indices = class_neg_indices + class_pos_indices

        X_filtered = X[classes_indices, :]
        y_filtered = y[classes_indices]
        y_filtered[y_filtered == label_1] = -1
        y_filtered[y_filtered == label_2] = 1

        return X_filtered, y_filtered

    def predict(self, X):
        n_samples = X.shape[0]
        predicted_labels = np.zeros((n_samples, self._nb_pairs))
        for j, pair_of_labels in enumerate(self._pairs):
            print(pair_of_labels)
            binary_prediction = self._classifiers[j].predict(X)
            binary_prediction[binary_prediction == -1] = pair_of_labels[0]
            binary_prediction[binary_prediction == 1] = pair_of_labels[1]
            predicted_labels[:,j] = binary_prediction
        #print(predicted_labels)
        return np.ravel(mode(predicted_labels, axis=1).mode)

    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class multiclass_1vsall(object):
    def __init__(self, kernel, C=1.0):
        self._kernel = kernel
        self._C = C
        self._classifiers = []

    def fit(self, X, y):
        self._classes = set(y)

        for label in self._classes:
            print(label)
            SVM_binary = binary_classification(kernel=self._kernel, C=1.0)
            X_filtered, y_filtered = self._filter_dataset_by_labels(X, y, label)
            SVM_binary.fit(X_filtered, y_filtered)
            self._classifiers.append(SVM_binary)

    def _filter_dataset_by_labels(self, X, y, label):
        class_pos_indices = (y == label)
        class_neg_indices = (y != label)
        X_filtered = X.copy()
        y_filtered = y.copy()
        y_filtered[class_pos_indices] = 1
        y_filtered[class_neg_indices] = -1
        return X_filtered, y_filtered

    def predict(self, X):
        n_samples = X.shape[0]
        predicted_scores = np.zeros((n_samples, len(self._classes)))
        for i, label in enumerate(self._classes):
            print(i)
            predicted_scores[:,i] = self._classifiers[i].predict(X, True)
        return np.argmax(predicted_scores, axis=1)

    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)