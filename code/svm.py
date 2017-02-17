#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:33:30 2017

@author: EmileMathieu
"""

#%%

import cvxopt
import numpy as np
#from functools import lru_cache
#from collections import Counter
import itertools
from scipy.stats import mode

MIN_SUPPORT_VECTOR_VALUE = 1e-3

def linear_kernel():
        def f(x, y):
            return np.inner(x, y)
        return f

def rbf_kernel(sigma):
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

def quadratic_kernel():
    def f(x,y):
        return np.inner(x,y)**2
    return f

class Base_binary_classification(object):
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
        lagrange_multipliers = self._compute_weights(X, y)
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_VALUE
        
        self._weights = lagrange_multipliers[support_vector_indices]
        self._support_vectors = X[support_vector_indices]
        self._support_vector_labels = y[support_vector_indices]
        
        bias = np.mean(self._support_vector_labels - self.predict(self._support_vectors))
        self._bias = bias

    def _compute_weights(self, X, y):
        raise NotImplementedError()
        
#    def _compute_kernel_support_vectors(self, X):
#        res = np.zeros((X.shape[0], self._support_vectors.shape[0]))
#        for i,x_i in enumerate(X):
#            for j,x_j in enumerate(self._support_vectors):
#                res[i, j] = self._kernel(x_i, x_j)
#        return res

    def predict(self, X):
##        n_samples = X.shape[0]
##        prediction = np.zeros(n_samples)
##        bias = self._bias
##        weights_times_labels = np.matrix(self._weights * self._support_vector_labels)
##        res1 = self._compute_kernel_support_vectors(X)
##        prod = np.multiply(res1,weights_times_labels)
##        prediction=bias+np.sum(prod,1)
##        return np.squeeze(np.asarray(prediction))
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
    
    def predict_value(self, X):
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i, x in enumerate(X):
            result = self._bias
            for z_i, x_i, y_i in zip(self._weights,
                                     self._support_vectors,
                                     self._support_vector_labels):
                result += z_i * y_i * self._kernel(x_i, x)
            prediction[i] = result
        return prediction

    def score(self, X, y):
        """
        Computes average score for the given features X and labels y.
        """
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class binary_classification_smo(Base_binary_classification):
    def __init__(self, kernel, C=1.0, epsilon=1e-3, max_iteration=1000):
        super().__init__(kernel, C=1.0)
        self._epsilon = epsilon
        self._max_iteration = max_iteration
        
        #@lru_cache()
    def _compute_kernel_matrix_row(self, X, index):
        if index in self._cache_kernels:
            return self._cache_kernels[index]
        n_samples, n_features = X.shape
        row = np.zeros(n_samples)
        x_i = X[index, :]
        for j,x_j in enumerate(X):
            row[j] = self._kernel(x_i, x_j)
        if (self._len_cache_kernels < 100):
            self._cache_kernels[index] = row
        return row
   
    def _compute_kernel_matrix_diag(self, X):
        n_samples, n_features = X.shape
        diag = np.zeros(n_samples)
        for j,x_j in enumerate(X):
            diag[j] = self._kernel(x_j, x_j)
        return diag
    
    def _reset_cache_kernels(self):
        self._cache_kernels = {}
        self._len_cache_kernels = 0
        
    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        stop_criterion = True
        alpha = np.zeros(n_samples)
        g = np.ones(n_samples)
        diag = self._compute_kernel_matrix_diag(X)
        self._reset_cache_kernels()
        while stop_criterion:
            #print(iteration)
            yg = g * y
            indices_y_pos = (y == 1)
            indices_y_neg = (np.ones(n_samples) - indices_y_pos).astype(bool)#(y == -1)
            indices_alpha_big = (alpha >= self._C)
            indices_alpha_neg = (alpha <= 0)
            
            indices_violate_Bi_1 = indices_y_pos * indices_alpha_big
            indices_violate_Bi_2 = indices_y_neg * indices_alpha_neg
            yg_i = yg.copy()
            yg_i[indices_violate_Bi_1 + indices_violate_Bi_2] = float('-inf')
            
            indices_violate_Ai_1 = indices_y_pos * indices_alpha_neg
            indices_violate_Ai_2 = indices_y_neg * indices_alpha_big
            yg_j = yg.copy()
            yg_j[indices_violate_Ai_1 + indices_violate_Ai_2] = float('+inf')
            
            i = np.argmax(yg_i)
            Ki = self._compute_kernel_matrix_row(X, i)
            Kii = Ki[i]
            diff = yg_i[i] - yg
            indices_violate_criterion = diff <= 0
            vec_j = (diff)**2 / (Kii - diag - 2*Ki)
            vec_j[indices_violate_Ai_1+indices_violate_Ai_2+indices_violate_criterion] = float('-inf')
            
            #j = np.argmin(yg_j)
            j = np.argmax(vec_j)
            Kj = self._compute_kernel_matrix_row(X, j)

            stop_criterion = not (yg_i[i] - yg_j[j] < self._epsilon)
            if (not stop_criterion) or iteration >= self._max_iteration:
                break
            
            #compute lambda
            min_1 = (y[i]==1)*self._C -y[i] * alpha[i]
            min_2 = y[j] * alpha[j] + (y[j]==-1)*self._C
            min_3 = (yg_i[i] - yg_j[j])/(Kii + Kj[j] - 2*Ki[j])
            lambda_param = np.min([min_1, min_2, min_3])
            
            #update g
            g = g + lambda_param * y * (Kj - Ki)
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1
            #list_kernel_called.append(i)
            #list_kernel_called.append(j)
        print(iteration)
        self._reset_cache_kernels()
        #print(Counter(list_kernel_called))
        #print(self._compute_kernel_matrix_row.cache_info())
        return alpha
    
class binary_classification_qp(Base_binary_classification):
    def __init__(self, kernel, C=1.0):
        super().__init__(kernel, C=1.0)
        
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

class Base_multiclass(object):
    def __init__(self, kernel, C=1.0, algo='qp'):
        self._algo = algo
        self._kernel = kernel
        self._C = C
        self._classifiers = []
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()
        
    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class multiclass_ovo(Base_multiclass):
    def __init__(self, kernel, C=1.0, algo='qp'):
        super().__init__(kernel, C=1.0, algo='qp')

    def fit(self, X, y):
        self._classes = set(y)
        self._pairs = list(itertools.combinations(self._classes, 2))
        self._nb_pairs = len(self._pairs)
        classifiers = np.empty(self._nb_pairs, dtype=object)
        for i,pair_of_labels in enumerate(self._pairs):
            print(pair_of_labels)
            if self._algo == 'smo':
                SVM_binary = binary_classification_smo(kernel=self._kernel, C=1.0)
            else:
                SVM_binary = binary_classification_qp(kernel=self._kernel, C=1.0)
            X_filtered, y_filtered = self._filter_dataset_by_labels(X, y, pair_of_labels)
            SVM_binary.fit(X_filtered, y_filtered)
            classifiers[i] = SVM_binary
        self._classifiers = classifiers
    
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
        #predicted_labels = predicted_labels[:,np.random.permutation(self._nb_pairs)]
        #print(predicted_labels)
        prediction = np.ravel(mode(predicted_labels, axis=1).mode)
        print(prediction)
        return prediction

class multiclass_ova(Base_multiclass):
    def __init__(self, kernel, C=1.0, algo='qp'):
        super().__init__(kernel, C=1.0, algo='qp')

    def fit(self, X, y):
        self._classes = set(y)
        classifiers = np.empty(len(self._classes), dtype=object)
        for i, label in enumerate(self._classes):
            print(label)
            if self._algo == 'smo':
                SVM_binary = binary_classification_smo(kernel=self._kernel, C=1.0)
            else:
                SVM_binary = binary_classification_qp(kernel=self._kernel, C=1.0)
            X_filtered, y_filtered = self._filter_dataset_by_labels(X, y, label)
            SVM_binary.fit(X_filtered, y_filtered)
            classifiers[i] = SVM_binary
        self._classifiers = classifiers

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
            predicted_scores[:,i] = self._classifiers[i].predict_value(X)
        return np.argmax(predicted_scores, axis=1)