#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:33:30 2017

@author: EmileMathieu
"""

import cvxopt
import numpy as np
import itertools
from scipy.stats import mode
from collections import OrderedDict
from sklearn.svm import SVC

def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking ovr decision function.
    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.
    Parameters
    ----------
    predictions : array-like, shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.
    confidences : array-like, shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.
    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    max_confidences = sum_of_confidences.max()
    min_confidences = sum_of_confidences.min()

    if max_confidences == min_confidences:
        return votes

    # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    eps = np.finfo(sum_of_confidences.dtype).eps
    max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
    scale = (0.5 - eps) / max_abs_confidence
    return votes + sum_of_confidences * scale

# For cache uses
class LimitedSizeDict(OrderedDict):
  def __init__(self, *args, **kwds):
    self.size_limit = kwds.pop("size_limit", None)
    OrderedDict.__init__(self, *args, **kwds)
    self._check_size_limit()

  def __setitem__(self, key, value):
    OrderedDict.__setitem__(self, key, value)
    self._check_size_limit()

  def _check_size_limit(self):
    if self.size_limit is not None:
      while len(self) > self.size_limit:
        self.popitem(last=False)

# Collection of usual kernels
class Kernel(object):
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def rbf(gamma):
        # TODO: Compute automatically sigma= 1/n_features
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * gamma ** 2))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset=0.0, gamma=1.0):
        def f(x, y):
            return (gamma * (offset + np.dot(x, y)) )** dimension
        return f

    @staticmethod
    def quadratic(offset=0.0, gamma=1.0):
        def f(x, y):
            return (gamma * (offset + np.dot(x, y)) )** 2
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
    def __init__(self, kernel, C=1.0, support_vector_tol=0.0):
 
        self.kernel = kernel
        self.C = C
        self.support_vectors_ = []
        self.dual_coef_ = []
        self.intercept_ = 0.0  
        self.support_vector_tol = support_vector_tol
               
    def fit(self, X, y):
        """
        Learn parameters of the model
        """
        self.shape_fit_ = X.shape

        res_compute_weights = self._compute_weights(X, y)
        intercept_already_computed = type(res_compute_weights) == tuple
        if intercept_already_computed:
            lagrange_multipliers, intercept = res_compute_weights
            self.intercept_ = intercept
        else:
            lagrange_multipliers = res_compute_weights

        support_vector_indices = lagrange_multipliers > self.support_vector_tol
        
        self.support_ = (support_vector_indices * range(self.shape_fit_[0])).nonzero()[0]
        if support_vector_indices[0]:
            self.support_ = np.insert(self.support_,0,0)
        
        self.dual_coef_ = lagrange_multipliers[support_vector_indices] * y[support_vector_indices]
        self.support_vectors_ = X[support_vector_indices]
        self.n_support_ = np.array([sum(y[support_vector_indices] == -1),
                                    sum(y[support_vector_indices] == 1)])

        if not intercept_already_computed:
            self.intercept_ = np.mean(y[support_vector_indices] - self.predict(self.support_vectors_))

    def _compute_weights(self, X, y):
        raise NotImplementedError()

    def _compute_kernel_support_vectors(self, X):
        res = np.zeros((X.shape[0], self.support_vectors_.shape[0]))
        for i,x_i in enumerate(X):
            for j,x_j in enumerate(self.support_vectors_):
                res[i, j] = self.kernel(x_i, x_j)
        return res

    def _predict_proba(self, X, kernel_support_vectors = None):
        if kernel_support_vectors is None:
            kernel_support_vectors = self._compute_kernel_support_vectors(X)
        prod = np.multiply(kernel_support_vectors, self.dual_coef_)
        prediction = self.intercept_ + np.sum(prod,1)#, keepdims=True)
        return prediction

    def predict_proba(self, X):
        return self._predict_proba(X)
    
    def predict(self, X):
        return np.sign(self._predict_proba(X))

    def predict_value(self, X):
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i, x in enumerate(X):
            result = self.intercept_
            for z_i, x_i in zip(self.dual_coef_,
                                     self.support_vectors_):
                result += z_i * self.kernel(x_i, x)
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
    def __init__(self, kernel, C=1.0, max_iter=1000, cache_size = 200, tol=0.001):
        super().__init__(kernel, C)
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.tol = tol
        self._reset_cache_kernels() 

    def _compute_kernel_matrix_row(self, X, index):
        if self.cache_size != 0 and index in self._cache_kernels:
            return self._cache_kernels[index]
        row = np.zeros(X.shape[0])
        x_i = X[index, :]
        for j,x_j in enumerate(X):
            row[j] = self.kernel(x_i, x_j)
        self._cache_kernels[index] = row
        return row
   
    def _compute_kernel_matrix_diag(self, X):
        n_samples, n_features = X.shape
        diag = np.zeros(n_samples)
        for j,x_j in enumerate(X):
            diag[j] = self.kernel(x_j, x_j)
        return diag
    
    def _reset_cache_kernels(self):
        self._cache_kernels = LimitedSizeDict(size_limit=self.cache_size)
        
    def _compute_intercept(self, alpha, yg):
        indices = (alpha < self.C) * (alpha > 0)
        if len(indices) > 0:
            return np.mean(yg[indices])
        else:
            print('INTERCEPT COMPUTATION ISSUE')
            return 0.0
        
    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples) #feasible solution
        g = np.ones(n_samples) #gradient initialization
        #diag = self._compute_kernel_matrix_diag(X)
        self._reset_cache_kernels()
        while True:

            yg = g * y
            # Working Set Selection
            indices_y_pos = (y == 1)
            indices_y_neg = (np.ones(n_samples) - indices_y_pos).astype(bool)#(y == -1)
            indices_alpha_big = (alpha >= self.C)
            indices_alpha_neg = (alpha <= 0)
            
            indices_violate_Bi_1 = indices_y_pos * indices_alpha_big
            indices_violate_Bi_2 = indices_y_neg * indices_alpha_neg
            indices_violate_Bi = indices_violate_Bi_1 + indices_violate_Bi_2
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf') #do net select violating indices
            
            indices_violate_Ai_1 = indices_y_pos * indices_alpha_neg
            indices_violate_Ai_2 = indices_y_neg * indices_alpha_big
            indices_violate_Ai = indices_violate_Ai_1 + indices_violate_Ai_2
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf') #do net select violating indices
            
            i = np.argmax(yg_i)
            Ki = self._compute_kernel_matrix_row(X, i)
            Kii = Ki[i]
            #indices_violate_criterion = yg_i[i] - yg <= 0
            #vec_j = (yg_i[i] - yg)**2 / (Kii - diag - 2*Ki)
            #vec_j[indices_violate_Ai_1+indices_violate_Ai_2+indices_violate_criterion] = float('-inf')
            
            j = np.argmin(yg_j)
            #j = np.argmax(vec_j)
            Kj = self._compute_kernel_matrix_row(X, j)

            # Stop criterion: stationary point or max iterations
            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break
            
            #compute lambda
            min_1 = (y[i]==1)*self.C -y[i] * alpha[i]
            min_2 = y[j] * alpha[j] + (y[j]==-1)*self.C
            min_3 = (yg_i[i] - yg_j[j])/(Kii + Kj[j] - 2*Ki[j])
            lambda_param = np.min([min_1, min_2, min_3])
            
            #update gradient
            g = g + lambda_param * y * (Kj - Ki)
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1
        # compute intercept
        intercept = self._compute_intercept(alpha, yg)
        
        print('{} iterations for gradient ascent'.format(iteration))
        self._reset_cache_kernels()
        return alpha, intercept        
    
class binary_classification_qp(Base_binary_classification):
    def __init__(self, kernel, C=1.0):
        super().__init__(kernel, C=C)
        
    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
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
        h_2 = cvxopt.matrix(np.ones(n_samples) * self.C)
        G = cvxopt.matrix(np.vstack((G_1, G_2)), tc='d')
        h = cvxopt.matrix(np.vstack((h_1, h_2)), tc='d')
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(sol['x'])

class Base_multiclass(object):
    def __init__(self, kernel, C, algo='qp'):
        self._algo = algo
        self.kernel = kernel
        self.C = C
        self.estimators_ = []
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()
        
    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class multiclass_ovo(Base_multiclass):
    def __init__(self, kernel, C=1.0, algo='smo', max_iter=1000, cache_size = 200, tol=1.0):
        super().__init__(kernel, C=C, algo=algo)
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.tol = tol

    def fit(self, X, y):
        #self._X_shape = X.shape
        self.classes_ = np.unique(y)
        self._pairs = list(itertools.combinations(self.classes_, 2))
        estimators = np.empty(len(self._pairs), dtype=object)
        pairs_indices = []
        for i,pair_of_labels in enumerate(self._pairs):
            print(pair_of_labels)
            if self._algo == 'smo':
                SVM_binary = binary_classification_smo(
                        kernel=self.kernel, C=self.C, max_iter=self.max_iter,
                        cache_size=self.cache_size, tol=self.tol)
            elif self._algo == 'qp':
                SVM_binary = binary_classification_qp(kernel=self.kernel, C=self.C)
            elif self._algo == 'sklearn':
                SVM_binary = SVC(C=self.C, kernel='linear', probability=True)
            else:
                raise Exception('algo not known')  

            X_filtered, y_filtered, classes_indices = self._filter_dataset_by_labels(X, y, pair_of_labels)
            pairs_indices.append(classes_indices)
            SVM_binary.fit(X_filtered, y_filtered)
            estimators[i] = SVM_binary
        self.estimators_ = estimators
        self.pairs_indices = pairs_indices
        self._save_support_vectors(X)

    def _get_class_indices(self, y, label):
        return np.where(y == label)[0]
    
    def _get_classes_indices(self, y, classes):
        res = np.zeros(len(classes), len(y))
        for i, label in enumerate(classes):
            res[i,:] = self._get_class_indices(y, label)
        return res
    
    def _get_two_classes_indices(self, y, label_1, label_2):
        indices_1 = self._get_class_indices(y, label_1)
        indices_2 = self._get_class_indices(y, label_2)
        return np.concatenate((indices_1,indices_2))
    
    def _filter_dataset_by_labels(self, X, y, pair_of_labels):
        label_1, label_2 = pair_of_labels
        classes_indices = self._get_two_classes_indices(y, label_1, label_2)
        X_filtered = X[classes_indices, :]
        y_filtered = y[classes_indices]
        y_filtered[y_filtered == label_1] = -1
        y_filtered[y_filtered == label_2] = 1
        return X_filtered, y_filtered, classes_indices
    
    def _save_support_vectors(self, X):
        indices_to_compute = np.zeros(X.shape[0], dtype=int) 
        for j, estimator in enumerate(self.estimators_):
            indices_to_compute[self.pairs_indices[j][estimator.support_]] = 1
        self.indices_to_compute = indices_to_compute.astype(bool)
        self.all_support_vectors = X.compress(self.indices_to_compute,axis=0)

    def _pre_compute_kernel_vectors(self, X):
        print('Precompute kernel terms between test dataset and support vectors')
        kernel_matrix = np.zeros((X.shape[0], len(self.indices_to_compute)))
        for i,x_i in enumerate(X):
            if i % 500 == 0:
                print('{} / {}'.format(i,X.shape[0]))
            k = 0
            for j, need_compute in enumerate(self.indices_to_compute):
                if need_compute:
                    kernel_matrix[i, j] = self.kernel(x_i, self.all_support_vectors[k,:])
                    k += 1

        kernel_matrices = []
        for j, estimator in enumerate(self.estimators_):
            kernel_matrices.append(kernel_matrix[:,self.pairs_indices[j][estimator.support_]])
        return kernel_matrices
    
    def predict(self, X):
        kernel_matrices = self._pre_compute_kernel_vectors(X)
        confidences = np.empty((X.shape[0], len(self._pairs)))
        for j, estimator in enumerate(self.estimators_):
            confidences[:,j] = estimator._predict_proba(X,kernel_matrices[j])
        predictions = ((np.sign(confidences) + 1 ) / 2).astype(int)
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))    
        return self.classes_[Y.argmax(axis=1)]

#    def predict_by_voting(self, X):
#        n_samples = X.shape[0]
#        predicted_labels = np.zeros((n_samples, len(self._pairs)))
#        for j, pair_of_labels in enumerate(self._pairs):
#            print(pair_of_labels)
#            binary_prediction = self.estimators_[j].predict(X)
#            binary_prediction[binary_prediction == -1] = pair_of_labels[0]
#            binary_prediction[binary_prediction == 1] = pair_of_labels[1]
#            predicted_labels[:,j] = binary_prediction
#        prediction = np.ravel(mode(predicted_labels, axis=1).mode)
#        return prediction

#class multiclass_ova(Base_multiclass):
#    def __init__(self, kernel, C=1.0, algo='smo', max_iter=1000, cache_size = 200, tol=1.0):
#        super().__init__(kernel, C=C, algo=algo)
#        self.max_iter = max_iter
#        self.cache_size = cache_size
#        self.tol = tol
#
#    def fit(self, X, y):
#        self.classes_ = set(y)
#        classifiers = np.empty(len(self.classes_), dtype=object)
#        for i, label in enumerate(self.classes_):
#            print(label)
#            if self._algo == 'smo':
#                SVM_binary = binary_classification_smo(
#                        kernel=self.kernel, C=self.C, max_iter=self.max_iter,
#                        cache_size=self.cache_size, tol=self.tol)
#            else:
#                SVM_binary = binary_classification_qp(self.C, kernel=self.kernel, C=self.C)
#            X_filtered, y_filtered = self._filter_dataset_by_labels(X, y, label)
#            SVM_binary.fit(X_filtered, y_filtered)
#            classifiers[i] = SVM_binary
#        self.estimators_ = classifiers
#
#    def _filter_dataset_by_labels(self, X, y, label):
#        class_pos_indices = (y == label)
#        class_neg_indices = (y != label)
#        X_filtered = X.copy()
#        y_filtered = y.copy()
#        y_filtered[class_pos_indices] = 1
#        y_filtered[class_neg_indices] = -1
#        return X_filtered, y_filtered
#
#    def predict(self, X):
#        n_samples = X.shape[0]
#        predicted_scores = np.zeros((n_samples, len(self.classes_)))
#        for i, label in enumerate(self.classes_):
#            print(i)
#            predicted_scores[:,i] = self.estimators_[i].predict_value(X)
#        return np.argmax(predicted_scores, axis=1)