#import cvxopt
import numpy as np
from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    """ Ordered dictionary with limited size of keys
    Used for cache purposes
    """
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

class Base_binary_classification(object):
    """Base class for C-Support Vector Classification.
    The SVM dual formulation is solve with a quadratic problem solver (cvxopt)
    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    kernel : callable, (R^n_features,R^n_features) -> R
            kernel function
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.
    n_support_ : array-like, dtype=int32, shape = [2]
        Number of support vectors for each class.
    dual_coef_ : array, shape = [n_SV]
        Coefficients of the support vector in the decision function.
    intercept_ : float
        Constant in decision function.
    """
    def __init__(self, kernel, C=1.0, support_vector_tol=0.0):
        self.kernel = kernel
        self.C = C
        self.support_vectors_ = []
        self.dual_coef_ = []
        self.intercept_ = 0.0  
        self.support_vector_tol = support_vector_tol
               
    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
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
        """Predict data's labels according to previously fitted data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        labels : array-like, shape = [n_samples]
            Predicted labels of X.
        """
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
        """ Compute the average score of the model on X given labels y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        score : float
            Score of the model.
        """
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class binary_classification_smo(Base_binary_classification):
    """C-Support Vector Classification.
    Follows SMO scheme. See "Support Vector Machine Solvers" from Léon Bottou & Chih-Jen Lin
    See also "LIBSVM: A Library for Support Vector Machines"
    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    kernel : callable, (R^n_features,R^n_features) -> R
            kernel function
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    cache_size : float, optional (default=200)
        Specify the size of the kernel cache.
    max_iter : int, optional (default=1000)
        Hard limit on iterations, or -1 for no limit.
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.
    n_support_ : array-like, dtype=int32, shape = [2]
        Number of support vectors for each class.
    dual_coef_ : array, shape = [n_SV]
        Coefficients of the support vector in the decision function.
    intercept_ : float
        Constant in decision function.
    """
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
    """C-Support Vector Classification.
    The SVM dual formulation is solve with a quadratic problem solver (cvxopt)
    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    kernel : callable, (R^n_features,R^n_features) -> R
            kernel function
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.
    n_support_ : array-like, dtype=int32, shape = [2]
        Number of support vectors for each class.
    dual_coef_ : array, shape = [n_SV]
        Coefficients of the support vector in the decision function.
    intercept_ : float
        Constant in decision function.
    """
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

#        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
#        q = cvxopt.matrix(-1 * np.ones(n_samples), tc='d')
#        G_1 = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
#        h_1 = cvxopt.matrix(np.zeros(n_samples))
#        G_2 = cvxopt.matrix(np.diag(np.ones(n_samples)))
#        h_2 = cvxopt.matrix(np.ones(n_samples) * self.C)
#        G = cvxopt.matrix(np.vstack((G_1, G_2)), tc='d')
#        h = cvxopt.matrix(np.vstack((h_1, h_2)), tc='d')
#        A = cvxopt.matrix(y, (1, n_samples), tc='d')
#        b = cvxopt.matrix(0.0, tc='d')
#        
#        cvxopt.solvers.options['show_progress'] = False
#        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(sol['x'])
