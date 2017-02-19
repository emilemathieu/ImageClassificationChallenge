#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:42:31 2017

@author: EmileMathieu
"""

import numpy as np

class Optimizer(object):
    def __init__(self, parameters = []):
        self._parameters = parameters   
        self.state = {}
    def __call__(self, objective, gradient):
        raise NotImplementedError()
#    def zero_grad(self):
#        for parameter in self._parameters:
#            print(parameter)
#            # set gradient of parameter to zero
#    def step(self):
#        for parameter in self._parameters:
#            print(parameter)
#            # step in direction of parameter's gradient

class SGD(Optimizer):
    def __init__(self, lr=0.1, momentum=0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        
    def get_state(self, key):
        if self.momentum != 0 and key in self.state:
            mu = self.state[key]
        else:
            mu = 0
        return mu

    def __call__(self, layer_id, weight_type, objective, gradient):
        key = str(layer_id) + weight_type
        old_v = self.get_state(key)
        new_v = self.lr * gradient

        new_v += self.momentum * old_v
        self.state[key] = new_v
        return objective - new_v

class Adam(Optimizer):
    """
	ADAM Gradient descent optimization algorithm
	-------------
	Parameters:
		eta 	(float, default: 0.001): 
			learning rate of the gradient descent
		betas	(tuple[float, float], default: [0.9, 0.999]):
			decay rates for first and second order momentums
		eps		(float, default: 10e-8): 
			term for numerical stability
		dJ		(function):
			Gradient of the function to optimize
		theta	(array):
			Initial point
	"""
    def __init__(self, lr=0.001, betas=[0.9, 0.999], eps=10e-8):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
    def get_state(self, key, shape):
        if key in self.state:
            m, v = self.state[key]
        else:
            m = np.zeros(shape)
            v = np.zeros(shape)
        return m, v

    def __call__(self, layer_id, weight_type, objective, gradient):
        """
        Performs a step of Gradient descent with ADAM optimization
        """
        key = str(layer_id) + weight_type
        m, v = self.get_state(key, gradient.shape)

        g = gradient
        beta1 = self.betas[0]
        beta2 = self.betas[1]
        m = beta1 * m + (1 - beta1) * g ## Estimate of the mean of the gradient
        v = beta2 * v + (1 - beta2) * (g**2) ## Estimate of the variance of the gradient
        self.state[key] = m, v
        ## Correct biases
        m_unbiased = m / (1 - beta1)
        v_unbiased = v / (1 - beta2)
        ## Define new step
        objective = objective - self.lr / (np.sqrt(v_unbiased) + self.eps) * m_unbiased
        return objective