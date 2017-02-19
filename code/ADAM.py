# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:49:20 2017

@author: Thomas PESNEAU
"""

import numpy as np
import math

class ADAM_optimizer(object):
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
	def __init__(self, eta=0.001, betas=[0.9, 0.999], eps=10e-8, dJ, theta):
		self.eta = eta
		self.betas = betas
		self.eps = eps
		self.dJ = dJ
		self.m = np.zeros(dJ.shape)
		self.v = np.zeros(dJ.shape)
		self.theta = theta
		self.step = 0

	def step(self):
		"""
		Performs a step of Gradient descent with ADAM optimization
		"""
		g = self.dJ(self.theta)
		beta1 = self.betas[0]
		beta2 = self.betas[1]
		self.m = beta1 * self.m + (1 - beta1) * g ## Estimate of the mean of the gradient
		self.v = beta2 * self.v + (1 - beta2) * (g**2) ## Estimate of the variance of the gradient
		## Correct biases
		m_unbiased = self.m / (1 - beta1)
		v_unbiased = self.v / (1 - self.betas[1])
		## Define new step
		self.theta = self.theta - ( eta / (math.sqrt(v_unbiased) + self.eps) ) * m_unbiased
		self.step += 1


		


