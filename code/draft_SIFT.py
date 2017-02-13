# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:56:05 2017

@author: Thomas
Implement the SIFT features of an image 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doctest
import math
import scipy.ndimage.filters as filters
from PIL import Image 

## Open test image
im_test = Image.open("test_sift.jpg")


def L(I,sigma,k):
	"""
	Compute the convolution of an image by a Gaussian function
	Parameters:
		I 		: grayscale image
		sigma 	: variance of the gaussian kernel
		k 		: scale factor
	Outputs:
		L : Smoothed image
	"""
	L = filters.gaussian_filter(I, k * sigma)
	return L

def D(I,sigma,s):
	"""
	Compute the difference of Gaussians over an Octave
	Parameters:
		I 		: grayscale image
		sigma 	: variance of the gaussian kernel
		k 		: scale factor
	Outputs:
		D : Smoothed image
	"""
	## Compute the DoG of an Octave
	k = 2 ** (1./s)
	octave = np.ones((I.shape[0],I.shape[1],s+3))
	DoG = np.ones((I.shape[0],I.shape[1],s+2))
	for i in range(s+3):
		octave[:,:,i] = L(I,sigma,k)
		plt.figure()
		plt.imshow(octave[:,:,i], cmap= plt.get_cmap('gray'))
		k *= k
	for i in range(s+2):
		DoG[:,:,i] = octave[:,:,i+1] - octave[:,:,i]
	return DoG
 
#DoG = D(image1,1,5)

########## SIFT algorithm ############

## 1st step: Generate the scale space





