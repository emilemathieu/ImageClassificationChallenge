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
im_test = Image.open("../data/test_sift.jpg")
im_test = np.array(im_test)


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

########## SIFT algorithm ############

## 1st step: Constructing a scale space
# Create 4 octaves at 5 blur levels
k = math.sqrt(2)/5
nb_levels = 5
sigma = 1. / math.sqrt(2)

def rgb2gray(im):
    """
    Convert image im to grayscale
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im

image = rgb2gray(im_test)
plt.figure()
plt.imshow(image,cmap='gray')

def build_octave(image,nb_levels,k,sigma):
	octave = np.ones((image.shape[0],image.shape[1],nb_levels))
	for i in range(nb_levels):
		octave[:,:,i] = filters.gaussian_filter(image, (1+i) * k * sigma)
	return octave

def resample2(image):
	[h,w] = [image.shape[0], image.shape[1]]
	im_resample = np.ones((int(math.ceil(h/2)),int(math.ceil(w/2))))
	count_i = 0
	for i in range(0,int(h),2):
		count_j = 0
		for j in range(0,int(w),2):
			im_resample[count_i, count_j] = image[i, j]
			count_j += 1
		count_i +=1
	return im_resample

octave1 = build_octave(image, nb_levels, k, sigma)
image_resample1 = resample2(image)
octave2 = build_octave(image_resample1, nb_levels, k, sigma)
image_resample2 = resample2(image_resample1)
octave3 = build_octave(image_resample2, nb_levels, k, sigma)
image_resample3 = resample2(image_resample2)
octave4 = build_octave(image_resample3, nb_levels, k, sigma)

for i in range(nb_levels):
	plt.figure()
	plt.imshow(octave1[:,:,i],cmap='gray')
	plt.title('octave1')
for i in range(nb_levels):
	plt.figure()
	plt.imshow(octave2[:,:,i],cmap='gray')
	plt.title('octave2')
for i in range(nb_levels):
	plt.figure()
	plt.imshow(octave3[:,:,i],cmap='gray')
	plt.title('octave3')
for i in range(nb_levels):
	plt.figure()
	plt.imshow(octave4[:,:,i],cmap='gray')
	plt.title('octave4')






