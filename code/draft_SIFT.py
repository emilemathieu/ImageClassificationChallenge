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

# for i in range(nb_levels):
# 	plt.figure()
# 	plt.imshow(octave1[:,:,i],cmap='gray')
# 	plt.title('octave1')
# for i in range(nb_levels):
# 	plt.figure()
# 	plt.imshow(octave2[:,:,i],cmap='gray')
# 	plt.title('octave2')
# for i in range(nb_levels):
# 	plt.figure()
# 	plt.imshow(octave3[:,:,i],cmap='gray')
# 	plt.title('octave3')
# for i in range(nb_levels):
# 	plt.figure()
# 	plt.imshow(octave4[:,:,i],cmap='gray')
# 	plt.title('octave4')

## 2nd step: Difference of Gaussians

def log_approx(octave):
	nb_levels = int(octave.shape[2])
	DOG = np.ones((octave.shape[0],octave.shape[1],octave.shape[2]-1))
	for i in range(nb_levels - 1):
		DOG[:,:,i] = octave[:,:,i] - octave[:,:,i + 1]
	return DOG

DOG1 = log_approx(octave1)
DOG2 = log_approx(octave2)
DOG3 = log_approx(octave3)
DOG4 = log_approx(octave4)

## 3d step: Extrema detection

def extrema_map(DOG):
	extrema_map = np.zeros((DOG.shape[0], DOG.shape[1], DOG.shape[2] - 2))
	for k in range(1, int(DOG.shape[2])-1):
		extrema = np.zeros((DOG.shape[0], DOG.shape[1]))
		for i in range(1, int(DOG.shape[0]) - 1):
			for j in range(1, int(DOG.shape[1]) - 1):
				neighborhood_upscale = DOG[i-1:i+2, j-1:j+2, k-1].flatten()
				neighborhood = DOG[i-1:i+2, j-1:j+2, k].flatten()
				neighborhood_downscale = DOG[i-1:i+2, j-1:j+2, k+1].flatten()
				N = np.concatenate((neighborhood_upscale, neighborhood), axis=0)
				N = np.concatenate((N, neighborhood_downscale), axis=0)
				sample = DOG[i,j,k]
				if(sample == np.max(N)):
					extrema[i,j] = 1
				elif(sample == np.min(N)):
					extrema[i,j] = 1
		extrema_map[:,:,k-1] = extrema
	return extrema_map













