# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:04:58 2017

@author: Thomas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doctest
import math
import scipy.ndimage.filters as filters
from PIL import Image 


def resample2(image):
	"""
	Resample an array by taking every second value
	Parameters:
		image: image in grayscale as numpy array
	"""
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

def rgb2gray(im):
    """
    Convert image im to grayscale
    Parameters:
    	im: RGB image as numpy array
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im

class keypoint(object):
	"""
	SIFT keypoint with features
	"""
	def __init__(self, x, y, scale):
		self.x = x
		self.y = y
		self.scale = scale
		self.orientation = 0
		self.magnitude = 0
		self.features = []

class Octave(object):
	def __init__(self, image, nb_levels, k, sigma):
		self.image = image
		self.nb_levels = nb_levels
		self.k = k
		self.sigma = sigma
		self.DoG = 0
		self.keys = []
		self.scale = []
		for i in range(self.nb_levels):
			self.scale.append(k ** (i+1) * self.sigma)
		self.octave = np.array()
		self.DOG = np.array()
		
	def build_octave(self):
		"""
		Generate an octave of the image
		Parameters:
			image: image in grayscale as numpy array
			nb_levels: number of scale levels in the octave
			k: blur ratio between scales
			sigma: variance of the gaussian blur
		"""
		octave = np.ones((image.shape[0],image.shape[1],nb_levels))
		for i in range(nb_levels):
			octave[:,:,i] = filters.gaussian_filter(image, (1+i) * k * sigma)
		self.octave = octave

	def DoG(self):
		"""
		Compute the Difference of Gaussian images as an approximation of the image Laplacian
		Parameters:
			octave: an octave of the grayscale image
		"""
		octave = self.octave
		nb_levels = int(octave.shape[2])
		DOG = np.ones((octave.shape[0],octave.shape[1],octave.shape[2]-1))
		for i in range(nb_levels - 1):
			DOG[:,:,i] = octave[:,:,i] - octave[:,:,i + 1]
		self.DOG = DOG

	def find_extrema(self):
		DOG = self.DOG
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
						key = keypoint(i, j, (k-1)*self.sigma)
						self.keys.append(key)
					elif(sample == np.min(N)):
						key = keypoint(i, j, (k-1)*self.sigma)
						self.keys.append(key)

	def rm_bkeys(self, threshold_contrast, threshold_edge):
		new_keys = []
		for key in self.keys:
			DOG_image = self.DOG[:,:,int(key.scale / self.sigma)]

			## Remove low constrasts
			if(DOG_image[key.x, key.y] < threshold_contrast):
				new_keys.append(key)

			## Remowe edges
			k = key.x
			l = key.y
			H = np.zeros((2,2))
			H[0,0] = (DOG_image[k+1,l] - DOG_image[k,l]) - (DOG_image[k,l] - DOG_image[k-1,l])
			dx1 = (DOG_image[k+1,l-1] - DOG_image[k-1,l-1]) / 2
			dx2 = (DOG_image[k+1,l+1] - DOG_image[k-1,l+1]) / 2
			H[0,1] = d2 - d1
			dy1 = (DOG_image[k-1,l+1] - DOG_image[k-1,l-1]) / 2
			dy2 = (DOG_image[k+1,l+1] - DOG_image[k+1,l-1]) / 2
			H[1,0] = dy2 - dy1
			H[1,1] = (DOG_image[k,l+1] - DOG_image[k,l]) - (DOG_image[k,l] - DOG_image[k,l-1])
			## Compute the Trace and Determinant
			trH = np.trace(H)
			detH = np.linalg.det(H)
			## Test on the ratio
			if((trH**2 / detH) < ((threshold_edge + 1)**2 / threshold_edge)):
				new_keys.append(key)
		self.keys = new_keys

	def assign_orientation(self):
		for key in self.keys:
			L_image = self.octave[:,:,int(key.scale / self.sigma)]
			## Compute magnitude and orientation around the keypoint
			k = key.x
			l = key.y
			N = L_image[k-1:k+2,l-1:l+2]
			M = np.zeros((3,3))
			Theta = np.zeros((3,3))
			for mi in range(3):
				for mj in range(3):
					M[mi, mj] = math.sqrt( (L_image[mi+1, mj] - L_image[mi-1,mj])**2 + (L_image[mi, mj+1] - L_image[mi, mj-1])**2 )
					Theta[mi, mj] = math.degrees(math.atan( (L_image[mi, mj+1] - L_image[mi, mj-1])/(L_image[mi+1, mj] - L_image[mi-1, mj])) )
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Build a histogram
			H = np.zeros((36,1))
			M = M.flatten()
			Theta = Theta.flatten()
			for mi in range(36):
				bin = math.floor(Theta[mi] / 10) ## between 0 and 35
				H[bin] = Theta[mi] * M[mi]
			## Define the orientation of the keypoint
			keypoint_orientation = np.argmax(H)
			keypoint_magnitude = np.max(H)
			key.orientation = keypoint_orientation
			key.magnitude = keypoint_magnitude
			## Check if another orientation is above 80% of the maximum
			for mi in range(36):
				if(H[mi] > 0.8 * np.max(H)):
					if(mi != np.argmax(H)):
						keypoint_orientation = mi
						keypoint_magnitude = H[mi]
						key = keypoint(k, l, key.scale, keypoint_orientation, keypoint_magnitude)

	def generate_features(self):


