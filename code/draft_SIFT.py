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
    Parameters:
    	im: RGB image as numpy array
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im

image = rgb2gray(im_test)
plt.figure()
plt.imshow(image,cmap='gray')

def build_octave(image,nb_levels,k,sigma):
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
	return octave

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

## 2nd step: Difference of Gaussians

def log_approx(octave):
	"""
	Compute the Difference of Gaussian images as an approximation of the image Laplacian
	Parameters:
		octave: an octave of the grayscale image
	"""
	nb_levels = int(octave.shape[2])
	DOG = np.ones((octave.shape[0],octave.shape[1],octave.shape[2]-1))
	for i in range(nb_levels - 1):
		DOG[:,:,i] = octave[:,:,i] - octave[:,:,i + 1]
	return DOG

## 3d step: Extrema detection

def extrema_map(DOG):
	"""
	Compute the extrema in scale neighborhood of the image Laplacian
	Parameters:
		DoG: images of the image Laplacian log approximation
	"""
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

# Optional Taylor expansion for subpixel value

## 4th step: Remove bad keypoints

def rm_low_constrast(DOG, Emap, threshold):
	"""
	Remove the keypoints with low constrast features
	Parameters:
		DOG: images of the image laplacian log approximation
		Emap: extrema map of the image laplacian
		threshold: intensity threshold for the DOG image
	"""
	for i in range(int(Emap.shape[2])):
		extrema = Emap[:,:,i]
		DOG_image = DOG[:,:,i+1]
		for k in range(int(extrema.shape[0])):
			for l in range(int(extrema.shape[1])):
				if(abs(DOG_image[k,l]) <= threshold):
					extrema[k,l] = 0
		Emap[:,:,i] = extrema
	return Emap

def rm_edges(DOG, Emap, threshold):
	"""
	Remove the edges and flat keypoints
	Parameters:
		DOG: difference of Gaussians matrix
		threshold: ratio between the two perpendicular gradients at keypoints
	"""
	for i in range(int(Emap.shape[2])):
		extrema = Emap[:,:,i]
		DOG_image = DOG[:,:,i+1]
		for k in range(int(extrema.shape[0])):
			for l in range(int(extrema.shape[1])):
				if(extrema[k,l] == 1): ## Keypoint
					## Compute the Hessian
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
					if((trH**2 / detH) >= ((threshold + 1)**2 / threshold)):
						extrema[k,l] = 0 ## Remove the keypoint
		Emap[:,:,i] = extrema
	return Emap

## 5th step: Assign orientations

def assign_orientation(L,Emap,sigma):
	"""
	Assign orientation and magnitude to each keypoint
	Parameters:
		L: Gaussian blured image
		Emap: extremas of the lagrangian
	"""
	keypoints = np.zeros((1,5)) ## location, scale, orientation, magnitude
	for i in range(int(Emap.shape[2])):
		extrema = Emap[:,:,i]
		L_image = L[:,:,i+1]
		for k in range(int(extrema.shape[0])):
			for l in range(int(extrema.shape[1])):
				if(extrema[k,l] == 1): ## Keypoint
					## Compute magnitude and orientation for all pixels of a 3x3 neighborhood
					N = L_image[k-1:k+2,l-1:l+2]
					M = np.zeros((3,3))
					Theta = np.zeros((3,3))
					for mi in range(3):
						for mj in range(3):
							M[mi, mj] = math.sqrt( (L_image[mi+1, mj] - L_image[mi-1,mj])**2 + (L_image[mi, mj+1] - L_image[mi, mj-1])**2 )
							Theta[mi, mj] = math.degrees(math.atan( (L_image[mi, mj+1] - L_image[mi, mj-1])/(L_image[mi+1, mj] - L_image[mi-1, mj])) )
					## Blur the magnitude matrix
					M = filters.gaussian_filter(M, 1.5 * sigma)
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
					key = [k,l,(i+1) * sigma, keypoint_orientation, keypoint_magnitude]
					keypoints = np.concatenate((keypoints, key), axis=1)
					## Check if another orientation is above 80% of the maximum
					for mi in range(36):
						if(H[mi] > 0.8 * np.max(H)):
							if(mi != np.argmax(H)):
								keypoint_orientation = mi
								keypoint_magnitude = H[mi]
								key = [k,l,(i+1) * sigma, keypoint_orientation, keypoint_magnitude]
								keypoints = np.concatenate((keypoints, key), axis=1)
	return keypoints















