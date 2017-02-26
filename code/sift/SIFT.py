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
from astropy.convolution import convolve, Gaussian2DKernel


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
	def __init__(self, x, y, scale,orientation=0,magnitude=0):
		self.x = x
		self.y = y
		self.scale = scale
		self.orientation = orientation
		self.magnitude = magnitude
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
		self.octave = np.array([])
		self.DOG = np.array([])
		
	def build_octave(self):
		"""
		Generate an octave of the image
		Parameters:
			image: image in grayscale as numpy array
			nb_levels: number of scale levels in the octave
			k: blur ratio between scales
			sigma: variance of the gaussian blur
		"""
		octave = np.ones((self.image.shape[0],self.image.shape[1],self.nb_levels))
		for i in range(self.nb_levels):
			gkernel = Gaussian2DKernel(stddev=math.sqrt((self.k**i) * self.sigma), x_size=3, y_size=3)
			plt.figure()
			plt.imshow(gkernel)
			#octave[:,:,i] = filters.gaussian_filter(self.image, i * self.k * self.sigma)
			octave[:,:,i] = convolve(self.image,gkernel)
		self.octave = octave

	def log_approx(self):
		"""
		Compute the Difference of Gaussian images as an approximation of the image Laplacian
		Parameters:
			octave: an octave of the grayscale image
		"""
		octave = self.octave
		nb_levels = int(octave.shape[2])
		DOG = np.ones((self.octave.shape[0],self.octave.shape[1],nb_levels-1))
		for i in range(self.nb_levels - 1):
			DOG[:,:,i] = octave[:,:,i] - octave[:,:,i + 1]
		self.DOG = DOG

	def find_extrema(self):
		"""
		Find the extrema of the DoG ie image lagrangian approximation function
		Parameters:
			DOG: the difference of gaussians function
		"""
		DOG = self.DOG
		for k in range(1, int(DOG.shape[2])-1):
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
		"""
		Remove the keypoints which are not pertinent (low constrast or on an edge)
		Parameters:
			threshold_contrast: threshold value for keypoint contrast 
			threshold_edge: threshold value for keypoint gradient magnitude (ratio between the two eigenvalues)
		"""
		new_keys = []
		for key in self.keys:
			DOG_image = self.DOG[:,:,int(key.scale / self.sigma)]

			## Remove low constrasts
			if(DOG_image[key.x, key.y] < threshold_contrast):
				new_keys.append(key)

			## Remove edges
			k = key.x
			l = key.y
			H = np.zeros((2,2))
			H[0,0] = (DOG_image[k+1,l] - DOG_image[k,l]) - (DOG_image[k,l] - DOG_image[k-1,l])
			dx1 = (DOG_image[k+1,l] - DOG_image[k,l])
			dx2 = (DOG_image[k+1,l+1] - DOG_image[k,l+1])
			H[0,1] = dx2 - dx1
			dy1 = (DOG_image[k,l+1] - DOG_image[k,l])
			dy2 = (DOG_image[k+1,l+1] - DOG_image[k+1,l])
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
		"""
		Assign the dominant orientation of the gradient of each keypoint
		"""
		#print("Number of keypoints: {}".format(len(self.keys)))
		compt = 0
		new_keys = []
		for key in self.keys:
			compt +=1
			#print("keypoint #{}".format(compt))
			L_image = self.octave[:,:,int(key.scale / self.sigma)]
			## Compute magnitude and orientation around the keypoint
			k = key.x
			l = key.y
			M = np.zeros((3,3))
			Theta = np.zeros((3,3))
			compt_i = 0
			for mi in range(-1,2):
				compt_j = 0
				for mj in range(-1,2):
					#print("compt_i: {}\ncompt_j: {}".format(compt_i, compt_j))
					current_x = max(0,min(k+mi,L_image.shape[0]-2))
					current_y = max(0,min(l+mj,L_image.shape[1]-2))
#					print("current x: {}".format(current_x))
#					print("current y: {}".format(current_y))
					M[compt_i, compt_j] = math.sqrt( (L_image[current_x+1, current_y] - L_image[current_x-1,current_y])**2 + (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])**2 )
					if((L_image[current_x+1, current_y] - L_image[current_x-1, current_y]) == 0):
						Theta[compt_i, compt_j] = 90
					else:
						Theta[compt_i, compt_j] = math.degrees(math.atan( (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])/(L_image[current_x+1, current_y] - L_image[current_x-1, current_y])) )
					compt_j += 1
				compt_i += 1
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Build a histogram
			H = np.zeros((36,1))
			M = M.flatten()
			Theta = Theta.flatten()
			for mi in range(9):
				bin_h = int(math.floor(Theta[mi] / 10)) ## between 0 and 35
				H[bin_h] = Theta[mi] * M[mi]
			## Define the orientation of the keypoint
			keypoint_orientation = np.argmax(H)
			keypoint_magnitude = np.max(H)
			key.orientation = keypoint_orientation
			key.magnitude = keypoint_magnitude
			new_keys.append(key)
			## Check if another orientation is above 80% of the maximum
			for mi in range(36):
				if(H[mi] > 0.8 * np.max(H)):
					if(mi != np.argmax(H)):
						keypoint_orientation = mi
						keypoint_magnitude = H[mi]
						new_key = keypoint(k, l, key.scale, keypoint_orientation, keypoint_magnitude)
						new_keys.append(new_key) ## add the new keypoint with different orientation and magnitude
		self.keys = new_keys

	def generate_features(self,wsize):
		"""
		Generate the actual SIFT features from the neighborhood gradient orientation and magnitude
		Parameters:
			wsize: size of the neighborhood (16x16 in D.Lowe, 8x8 in our case because of small images)
		"""
		vsize = int(wsize/2)
		for key in self.keys:
			L_image = self.octave[:,:,int(key.scale / self.sigma)]
			## Compute the magnitude and orientation around the keypoint
			k = key.x
			l = key.y

			## First quadrant
			M = np.zeros((vsize,vsize))
			Theta = np.zeros((vsize,vsize))
			for mi in range(1,1+vsize):
				for mj in range(1,1+vsize):
					current_x = max(0,min(k-mi,L_image.shape[0]-2))
					current_y = max(0,min(l-mj,L_image.shape[1]-2))
					M[mi-1,mj-1] = math.sqrt( (L_image[current_x+1,l-mj] - L_image[current_x-1,l-mj])**2 + (L_image[current_x, l-mj+1] - L_image[current_x, l-mj-1])**2 )
					if((L_image[current_x+1, l-mj] - L_image[current_x-1, l-mj]) == 0):
						Theta[mi-1,mj-1] = 90
					else:
						Theta[mi-1,mj-1] = math.degrees(math.atan( (L_image[current_x, l-mj+1] - L_image[current_x, l-mj-1])/(L_image[current_x+1, l-mj] - L_image[current_x-1, l-mj])) )
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Put orientations in a 8 bin histogram
			hist1 = [0]*8
			for mi in range(vsize):
				for mj in range(vsize):
					bin_h = int(math.floor( Theta[mi,mj] / 45)) ## between 0 and 7
					hist1[bin_h] += M[mi,mj] * Theta[mi,mj]

			## Second quadrant
			M = np.zeros((vsize,vsize))
			Theta = np.zeros((vsize,vsize))
			for mi in range(1,1+vsize):
				for mj in range(1,1+vsize):
					current_x = max(0,min(k-mi,L_image.shape[0]-2))
					current_y = max(0,min(l+mj,L_image.shape[1]-2))
					M[mi-1,mj-1] = math.sqrt( (L_image[current_x+1,current_y] - L_image[current_x-1,current_y])**2 + (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])**2 )
					if((L_image[current_x+1, current_y] - L_image[current_x-1, current_y]) == 0):
						Theta[mi-1,mj-1] = 90
					else:
						Theta[mi-1,mj-1] = math.degrees(math.atan( (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])/(L_image[current_x+1, current_y] - L_image[current_x-1, current_y])) )
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Put orientations in a 8 bin histogram
			hist2 = [0]*8
			for mi in range(vsize):
				for mj in range(vsize):
					bin_h = int(math.floor( Theta[mi,mj] / 45))
					hist2[bin_h] += M[mi,mj] * Theta[mi,mj]

			## Third quadrant
			M = np.zeros((vsize,vsize))
			Theta = np.zeros((vsize,vsize))
			for mi in range(1,1+vsize):
				for mj in range(1,1+vsize):
					current_x = max(0,min(k+mi,L_image.shape[0]-2))
					current_y = max(0,min(l-mj,L_image.shape[1]-2))
					M[mi-1,mj-1] = math.sqrt( (L_image[current_x+1,current_y] - L_image[current_x-1,current_y])**2 + (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])**2 )
					if((L_image[current_x+1, current_y] - L_image[current_x-1, current_y]) == 0):
						Theta[mi-1,mj-1] = 90
					else:
						Theta[mi-1,mj-1] = math.degrees(math.atan( (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])/(L_image[current_x+1, current_y] - L_image[current_x-1, current_y])) )
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Put orientations in a 8 bin histogram
			hist3 = [0]*8
			for mi in range(vsize):
				for mj in range(vsize):
					bin_h = int(math.floor( Theta[mi,mj] / 45))
					hist3[bin_h] += M[mi,mj] * Theta[mi,mj]

			## Fourth quadrant
			M = np.zeros((vsize,vsize))
			Theta = np.zeros((vsize,vsize))
			for mi in range(1,1+vsize):
				for mj in range(1,1+vsize):
					current_x = max(0,min(k+mi,L_image.shape[0]-2))
					current_y = max(0,min(l+mj,L_image.shape[1]-2))
					M[mi-1,mj-1] = math.sqrt( (L_image[current_x+1,current_y] - L_image[current_x-1,current_y])**2 + (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])**2 )
					if( (L_image[current_x+1,current_y] - L_image[current_x-1, current_y]) == 0):
						Theta[mi-1,mj-1] = 90
					else:
						Theta[mi-1,mj-1] = math.degrees(math.atan( (L_image[current_x, current_y+1] - L_image[current_x, current_y-1])/(L_image[current_x+1, current_y] - L_image[current_x-1, current_y])) )
			## Blur the magnitude matrix
			M = filters.gaussian_filter(M, 1.5 * self.sigma)
			## Put orientations in a 8 bin histogram
			hist4 = [0]*8
			for mi in range(vsize):
				for mj in range(vsize):
					bin_h = int(math.floor( Theta[mi,mj] / 45))
					hist4[bin_h] += M[mi,mj] * Theta[mi,mj]

			## Concatenate the histograms
			hist1 = np.array(hist1)
			hist2 = np.array(hist2)
			hist3 = np.array(hist3)
			hist4 = np.array(hist4)
			Hist = np.concatenate((hist1, hist2), axis=0)
			Hist = np.concatenate((Hist, hist3), axis=0)
			Hist = np.concatenate((Hist, hist4), axis=0)

			## Ensure rotation independence
			Hist = Hist - key.orientation

			## Ensure illumination independence
			Hist[Hist > 0.2] = 0.2

			## Assign Hist to keypoint features
			key.features = Hist


def SIFT_descriptor(image, nb_levels, k, sigma, t_contrast, t_edge, wsize):
	"""
	Main function
	"""
	image = rgb2gray(image)
	Oct = Octave(image, nb_levels, k, sigma)
	Oct.build_octave()
	Oct.log_approx()
	Oct.find_extrema()
	Oct.rm_bkeys(t_contrast, t_edge)
	Oct.assign_orientation()
	Oct.generate_features(wsize)
	Features = np.zeros(32)
	for key in Oct.keys:
		feature = key.features 
		Features = np.concatenate((Features,feature),axis=0)
	Features = Features[32:]
	return Features

#%%
X = pd.read_csv('../../data/Xtr.csv', header=None)
X = X.as_matrix()
X = X[:, 0:-1]

def dataset_SIFT(X,nb_levels,k,sigma,t_contrast,t_edge,wsize):
    X_features = []
    null_list = []
    for index in range(X.shape[0]):
        print("Sample #{}".format(index))
        image = X[index,:]
        image = image.reshape((3, 32*32))
        image = image.reshape((3, 32, 32))
        image = image.swapaxes(0,1)
        image = image.swapaxes(1,2)
        Features = SIFT_descriptor(image, nb_levels, k, sigma, t_contrast, t_edge, wsize)
        print("Features: {}".format(len(Features)))
        if(len(Features) == 0):
            null_list.append(index)
        X_features.append(Features)
    print("Zero descriptor for images: {}".format(len(null_list)))
    return X_features

nb_levels = 5
k = 1.1
sigma = 1. / math.sqrt(2)
t_contrast = 1.2
t_edge = 10
wsize = 8
#X_SIFT = dataset_SIFT(X,nb_levels,k,sigma,t_contrast,t_edge,wsize)
        




