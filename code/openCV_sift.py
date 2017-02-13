# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:17:37 2017

@author: Thomas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pandas as pd

print('openCV version: {}'.format(cv2.__version__))



IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

X = pd.read_csv('../data/Xtr.csv', header=None)
X = X.as_matrix()
X = X[:, 0:-1]

Y = pd.read_csv('../data/Ytr.csv')

#%% 
index = 100
image1 = X[index,:]
image1 = image1.reshape((3, CHANEL_SIZE))
image1 = image1.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
image1 = image1.swapaxes(0,1)
image1 = image1.swapaxes(1,2)

def rgb2gray(im):
    """
    Convert image im to grayscale
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im

im_gray = rgb2gray(image1)
plt.imshow(im_gray, cmap='gray')

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

im_gray_kp, im_gray_desc = gen_sift_features(im_gray)

plt.figure()
show_sift_features(im_gray, image1, im_gray_kp)
plt.title('SIFT features from openCV')

