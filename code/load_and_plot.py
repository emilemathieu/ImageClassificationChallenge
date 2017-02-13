#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pandas as pd

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
#plt.imshow(image1)

def rgb2gray(im):
    """
    Convert image im to grayscale
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im

im_gray = rgb2gray(image1)
plt.imshow(im_gray, cmap=plt.get_cmap('gray'))