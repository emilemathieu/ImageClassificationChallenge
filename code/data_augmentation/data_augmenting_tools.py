# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:58:07 2017

@author: Thomas PESNEAU
"""

import numpy as np
import scipy.ndimage.filters as filters
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import random
import pandas as pd
import scipy

## Open datacs
IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

#X = pd.read_csv('../../data/Xtr.csv', header=None)
#X = X.as_matrix()
#X = X[:, 0:-1]
#index = 100
#image = X[index,:]
#image = image.reshape((3, CHANEL_SIZE))
#image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
#image = image.swapaxes(0,1)
#image = image.swapaxes(1,2)

def rgb2gray(im):
    """
    Convert image im to grayscale
    Parameters:
    	im: RGB image as numpy array
    """
    im = np.dot(im[:,:,:3],[0.299, 0.587, 0.114])
    return im
## Rotation 90, 180, 270
def random_rotate(image):
    """
    Image: image to rotate w x h, numpy array
    """
    angle = random.randrange(-9,9)
    rotate_image = scipy.ndimage.interpolation.rotate(image,angle)
    if(rotate_image.shape[0] != 24):
        rotate_image = rotate_image[5:29,5:29,:]
    return rotate_image

def random_flip(image):
    """
    Flips the image randomly from left to right
    """
    r = random.random()
    if(r < 0.5):
        if(image.shape[0] != 24):
            image = image[5:29,5:29,:]
        return image
    else:
        f_image = np.fliplr(image)
        if(f_image.shape[0] != 24):
            f_image = f_image[5:29,5:29,:]
        return f_image

#def random_brightness(image):
#    """
#    Randomize brightness of the image
#    """
#    b_image = cls.rgb_to_hsv(image)
#    brightness = random.uniform(-0.2,0.2)
#    b_image[:,:,2] += brightness
#    return b_image
#    
### Blur the image
#def blur(image, sigma=0.5):
#    blur_image = filters.gaussian_filter(image, sigma)
#    return blur_image
#    
#
### Add noise
#def noise(image,sigma=0.01):
#    gaussian_noise = np.random.normal(loc=0.,scale=sigma,size=image.shape)
#    noisy_image = image + gaussian_noise
#    noisy_image[noisy_image > 1] = 1
#    noisy_image[noisy_image < -1] = -1
#    return noisy_image
    




