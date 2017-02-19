# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:58:07 2017

@author: Thomas PESNEAU
"""

import numpy as np
import scipy.ndimage.filters as filters

## Rotation 90, 180, 270
def rotate(image, quarter):
    """
    Image: image to rotate w x h, numpy array
    axis: 1->90, 2->180, 3->270
    """
    rotate_image = np.rot90(image, quarter)
    return rotate_image


## Blur the image
def blur(image, sigma=0.5):
    blur_image = filters.gaussian_filter(image, sigma)
    return blur_image
    

## Add noise
def noise(image,sigma=0.01):
    gaussian_noise = np.random.normal(loc=0.,scale=sigma,size=image.shape)
    noisy_image = image + gaussian_noise
    noisy_image[noisy_image > 1] = 1
    noisy_image[noisy_image < -1] = -1
    return noisy_image
    




