# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:39:08 2017

@author: Thomas PESNEAU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doctest
import math
import scipy.ndimage.filters as filters
from PIL import Image 
from importlib import reload

## Open test image
im_test = Image.open("../../data/test_sift.jpg")
image = np.array(im_test)

## Test 
import SIFT
nb_levels = 5
k = 1.1
sigma = 1. / math.sqrt(2)
t_contrast = 1.2
t_edge = 10
wsize = 8

#%%
image = SIFT.rgb2gray(image)
Oct = SIFT.Octave(image, nb_levels, k, sigma)
#%%
print("Build octave")
Oct.build_octave()
for i in range(nb_levels):
    plt.figure()
    plt.imshow(Oct.octave[:,:,i],cmap='gray')
    plt.title("scale {}".format(i))
#%%
print("Log approximation")
Oct.log_approx()
#%%
print("Extrema")
Oct.find_extrema()
#%%
print("Remove bad keypoints")
Oct.rm_bkeys(t_contrast, t_edge)
#%%
print("Assign orientation")
Oct.assign_orientation()
#%%
print("Generate features")
Oct.generate_features(wsize)

#%% Compute


