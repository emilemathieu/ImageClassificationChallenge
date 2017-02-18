# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:34:05 2017

@author: Thomas PESNEAU
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pandas as pd
import data_augmenting_tools as dtools
#################################### DATA AUGMENTING SCRIPT ####################################

## Open data
IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

X = pd.read_csv('../data/Xtr.csv', header=None)
X = X.as_matrix()
X = X[:, 0:-1]

Y = pd.read_csv('../data/Ytr.csv')
Y = Y.as_matrix()

## Apply transformation on each image
augmented_X = np.zeros((5 * X.shape[0], IMAGE_SIZE*IMAGE_SIZE*3))
augmented_Y = np.zeros((5 * Y.shape[0], 2))
compt = 0
for index in range(X.shape[0]):
    if(index%500 == 0):
        print("{} %".format((index // 500)*10))
    image = X[index,:]
    image = image.reshape((3, CHANEL_SIZE))
    image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
    image = image.swapaxes(0,1)
    image = image.swapaxes(1,2)
    
    ## Rotate 90, 180, 270
    image_rotate = np.zeros((3, IMAGE_SIZE*IMAGE_SIZE*3))
    for quarter in range(1,4):
        image_rot = dtools.rotate(image, quarter)
        image_rot = image_rot.reshape((1, IMAGE_SIZE*IMAGE_SIZE*3))
        image_rotate[quarter-1,:] = image_rot
    
    ## Blur image
    image_blur = dtools.blur(image, sigma=0.5)
    image_blur = image_blur.reshape((1, IMAGE_SIZE*IMAGE_SIZE*3))
    
    ## Add noise
    image_noise = dtools.noise(image,sigma=0.01)
    image_noise = image_noise.reshape((1, IMAGE_SIZE*IMAGE_SIZE*3))
    
    ## Save the new images 
    ind = index + compt*4
    ## print("index: {}, ind: {}".format(index,ind))
    augmented_X[ind:ind+3,:] = image_rotate
    augmented_X[ind+3,:] = image_blur
    augmented_X[ind+4,:] = image_noise
    #print("covered: {}".format([ind,ind+1,ind+2,ind+3,ind+4]))
    #raw_input("Press Enter to continue...")
    compt += 1
    
    ## Update labels
    augmented_Y[ind:ind+5,0] = range(ind+5001, ind+5006)
    augmented_Y[ind:ind+5,1] = int(Y[index, 1])
    
## Save in csv file
np.savetxt("../data/augmented_X.csv", augmented_X, delimiter=",")
np.savetxt("../data/augmented_Y.csv", augmented_Y, delimiter=",")
    
