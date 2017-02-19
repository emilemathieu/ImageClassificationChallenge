# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:45:43 2017

@author: Thomas PESNEAU
"""
import numpy as np

def simple_histogram(image):
    """
    image: numpy array
    """
    r = (image[:,:,0]+0.5)*255
    g = (image[:,:,1]+0.5)*255
    b = (image[:,:,2]+0.5)*255
    h_r = [0]*255
    h_g = [0]*255
    h_b = [0]*255
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_bin = int(r[i,j])
            g_bin = int(g[i,j])
            b_bin = int(b[i,j])
            h_r[r_bin] += 1
            h_g[g_bin] += 1
            h_b[b_bin] += 1
    h_r = np.array(h_r)
    h_g = np.array(h_g)
    h_b = np.array(h_b)
    H = np.concatenate((h_r,h_g),axis=0)
    H = np.concatenate((H,h_b),axis=0)
    return H


