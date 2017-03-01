# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:28:07 2017

@author: Thomas PESNEAU
TOOL FUNCTIONS FOR FEATURE EXTRACTION
"""
import numpy as np
import random

def extract_random_patches(X,nb_patches,rfSize,dim):
    """
    Crop random patches from a set of images
    -----------------
    Parameters:
        X: set of images, numpy array nb_images x nb_elements
        nb_patches: number of patches to extract, int
        rfSize: size of the patches to extract, int
        dim: dimension of the images in the set, list
    """
    N = rfSize * rfSize * 3
    nb_images = X.shape[0]
    patches = np.zeros((nb_patches,N))
    for i in range(nb_patches):
        im_no = i % nb_images
        if(i % 10000 == 0):
            print("Patch extraction: {} / {}".format(i,nb_patches))
        # Draw two random integers
        row = random.randint(0,dim[0] - rfSize)
        col = random.randint(0,dim[1] - rfSize)
        # Crop random patch
        image = X[im_no,:].reshape(tuple(dim))
        patch = image[row:row+rfSize, col:col+rfSize,:]
        patches[i,:] = patch.reshape((1,N))
    return patches
    
def pre_process(patches,eps):
    """
    Pre-process the patches by substracting the mean and dividing by the variance
    --------------
    Parameters:
        patches: collection of patches, numpy array nb_patches x dim_patch
        eps: constant to avoid division by 0
    """
    mean_patches = np.mean(patches, axis=1)
    mean_patches = mean_patches.reshape((len(mean_patches),1))
    #print("size mean: {}".format(mean_patches.shape))
    var_patches = np.var(patches, axis=1)
    var_patches = var_patches.reshape((len(var_patches),1))
    #print("size var: {}".format(var_patches.shape))
    patches = patches - mean_patches
    patches = np.divide(patches,np.sqrt(var_patches + eps))
    return patches
    

def whiten(patches,eps_zca):
    """
    Performs whitening of the patches
    -------------
    Parameters:
        patches: collection of patches, numpy array nb_patches x dim_patch
        eps_zca: zca whitening constant
    """
    C = np.cov(patches,rowvar=False)
    M = np.mean(patches,axis=0)
    M = M.reshape((len(M),1))
    D,V = np.linalg.eig(C)
    D = D.reshape((len(D),1))
    D_zca = np.sqrt(1 / (D + eps_zca))
    D_zca = D_zca.reshape(len(D_zca))
    D_zca = np.diag(D_zca)
    P = np.dot(V,np.dot(D_zca,V.transpose()))
    patches = np.dot((patches.transpose() - M).transpose(),P)
    return patches
    
def Kmeans(patches,nb_centroids,nb_iter):
    x2 = patches**2
    x2 = np.sum(x2,axis=1)
    x2 = x2.reshape((len(x2),1))
    centroids = np.random.normal(size=(nb_centroids,patches.shape[1])) * 0.1
    sbatch = 1000
    
    for i in range(nb_iter):
        print("K-means: {} / {} iterations".format(i,nb_iter))
        c2 = 0.5 * np.sum(centroids**2,axis=1)
        c2 = c2.reshape((len(c2),1))
        sum_k = np.zeros((nb_centroids,patches.shape[1]))
        compt = np.zeros(nb_centroids)
        compt = compt.reshape((len(compt),1))
        loss = 0
        for j in range(0,sbatch,patches.shape[0]):
            last = min(j+sbatch,patches.shape[0])
            m = last - j
#            print("m {}".format(m))
#            input("Keep going...")
            diff = np.dot(centroids,patches[i:last,:].transpose()) - c2
            labels = np.argmax(diff,axis=0)
            max_value = np.max(diff,axis=0)
            loss += np.sum(0.5*x2[i:last,:] - max_value)
            S = np.zeros((m,nb_centroids))
#            print("labels {}".format(labels.shape))
#            print("S {}".format(S.shape))
#            input("Keep going...")
            for ind in range(m):
#                print("ind {}".format(ind))
                S[ind,labels[ind]] = 1    
            sum_k += np.dot(S.transpose(),patches[i:last,:])
            sumS = np.sum(S,axis=0)
            sumS = sumS.reshape((len(sumS),1))
#            print("compt {}".format(compt.shape))
#            print("sumS {}".format(sumS.shape))
            compt += sumS
        centroids = np.divide(sum_k,compt)
        badCentroids = np.where(compt == 0)
        centroids[tuple(badCentroids),:] = 0
    return centroids
                

def extract_features():
    raise NotImplementedError
    
def standard():
    raise NotImplementedError


