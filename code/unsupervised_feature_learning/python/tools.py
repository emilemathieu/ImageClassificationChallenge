# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:28:07 2017

@author: Thomas PESNEAU
TOOL FUNCTIONS FOR FEATURE EXTRACTION
"""
import numpy as np
import random
import time

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
    return patches,M,P
    
def Kmeans(patches,nb_centroids,nb_iter):
    x2 = patches**2
    x2 = np.sum(x2,axis=1)
    x2 = x2.reshape((len(x2),1))
    centroids = np.random.normal(size=(nb_centroids,patches.shape[1])) * 0.1## initialize the centroids at random
    sbatch = 1000
    
    for i in range(nb_iter):
        print("K-means: {} / {} iterations".format(i,nb_iter))
        c2 = 0.5 * np.sum(centroids**2,axis=1)
        c2 = c2.reshape((len(c2),1))
        sum_k = np.zeros((nb_centroids,patches.shape[1]))## dictionnary of patches
        compt = np.zeros(nb_centroids)## number of samples per clusters
        compt = compt.reshape((len(compt),1))
        loss = 0
        ## Batch update
        for j in range(0,sbatch,patches.shape[0]):
            last = min(j+sbatch,patches.shape[0])
            m = last - j
            diff = np.dot(centroids,patches[j:last,:].transpose()) - c2## difference of distances
            labels = np.argmax(diff,axis=0)## index of the centroid for each sample
            max_value = np.max(diff,axis=0)## maximum value for each sample
            loss += np.sum(0.5*x2[j:last,:] - max_value)
            S = np.zeros((m,nb_centroids))
            ## Put the label of each sample in a sparse indicator matrix
            ## S(i,labels(i)) = 1, 0 elsewhere
            for ind in range(m):
                S[ind,labels[ind]] = 1    
            sum_k += np.dot(S.transpose(),patches[j:last,:])## update the dictionnary
            sumS = np.sum(S,axis=0)
            sumS = sumS.reshape((len(sumS),1))
            compt += sumS## update the number of samples per centroid in the batch
            
        centroids = np.divide(sum_k,compt)## Normalise the dictionnary, will raise a RunTimeWarning if compt has zeros
                                          ## this situation is dealt with in the two following lines  
        badCentroids = np.where(compt == 0)## Find the indices of empty clusters
        centroids[tuple(badCentroids),:] = 0## in the case where a cluster is empty, set the centroid to 0 to avoid NaNs
    return centroids
                

def extract_features(X,centroids,rfSize,dim,stride,eps,*args):
    """
    Extract the features of an input image based on the centroids
    -------------
    Parameters:
        X: numpy array, nb_samples x nb_elements
        centroids: numpy array, nb_centroids x size_patch
        rfSize: size of a patch
        dim: dimension of images in the dataset
        stride: step between each patch
        eps: normalization constant
        *args: optional arguments for whitening
    """
    ## Check number of inputs
    nb_centroids = centroids.shape[0]
    nb_samples = X.shape[0]
    Features = np.zeros((nb_samples, 4*nb_centroids))
    for i in range(nb_samples):
        if(i % 1000 == 0):
            print("Feature extraction: {} / {}".format(i,nb_samples))
        ## Extract patches
        Xi = X[i,:]
        Xi = Xi.reshape(tuple(dim))
        patches = block_patch(Xi,rfSize,stride)
        ## Pre-process patches
        pre_process(patches,eps)
        ## Whitening (optional)
        if(args):
            M = args[0]
            P = args[1]
            patches = patches.transpose() - M
            patches = patches.transpose()
            patches = np.dot(patches,P)
        ## Activation function (soft Kmeans assignement)
        n_patches = np.sum(patches**2,axis=1)
        n_centroids = np.sum(centroids**2,axis=1)
        CvsP = np.dot(patches,centroids.transpose())
        distance = n_patches - 2*CvsP
        distance = n_centroids + distance
        distance = np.sqrt(distance)## z in the article
#        min_dist = np.min(distance,axis=0)
#        labels = np.argmin(distance,axis=0)
        mu = np.mean(distance,axis=1)## average distance to centroids for each patch
        activation = mu - distance
        activation[activation <= 0] = 0
        ## Reshape patches
        rows = dim[0] - rfSize
        cols = dim[1] - rfSize
        patches = patches.reshape((rows,cols,nb_centroids))
        ## Pooling over 4 quadrants of the image to reduce number of features
        quad_x = round(rows / 2)
        quad_y = round(cols / 2)
        # up left quadrant
        q1 = np.sum(patches[0:quad_x,0:quad_y,:],axis=0)
        q1 = np.sum(q1)
        q1 = q1.reshape((1,nb_centroids))
        # up right quadrant
        q2 = np.sum(patches[quad_x:,0:quad_y,:],axis=0)
        q2 = np.sum(q2)
        q2 = q2.reshape((1,nb_centroids))
        # bottom left quadrant
        q3 = np.sum(patches[0:quad_x,quad_y:,:],axis=0)
        q3 = np.sum(q3)
        q3 = q3.reshape((1,nb_centroids))
        # bottom right quadrant
        q4 = np.sum(patches[quad_x:,quad_y:,:],axis=0)
        q4 = np.sum(q4)
        q4 = q4.reshape((1,nb_centroids))
        ## Get feature vector from max pooling
        Q = np.concatenate((q1,q2,q3,q4),axis=1)
        Features[i,:] = Q
    return Features
        
    
def standard(X):
    """
    Subtract the mean to each row and divide by the standard deviation
    ---------------
    Parameters:
        X: multidimensional numpy array
    """
    mean = np.mean(X,axis=0)
    var = np.var(X,axis=0)+0.01
    var = np.sqrt(var)
    X_standard = X.transpose() - mean
    X_standard = X_standard.transpose()
    X_standard = np.divide(X_standard,var)
    X_standard = np.concatenate((X_standard,np.ones((X_standard.shape[0],1))), axis=1)
    return X_standard
    
    
    
def block_patch(image,psize,stride):
    """
    Extract patches of size psize x psize in an image
    ---------------
    Parameters:
        image: multidimensional numpy array
        psize: size of the square patches
        stride: space between each patch
    """
    patches = np.zeros((1,psize*psize*3))
    width = image.shape[0]
    height = image.shape[1]
    channels = image.shape[3]
    for i in range(0,stride,width-psize):
        for j in range(0,stride,height-psize):
            patch = np.zeros((1,psize*psize*3))
            for c in range(channels):
                patch_c = image[i:i+psize,j:j+psize]
                patch_c = patch_c.reshape((1,psize*psize))
                patch[:,c*psize*psize:(c+1)*psize*psize] = patch_c
            patches = np.concatenate((patches,patch),axis=0)
    patches = patches[1:,:]
    return patches
            
            
    


