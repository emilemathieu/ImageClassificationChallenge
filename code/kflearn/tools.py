# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:28:07 2017

@author: Thomas PESNEAU
TOOL FUNCTIONS FOR FEATURE EXTRACTION
"""
import numpy as np
import random
from scipy.sparse import csc_matrix

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
        image = np.reshape(X[im_no,:],tuple(dim),'F')
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
    
def Kmeans(patches,nb_centroids,nb_iter,*args):
    patches = patches.copy()
    x2 = np.sum(patches**2,axis=1)
    x2 = x2.reshape((len(x2),1))
#    print("x2 {}".format(x2[0:10]))
    if(args):
        centroids = args[0].copy()
    else:
        centroids = np.random.normal(size=(nb_centroids,patches.shape[1])) * 0.1## initialize the centroids at random
#    print("centroids {}".format(centroids[0:5,0:5]))
    sbatch = 1000
#    input("Enter iterations...")
    for i in range(nb_iter):
        patches = patches.copy()
        print("K-means: {} / {} iterations".format(i+1,nb_iter))
        centroids = centroids.copy()
        c2 = 0.5 * np.sum(centroids**2,axis=1)
        c2 = c2.reshape((len(c2),1))
#        print("c2 {} ({})".format(c2[0:10],i))
        sum_k = np.zeros((nb_centroids,patches.shape[1])).copy()## dictionnary of patches
        compt = np.zeros(nb_centroids)## number of samples per clusters
        compt = compt.reshape((len(compt),1))
        loss = 0
        ## Batch update
        for j in range(0,patches.shape[0],sbatch):
            last = min(j+sbatch,patches.shape[0])
            m = last - j
#            print("last, m {}, {} ({})".format(last,m,i))
            diff = np.dot(centroids,patches[j:last,:].transpose()) - c2## difference of distances
            labels = np.argmax(diff,axis=0)## index of the centroid for each sample
#            print("labels {} ({})".format(labels[0:10],i))
            max_value = np.max(diff,axis=0)## maximum value for each sample
            max_value = max_value.reshape((len(max_value),1))
            loss += np.sum(0.5*x2[j:last,:] - max_value)
            ## Use sparse matrix
            rows = np.arange(m)
            data = np.ones(m)
            S= csc_matrix((data,(rows,labels)),shape=(m,nb_centroids))
            S_ = csc_matrix((data,(labels,rows)),shape=(nb_centroids,m))
            #S = np.zeros((m,nb_centroids))
            ## Put the label of each sample in a sparse indicator matrix
            ## S(i,labels(i)) = 1, 0 elsewhere
#            for ind in range(m):
#                S[ind,labels[ind]] = 1    
            sum_k = sum_k + S_.dot(patches[j:last,:])## update the dictionnary
#            print("sum_k {} ({})".format(sum_k[0:3,0:3],i))
            sumS = np.sum(S,axis=0)
            sumS = sumS.reshape((sumS.size,1))
            compt += sumS## update the number of samples per centroid in the batch
#            print("compt {} ({})".format(compt[0:10],i))
#            input("iteration {}, batch {}, keep going...".format(i,j))
            
        centroids = np.divide(sum_k,compt).copy()## Normalise the dictionnary, will raise a RunTimeWarning if compt has zeros
        print(compt.dtype)                       ## this situation is dealt with in the two following lines  
        print(sum_k.dtype)
#        print("centroids {} ({})".format(centroids[0:5,0:5],i))
#        input("#{} iteration end, keep going...".format(i))
        #badCentroids = np.where(compt == 0)## Find the indices of empty clusters
        ## Find indices for which compt[i] == 0
        indices = []
        for index in range(len(compt)):
            if(compt[index]==0):
                indices.append(index)
        if(len(indices) != 0):
            for index in indices:
                centroids[index,:] = 0## in the case where a cluster is empty, set the centroid to 0 to avoid NaNs
        #centroids[tuple(badCentroids),:] = 0## in the case where a cluster is empty, set the centroid to 0 to avoid NaNs
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
        if(i % 100 == 0):
            print("Feature extraction: {} / {}".format(i,nb_samples))
        ## Extract patches
#        Xi = X[i,:]
#        print("Xi {}".format(Xi.shape))
#        Xi = Xi.reshape(tuple(dim))
#        print("block patch")
#        print("Xi {}".format(Xi.shape))
#        patches = block_patch(Xi,rfSize,stride)
        patches1 = block_patch(X[i,0:1024],rfSize,stride)
        patches2 = block_patch(X[i,1024:2048],rfSize,stride)
        patches3 = block_patch(X[i,2048:],rfSize,stride)
        patches = np.concatenate((patches1,patches2,patches3),axis=0)
        patches = patches.transpose()
#        input("Keep going...")
        ## Pre-process patches
#        print("Preprocessing")
        patches = pre_process(patches,eps)
#        input("Keep going...")
        ## Whitening (optional)
#        print("Whitening")
        if(args):
            M = args[0]
            P = args[1]
            patches = patches.transpose() - M
            patches = patches.transpose()
            patches = np.dot(patches,P)
#        print("patches {}".format(patches.shape))
#        input("Keep going...")
        ## Activation function (soft Kmeans assignement)
#        print("Activation function")
        n_patches = np.sum(patches**2,axis=1)
        n_patches = n_patches.reshape((len(n_patches),1))
        n_centroids = np.sum(centroids**2,axis=1)
        n_centroids = n_centroids.reshape((1,len(n_centroids)))
        CvsP = np.dot(patches,centroids.transpose())
#        print("npatches {} (should be 729)".format(n_patches.shape))
#        print("ncentroids {} (should be 1500)".format(n_centroids.shape))
#        print("CvsP {} (should be 729*1500)".format(CvsP.shape))
        distance = -2*CvsP+n_patches
        distance = n_centroids + distance
        distance = np.sqrt(distance)## z in the article
#        print("Distance {} (should be 729 * 1500)".format(distance.shape))
#        min_dist = np.min(distance,axis=0)
#        labels = np.argmin(distance,axis=0)
        mu = np.mean(distance,axis=1)## average distance to centroids for each patch
#        print("mu {} (should be 729)".format(mu.shape))
        mu = mu.reshape((len(mu),1))
        activation = -(distance - mu)
        activation[activation <= 0] = 0
#        print("Activation {} (should be 729*1600)".format(activation.shape))
#        print("Activation {}".format(activation))
#        input("Keep going...")
        
        
        ## Reshape activation
#        print("Reshape activation")
        rows = dim[0] - rfSize + 1
        cols = dim[1] - rfSize + 1
#        print("rows {}".format(rows))
#        print("cols {}".format(cols))
#        print("activation {}".format(activation.shape))
        activation = activation.reshape((rows,cols,nb_centroids))
#        input("Keep going...")
        ## Pooling over 4 quadrants of the image to reduce number of features
#        print("Max pooling")
        quad_x = round(rows / 2)
        quad_y = round(cols / 2)
        # up left quadrant
#        print("quadrant {}".format(activation[0:quad_x,0:quad_y,:].shape))
        q1 = np.sum(activation[0:quad_x,0:quad_y,:],axis=0)
        q1 = np.sum(q1,axis=0)
#        print("q1 {} (should be 1500)".format(q1.shape))
        q1 = q1.reshape((1,nb_centroids))
        # up right quadrant
        q3 = np.sum(activation[quad_x:,0:quad_y,:],axis=0)
        q3 = np.sum(q3,axis=0)
        q3 = q3.reshape((1,nb_centroids))
        # bottom left quadrant
        q2 = np.sum(activation[0:quad_x,quad_y:,:],axis=0)
        q2 = np.sum(q2,axis=0)
        q2 = q2.reshape((1,nb_centroids))
        # bottom right quadrant
        q4 = np.sum(activation[quad_x:,quad_y:,:],axis=0)
        q4 = np.sum(q4,axis=0)
        q4 = q4.reshape((1,nb_centroids))
        ## Get feature vector from max pooling
        Q = np.concatenate((q1,q2,q3,q4),axis=1)
#        input("Keep going...")
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
    X_standard = X - mean
    X_standard = np.divide(X_standard,var)
    X_standard = np.concatenate((X_standard,np.ones((X_standard.shape[0],1))), axis=1)
    return X_standard
    
    
    
def block_patch(x,psize,stride):
    """
    Extract patches from a block
    ------------
    Parameters
        x: numpy array, flatten image
        psize: int, size of a patch
        stride: int, step between each patch
    """
    patches = np.zeros((psize*psize,1))
    image = x.reshape((32,32))
    for i in range(0,image.shape[0]-psize+1,stride):
        for j in range(0,image.shape[1]-psize+1,stride):
            patch = image[i:i+psize,j:j+psize]
            #print("patch {}".format(patch.shape))
            patch = patch.reshape((psize*psize,1))
            patches = np.concatenate((patches,patch),axis=1)
    return patches[:,1:]
            
            
    


