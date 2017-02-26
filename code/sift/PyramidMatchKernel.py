# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:18:31 2017

@author: Thomas PESNEAU
Implementation of the pyramid match kernel
"""
import numpy as np
import math


def sub2ind(A,sub):
    """
    Get the linear index from the matrix coordinates
    -----------
    Parameters:
        A: numpy array
        sub: matrix coordinates
    """
    if(len(A.shape) != len(sub)):
        raise ValueError("Dimensions mismatch")
    else:
        num_elements = len(A.flatten())
        vec = np.arange(num_elements)
        vec = vec.reshape(A.shape)
        ind = vec[tuple(sub)]
        return ind

def pretreat(X,inter=1):
    """
    Returns the set X with minimal distance between vectors equal to inter
    -----------
    Parameters:
        X: set of samples
            list of vectors of size num_descriptors x d
        inter: minimal distance between features
            int
    """
    feature_space = X[0]
    for i in range(1,len(X)):
        descriptor = X[i]
        feature_space = np.concatenate((feature_space,descriptor), axis=0)
    ## feature_space.shape = tot_num_descriptors x d
    ## Compute difference between all pairs of feature_space
    diff_matrix = np.zeros((feature_space.shape[0]))
    for i in range(feature_space.shape[0]):
        for j in range(feature_space.shape[0]):
            diff_matrix[i,j] = np.norm(feature_space[i] - feature_space[j])
    diff_matrix[diff_matrix == 0] = 10e5 ## avoid values 0 for non unique points
    min_dist = np.min(diff_matrix)
    feature_space = (feature_space / min_dist) * inter
    ## D is the maximum range of the elements of X
    D = np.max(feature_space)
    return X, D
    

class Histogram(object):
    def __init__(self,x,index,D,inter_dist=1):
        """
        Parameters:
            x: one set of features num_features x dim_features (ex: set of SIFT descriptors from an image)
            index: index of the histogram in the pyramid, useful for bin side lenght value
        """
        self.x = self.x
        self.index = index
        self.D = D
        self.d = x.shape[1]
        #self.inter = inter_dist
    
    def build_hist(self):
        side_length = 2**self.index
        dimension = (self.D / side_length)**self.d
        H = np.zeros((1,dimension))
        dim_list = [(self.D / side_length)] * self.d
        for i in range(self.x.shape[0]):
            descriptor = self.x[i,:]
            ld = descriptor / side_length
            ld = np.floor(ld).astype(int)
            bin_id = sub2ind(np.zeros(tuple(dim_list)),ld) ## get the bin corresponding to the descriptor
            H[bin_id] += 1
        return H

class FeatureExtraction(object):
    def __init__(self,x,D):
        self.L = math.log(D,2)+1
        self.x = x
        self.features = []
        self.D
        
    def extract(self):
        for i in range(self.L):
            h_ = Histogram(self.x, i, self.D)
            H = h_.build_hist()
            self.features.append(H)
            
def I(A,B):
    """
    Computes the overlap between histograms
    --------------
    Parameters:
        A,B: histograms
    """
    overlap = 0
    for j in range(len(A)):
        overlap += min(A[j], B[j])
    return overlap

def PM_kernel(x, y, D):
    """
    Computes the Pyramid Match Kernel from two lists of histograms
    -------------
    Parameters:
        x,y: input points
        D: maximum range of element values of the input space
    """
    assert(len(x) == len(y))
    d = len(x)
    feat1 = FeatureExtraction(x,D)
    feat2 = FeatureExtraction(y,D)
    feat1.extract()
    feat2.extract()
    Psi_x = feat1.features
    Psi_y = feat2.features
    assert(len(Psi_x) == len(Psi_y))
    score = 0
    L = len(Psi_x)
    for i in range(L):
        if(i==0):
            score += (1 / (d*2**i)) * I(Psi_x[i], Psi_y[i])
        else:
            score += (1 / (d*2**i)) * (I(Psi_x[i], Psi_y[i]) - I(Psi_x[i-1], Psi_y[i-1]))
    return score
    
        
        
        
            
            
            
        
    
        
        


