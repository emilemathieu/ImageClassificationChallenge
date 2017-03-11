import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from mllib.im2col_cyt import im2col_cython
from .tools import pre_process

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
    n_centroids = np.sum(centroids**2, axis=1, keepdims=True).T
    for i in range(nb_samples):
        if(i % 100 == 0):
            print("Feature extraction: {} / {}".format(i,nb_samples))
        ## Extract patches
        patches = im2col_cython(X[i,:].reshape(1,3,32,32), rfSize, rfSize, padding=0, stride=stride).T
        ## Pre-process patches
        patches = pre_process(patches,eps)
        ## Whitening (optional)
        if(args):
            M = args[0]
            P = args[1]
            patches = patches.transpose() - M
            patches = patches.transpose()
            patches = np.dot(patches,P)

        ## Activation function (soft Kmeans assignement)
        n_patches = np.sum(patches**2, axis=1, keepdims=True)
        CvsP = np.dot(patches,centroids.transpose())
        distance = np.sqrt(-2 * CvsP + n_patches + n_centroids)## z in the article
        mu = np.mean(distance, axis=1, keepdims=True)## average distance to centroids for each patch
        activation = - (distance - mu)
        activation[activation <= 0] = 0
        
        ## Reshape activation
        rows = dim[0] - rfSize + 1
        cols = dim[1] - rfSize + 1

        activation = activation.reshape((rows,cols,nb_centroids))
        ## Pooling over 4 quadrants of the image to reduce number of features
        quad_x = round(rows / 2)
        quad_y = round(cols / 2)
        # up left quadrant
        q1 = np.max(activation[0:quad_x,0:quad_y,:],axis=(0,1)).flatten()
        # up right quadrant
        q3 = np.max(activation[quad_x:,0:quad_y,:],axis=(0,1)).flatten()
        # bottom left quadrant
        q2 = np.max(activation[0:quad_x,quad_y:,:],axis=(0,1)).flatten()
        # bottom right quadrant
        q4 = np.max(activation[quad_x:,quad_y:,:],axis=(0,1)).flatten()
        ## Get feature vector from max pooling
        Features[i,:] = np.concatenate((q1,q2,q3,q4))
    return Features

def extract_features_2(X,centroids,rfSize,dim,stride,eps,*args):
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

    patches = im2col_cython(X.reshape(-1,3,32,32), rfSize, rfSize, padding=0, stride=stride).T
    patches = pre_process(patches,eps)
    print('patches', patches.shape)
    if(args):
        M = args[0]
        P = args[1]
        patches = patches.T - M
        patches = patches.T
        patches = np.dot(patches,P)

    ## Activation function (soft Kmeans assignement)
    n_patches = np.sum(patches**2, axis=1, keepdims=True)
    print('n_patches', n_patches.shape)
    n_centroids = np.sum(centroids**2, axis=1, keepdims=True).T
    print('n_centroids', n_centroids.shape)
    CvsP = np.dot(patches,centroids.T)
    print('CvsP', CvsP.shape)
    distance = np.sqrt(-2 * CvsP + n_patches + n_centroids)## z in the article
    print('distance', distance.shape)
    mu = np.mean(distance, axis=1, keepdims=True)## average distance to centroids for each patch
    print('mu', mu.shape)
    activation = - (distance - mu)
    activation[activation <= 0] = 0
    print('activation', activation.shape)
    ## Reshape activation
    rows = dim[0] - rfSize + 1
    cols = dim[1] - rfSize + 1

    activation = activation.reshape((-1, rows,cols,nb_centroids))
    print('activation', activation.shape)
    ## Pooling over 4 quadrants of the image to reduce number of features
    quad_x = round(rows / 2)
    quad_y = round(cols / 2)
    # up left quadrant
    q1 = np.max(activation[:,0:quad_x,0:quad_y],axis=(1,2))
    print('q1',q1.shape)
    # up right quadrant
    q3 = np.max(activation[:,quad_x:,0:quad_y],axis=(1,2))
    # bottom left quadrant
    q2 = np.max(activation[:,0:quad_x,quad_y:],axis=(1,2))
    # bottom right quadrant
    q4 = np.max(activation[:,quad_x:,quad_y:],axis=(1,2))
    ## Get feature vector from max pooling
    #Features[i,:] = np.concatenate((q1,q2,q3,q4))
    return np.hstack((q1,q2,q3,q4))