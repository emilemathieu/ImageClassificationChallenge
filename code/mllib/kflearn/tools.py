import numpy as np

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

    var_patches = np.var(patches, axis=1, ddof=1)
    var_patches = var_patches.reshape((len(var_patches),1))
    var_patches = np.sqrt(var_patches + eps)

    patches = patches - mean_patches
    patches = np.divide(patches,var_patches)
    return patches

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
        row = np.random.randint(0, dim[0] - rfSize + 1)
        col = np.random.randint(0, dim[1] - rfSize + 1)
        # Crop random patch
        image = X[im_no,:].reshape(dim[2], dim[0], dim[1])
        patch = image[:,row:row+rfSize, col:col+rfSize]
        patches[i,:] = patch.flatten()
    return patches

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

            patch = patch.reshape((psize*psize,1))
            patches = np.concatenate((patches,patch),axis=1)
    return patches[:,1:]