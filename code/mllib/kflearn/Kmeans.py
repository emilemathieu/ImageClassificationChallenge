import numpy as np
from scipy.sparse import csc_matrix

def Kmeans(patches,nb_centroids,nb_iter,*args):
    patches = patches.copy()
    x2 = np.sum(patches**2,axis=1)
    x2 = x2.reshape((len(x2),1))

    if(args):
        centroids = args[0].copy()
    else:
        centroids = np.random.normal(size=(nb_centroids,patches.shape[1])) * 0.1## initialize the centroids at random

    sbatch = 1000

    for i in range(nb_iter):
        patches = patches.copy()
        print("K-means: {} / {} iterations".format(i+1,nb_iter))
        centroids = centroids.copy()
        c2 = 0.5 * np.sum(centroids**2,axis=1)
        c2 = c2.reshape((len(c2),1))

        sum_k = np.zeros((nb_centroids,patches.shape[1])).copy()## dictionnary of patches
        compt = np.zeros(nb_centroids)## number of samples per clusters
        compt = compt.reshape((len(compt),1))
        loss = 0
        ## Batch update
        for j in range(0,patches.shape[0],sbatch):
            last = min(j+sbatch,patches.shape[0])
            m = last - j

            diff = np.dot(centroids,patches[j:last,:].transpose()) - c2## difference of distances
            labels = np.argmax(diff,axis=0)## index of the centroid for each sample

            max_value = np.max(diff,axis=0)## maximum value for each sample
            max_value = max_value.reshape((len(max_value),1))
            loss += np.sum(0.5*x2[j:last,:] - max_value)
            ## Use sparse matrix
            rows = np.arange(m)
            data = np.ones(m)
            S= csc_matrix((data,(rows,labels)),shape=(m,nb_centroids))
            S_ = csc_matrix((data,(labels,rows)),shape=(nb_centroids,m))
  
            sum_k = sum_k + S_.dot(patches[j:last,:])## update the dictionnary

            sumS = np.sum(S,axis=0)
            sumS = sumS.reshape((sumS.size,1))
            compt += sumS## update the number of samples per centroid in the batch

            
        centroids = np.divide(sum_k,compt)## Normalise the dictionnary, will raise a RunTimeWarning if compt has zeros
                                                 ## this situation is dealt with in the two following lines  

        ## Find indices for which compt[i] == 0
        indices = []
        for index in range(len(compt)):
            if(compt[index]==0):
                indices.append(index)
        if(len(indices) != 0):
            for index in indices:
                centroids[index,:] = 0## in the case where a cluster is empty, set the centroid to 0 to avoid NaNs
    return centroids