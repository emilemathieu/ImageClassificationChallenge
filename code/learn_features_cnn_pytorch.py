#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:58:48 2017

@author: EmileMathieu
"""
import torch
import numpy as np
import pandas as pd
#%%

X = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
#X = X * 2
#X = preprocess(X)
y = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#X_train = X
#y_train = y
#%%
N = 5000
permutation = np.random.permutation(N)
split = 3500
train_idx = permutation[0:split]
test_idx = permutation[split:]

new_N = X.shape[0]
ratio = int(new_N / N)
X_train = np.empty((split*ratio,3072), dtype=float)
y_train = np.empty((split*ratio), dtype=int)
#X_test = np.empty(((N-split)*ratio,3072), dtype=float)
#y_test = np.empty(((N-split)*ratio), dtype=int)
for i in range(ratio):
    print(i)
    X_train[i*split:(i+1)*split,:] = X[i*N+train_idx,:]
    y_train[i*split:(i+1)*split] = y[i*N+train_idx]
    #X_test[i*(N-split):(i+1)*(N-split),:] = X[i*N+test_idx,:]
    #y_test[i*(N-split):(i+1)*(N-split)] = y[i*N+test_idx]

y_test = y[test_idx]
X_test = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
X_test = X_test[test_idx]
#%%
X_train_tensor = torch.FloatTensor(X_train).view(-1, 3, 32, 32)
y_train_tensor = torch.LongTensor(y_train)
#X_train = X_train_tensor.numpy()
#y_train = y_train_tensor.numpy()

X_test_tensor = torch.FloatTensor(X_test).view(-1, 3, 32, 32)
y_test_tensor = torch.LongTensor(y_test)
#X_test = X_test_tensor.numpy()
#y_test = y_test_tensor.numpy()
del X_train, y_train, X_test, y_test
#%% Score on training and test datasets
def score(algo):
    correct_train = 0
    total_train = 0
    N = X_train_tensor.size()[0]
    nb_batchs = int(N / batch_size)
    for i in range(nb_batchs):
        inputs = X_train_tensor[i*batch_size:(i+1)*batch_size,:]
        labels = y_train_tensor[i*batch_size:(i+1)*batch_size]
        outputs = algo(torch.autograd.Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum()
    
    correct_test = 0
    total_test = 0
    N = X_test_tensor.size()[0]
    nb_batchs = int(N / batch_size)
    for i in range(nb_batchs):
        inputs = X_test_tensor[i*batch_size:(i+1)*batch_size,:]
        labels = y_test_tensor[i*batch_size:(i+1)*batch_size]
        outputs = algo(torch.autograd.Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum()
    
    print('Accuracy -- Train: {} | Test: {}'.format(
            round(100*correct_train/total_train,2), round(100*correct_test/total_test,2)))

#%%
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
              torch.nn.Conv2d(3, 6, 5),
              torch.nn.BatchNorm2d(6),
              torch.nn.ReLU(),
              #torch.nn.Dropout2d(p=0.2),
              torch.nn.MaxPool2d(2,2),
              torch.nn.Conv2d(6, 16, 5),
              torch.nn.BatchNorm2d(16),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(2,2),
            )
        
        self.middle = torch.nn.Sequential(
                torch.nn.Linear(16*5*5, 120),
                )
        
        self.classifier = torch.nn.Sequential(
              
              torch.nn.Linear(16*5*5, 10),
              #torch.nn.Linear(16*5*5, 84),
              #torch.nn.ReLU(),
              #torch.nn.Linear(84, 10)                
                              
              #torch.nn.Linear(16*5*5, 120),
              #torch.nn.BatchNorm1d(120),
              #torch.nn.ReLU(),
              #torch.nn.Linear(120, 84),
              #torch.nn.BatchNorm1d(84),
              #torch.nn.ReLU(),
              #torch.nn.Linear(84, 10)
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        #x = self.middle(x)
        x = self.classifier(x)
        return x
#%%
net = Net()

#%%
import timeit

criterion = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adagrad(net.parameters())
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
optimizer = torch.optim.RMSprop(net.parameters())

start_global = timeit.default_timer()
N = X_train_tensor.size()[0]
batch_size = 16
nb_batchs = int(N / batch_size)
for epoch in range(0, 15, 1): # loop over the dataset multiple times
    
    running_loss = 0.0
    start = timeit.default_timer()
    suffle = np.random.permutation(N)
    X_train_tensor = torch.FloatTensor(X_train_tensor.numpy()[suffle,:])
    y_train_tensor = torch.LongTensor(y_train_tensor.numpy()[suffle])
    for i in range(nb_batchs):
    #for i in np.random.permutation(nb_batchs):
        # get the inputs
        #inputs, labels = data
        inputs = X_train_tensor[i*batch_size:(i+1)*batch_size,:]
        labels = y_train_tensor[i*batch_size:(i+1)*batch_size]
        
        # wrap them in Variable
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #print(loss)
        loss.backward()        
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99: # print every 2000 mini-batches
            print('[{}, {}] - loss: {} | time: '.format(
                    epoch+1, i+1, round(running_loss / 100, 3)),
                    round(timeit.default_timer() - start, 2))
            running_loss = 0.0
            start = timeit.default_timer()
    if (epoch + 1) % 1 == 0:
        score(net)            
            
print('Finished Training | {} seconds'.format(round(timeit.default_timer() - start_global, 2)))

#%%
X_out = Z.transpose(0, 3, 1, 2)
#X_out = X.reshape(-1, 3, 32, 32)
N = X_out.shape[0]
nb_features = 16*5*5
X_features = np.empty((N, nb_features))
batch_size = 8
nb_batchs = int(N / batch_size)
#X = X.reshape(-1,3,32,32)
for i in range(nb_batchs):
    inputs = X_out[i*batch_size:(i+1)*batch_size,:]
    inputs = torch.autograd.Variable(torch.FloatTensor(inputs))
    outputs = net.features.forward(inputs)
    #outputs = outputs.view(-1, 16*5*5)
    #outputs = net.middle(outputs)
    X_features[i*batch_size:(i+1)*batch_size, :] = outputs.data.numpy()#.reshape(-1,nb_features)

pd.DataFrame(X_features).to_csv('../data/cnn/Xtr_features_cnn_dataaugm.csv',header=False, index=False)
#pd.DataFrame(y).to_csv('../data/cnn/Ytr_features_cnn_dataaugm_test2.csv',header=True, index=True)

#%%
#X_train_tensor = X_train_tensor[:,:,4:28,4:28]
#X_test_tensor = X_test_tensor[:,:,4:28,4:28]
#from torch.legacy.nn import SpatialCrossMapLRN
#class TFNet(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.features = torch.nn.Sequential(
#                torch.nn.Conv2d(3, 64, 5, stride=1, padding=2),#Conv1
#                torch.nn.ReLU(),
#                torch.nn.MaxPool2d(3, stride=2),#Pool1
#                #SpatialCrossMapLRN(4, 0.001 / 9.0, 0.75, 1),#norm1
#                torch.nn.Conv2d(64, 64, 5, stride=1, padding=2),#Conv2
#                torch.nn.ReLU(),
#                #SpatialCrossMapLRN(4, 0.001 / 9.0, 0.75, 1),#norm2
#                torch.nn.MaxPool2d(3, stride=2)#pool2
#            )    
#        self.classifier = torch.nn.Sequential(
#               torch.nn.Linear(64*5*5,384),#local3
#               torch.nn.ReLU(),
#               torch.nn.Linear(384,192),#local4
#               torch.nn.ReLU(),
#               torch.nn.Linear(192,10)
#            )
#    def forward(self, x):
#        x = self.features(x)
#        x = x.view(-1, 64*5*5)
#        x = self.classifier(x)
#        return x


#%%
import matplotlib.pyplot as plt
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1,2,0)))
    
# print images
#imshow()
#%%
def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/
    X: numpy 2d array
        input data, rows are data points, columns are features
    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5
    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white

def scale(X):
    X -= np.mean(X, axis = 0)

def preprocess(X):
    # Center
    #X -= np.mean(X, axis = 0)
    # Normalize
    #X /= np.sqrt(np.var(X, axis = 0) + 0.0)
    # Whitening
    X = zca_whiten(X)
    return X