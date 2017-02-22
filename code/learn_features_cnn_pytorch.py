#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:58:48 2017

@author: EmileMathieu
"""
import torch

#%%
# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    
# print images
#imshow()

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
X = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
X = X * 2
y = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train_tensor = torch.FloatTensor(X_train).view(-1, 3, 32, 32)
y_train_tensor = torch.LongTensor(y_train)
X_train = X_train_tensor.numpy()
y_train = y_train_tensor.numpy()

X_test_tensor = torch.FloatTensor(X_test).view(-1, 3, 32, 32)
y_test_tensor = torch.LongTensor(y_test)
X_test = X_test_tensor.numpy()
y_test = y_test_tensor.numpy()

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
#X_train_tensor = X_train_tensor[:,:,4:28,4:28]
#X_test_tensor = X_test_tensor[:,:,4:28,4:28]
#
#from torch.legacy.nn import SpatialCrossMapLRN
#
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
#        
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
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
              torch.nn.Conv2d(3, 6, 5),
              torch.nn.ReLU(),
              #torch.nn.Dropout2d(p=0.2),
              torch.nn.MaxPool2d(2,2),
              torch.nn.Conv2d(6, 12, 5),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(2,2),
            )
        
        self.classifier = torch.nn.Sequential(
              
              torch.nn.Linear(12*5*5, 10),
              #torch.nn.Linear(16*5*5, 84),
              #torch.nn.ReLU(),
              #torch.nn.Linear(84, 10)                
                              
              #torch.nn.Linear(16*5*5, 120),
              #torch.nn.ReLU(),
              #torch.nn.Linear(120, 84),
              #torch.nn.ReLU(),
              
              #torch.nn.Linear(84, 10)
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 12*5*5)
        x = self.classifier(x)
        return x
#%%
net = Net()
net
#outputs = net(torch.autograd.Variable(X_train_tensor[0:1,:]))

#%%
import timeit

criterion = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adagrad(net.parameters())
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#optimizer = torch.optim.RMSprop(net.parameters())

start_global = timeit.default_timer()
N = X_train_tensor.size()[0]
batch_size = 8
nb_batchs = int(N / batch_size)
for epoch in range(0, 40, 1): # loop over the dataset multiple times
    
    running_loss = 0.0
    start = timeit.default_timer()
    #for i in range(nb_batchs):
    for i in np.random.permutation(nb_batchs):
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
    if (epoch + 1) % 5 == 0:
        score(net)            
            
print('Finished Training | {} seconds'.format(round(timeit.default_timer() - start_global, 2)))

#%%
#import torchvision.models as models
features_net = torch.nn.Sequential()

for i, layer in enumerate(list(net.features)):
    print(i, type(layer), layer.__str__())
    features_net.add_module("L{}_{}".format(i+1,layer.__str__()),layer)

X_tensor = torch.FloatTensor(X).view(-1, 3, 32, 32)
X_features = features_net.forward(torch.autograd.Variable(X_tensor))
X_features = X_features.view(-1,16*5*5)
X_features = X_features.data.numpy()
pd.DataFrame(X_features).to_csv('../data/Xtr_features_cnn.csv',header=False, index=False)