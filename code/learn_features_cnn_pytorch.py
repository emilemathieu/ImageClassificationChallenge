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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X_train_tensor = torch.FloatTensor(X_train).view(-1, 3, 32, 32)
y_train_tensor = torch.LongTensor(y_train)
X_train = X_train_tensor.numpy()
y_train = y_train_tensor.numpy()

X_test_tensor = torch.FloatTensor(X_test).view(-1, 3, 32, 32)
y_test_tensor = torch.LongTensor(y_test)
X_test = X_test_tensor.numpy()
y_test = y_test_tensor.numpy()
#%%

class Net_deep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
              torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=2,stride=2),
              torch.nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=2),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=2,stride=2),
              torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=2,stride=2)
            )
        
        self.classifier = torch.nn.Sequential(
              torch.nn.Linear(20*4*4, 10),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 20*4*4)
        x = self.classifier(x)
        return x

#%%
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
              torch.nn.Conv2d(3, 6, 5),
              torch.nn.ReLU(),
              #torch.nn.Dropout2d(p=0.2),
              torch.nn.MaxPool2d(2,2),
              torch.nn.Conv2d(6, 16, 5),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(2,2)
            )
        
        self.classifier = torch.nn.Sequential(
              #torch.nn.Dropout(p=0.2),
              torch.nn.Linear(16*5*5, 10),
              #torch.nn.Linear(16*5*5, 120),
              #torch.nn.ReLU(),
              #torch.nn.Linear(120, 84),
              #torch.nn.ReLU(),
              
              #torch.nn.Linear(84, 10)
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x
#%%
net = Net()
net

#%%

criterion = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adagrad(net.parameters())
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#optimizer = torch.optim.RMSprop(net.parameters())

N = X_train_tensor.size()[0]
batch_size = 8
nb_batchs = int(N / batch_size)
for epoch in range(20): # loop over the dataset multiple times
    
    running_loss = 0.0
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
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 100))
            running_loss = 0.0
        #break
            

print('Finished Training')

#%% Score on training and test datasets
correct_train = 0
total_train = 0
N = X_train_tensor.size()[0]
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = X_train_tensor[i*batch_size:(i+1)*batch_size,:]
    labels = y_train_tensor[i*batch_size:(i+1)*batch_size]
    outputs = net(torch.autograd.Variable(inputs))
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
    outputs = net(torch.autograd.Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum()

print('Accuracy -- Train: {} | Test: {}'.format(
        round(100*correct_train/total_train,2), round(100*correct_test/total_test,2)))

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