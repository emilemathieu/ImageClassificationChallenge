#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:29:39 2017

@author: EmileMathieu
"""

import numpy as np
import scipy.signal
import torch
from importlib import reload
import im2col

class Module(object):
     def __init__(self):
         pass
     def forward(self, X):
         raise NotImplementedError()
     def __call__(self, X):
         return self.forward(X)
     def backward(self, x):
         raise NotImplementedError()
     
     def step(self, optimizer):
         self._bias = optimizer(id(self), 'bias', self._bias, self._grad_bias)
         self._weight = optimizer(id(self), 'weight', self._weight, self._grad_weight)

     def zero_grad(self):
         self._grad_bias = None
         self._grad_weight = None
         
class ReLU(Module):
    def __init__(self):
         super().__init__()
         
    def func(self, X):
         return np.maximum(X, 0)

    def forward(self, X):
        self._last_input = X
        return self.func(X)
    
    def backward(self, output_grad):
        #print('ReLU - backward')
        #print(output_grad.shape)
        return output_grad * self.func(self._last_input)

class MaxPool2d(Module):
    def __init__(self,kernel_size,stride=2):
         super().__init__()
         # TODO: add padding ?
         self._kernel_size = kernel_size
         self._stride = stride
         
    def forward(self, X):
         N,d,h,w = X.shape
         self._X_shape = N,d,h,w
         h_out = int((h - self._kernel_size)/self._stride + 1)
         w_out = int((w - self._kernel_size)/self._stride + 1)
         
         # Im2Col approach http://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
         X_reshaped = X.reshape(N*d, 1, h, w)
         X_col = im2col.im2col_indices(X_reshaped, self._kernel_size, self._kernel_size, padding=0, stride=self._stride)
         max_idx = np.argmax(X_col, axis=0)
         self._max_idx = max_idx
         res = X_col[max_idx, range(max_idx.size)]
         res = res.reshape(h_out, w_out, N, d)
         res = res.transpose(2, 3, 0, 1)
         
#         res = np.empty((N, d, h_out, w_out))
#         for i, x in enumerate(X):
#             res[i,:] = np.max([x[:,(i>>1)&1::2,i&1::2] for i in range(4)],axis=0) #fastest
#             #res[i,:] = x.reshape(d, int(h/2), 2, int(w/2), 2).max(axis=(2, 4)) #2sd fastest
         return res

    def backward(self, output_grad):
        #print('MaxPool2d - backward')
        #print(output_grad.shape)
        N,d,h,w = self._X_shape
        kernel_area = self._kernel_size*self._kernel_size
        dX_col = np.zeros((kernel_area, int(N*d*h*w/kernel_area)))
        dout_flat = output_grad.transpose(2, 3, 0, 1).ravel()
        dX_col[self._max_idx, range(self._max_idx.size)] = dout_flat
        dX = im2col.col2im_indices(dX_col, (N * d, 1, h, w), self._kernel_size, self._kernel_size, padding=0, stride=self._stride)
        return dX.reshape(self._X_shape)
         
class Conv2d(Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
         super().__init__()
         self._in_channels = in_channels
         self._out_channels = out_channels
         self._kernel_size = kernel_size
         self._stride = stride
         self._padding = padding
         law_bound = 1/np.sqrt(kernel_size*kernel_size*in_channels)
         self._bias = np.random.uniform(-law_bound,law_bound,size=(out_channels))
         self._weight = np.random.uniform(-law_bound,law_bound,size=(out_channels, in_channels, kernel_size, kernel_size))
         
    def forward(self, X):
        N,d,h,w = X.shape
        self._X_shape = N,d,h,w
        n_filters = self._weight.shape[0]
        h_out = int((h - self._kernel_size + 2*self._padding) / self._stride + 1)
        w_out = int((w - self._kernel_size + 2*self._padding) / self._stride + 1)
        
        # Im2Col approach: http://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
        # (https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html)
        X_col = im2col.im2col_indices(X, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
        self._X_col = X_col
        W_col = self._weight.reshape(n_filters, -1)
        res = W_col @ X_col
        res = res + np.tile(self._bias, (res.shape[1],1)).T
        res = res.reshape(n_filters, h_out, w_out, N)
        res = res.transpose(3, 0, 1, 2)
        
#        res = np.empty((N, self._out_channels, h_out, w_out))
#        #for n, x in enumerate(X):
#            for f,bias in enumerate(self._bias):
#                res[n, f, :] = bias + np.sum([scipy.ndimage.filters.correlate(x[i], self._weight[f][i], mode='constant')[2:-2,2:-2] for i in range(d)], axis=0)
        return res

    def backward(self, output_grad):
        #print('Conv2d - backward')
        #print(output_grad.shape)
        n_filters = self._weight.shape[0]
        
        self._grad_bias = np.sum(output_grad, axis=(0, 2, 3))
    
        output_grad_reshaped = output_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        grad_weight = output_grad_reshaped @ self._X_col.T
        self._grad_weight = grad_weight.reshape(self._weight.shape)
        
        W_reshape = self._weight.reshape(n_filters, -1)
        dX_col = W_reshape.T @ output_grad_reshaped
        return im2col.col2im_indices(dX_col, self._X_shape, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
    
class Linear(Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        law_bound = 1/np.sqrt(in_features)
        self._bias = np.random.uniform(-law_bound,law_bound,size=(out_features))
        self._weight = np.random.uniform(-law_bound,law_bound,size=(out_features, in_features))
        
    def forward(self, X):
        self._last_input = X
        return np.matmul(X, self._weight.T) + np.tile(self._bias, (X.shape[0],1))
    
    def backward(self, output_grad):
        #print('Linear - backward')
        #print(output_grad.shape)
        self._grad_bias = np.sum(output_grad,axis=0)
        self._grad_weight = np.dot(output_grad.T, self._last_input)
        return np.dot(output_grad, self._weight)
    
class Sequential(Module):
    def __init__(self, *modules):
        self._modules = list(modules)
    
    def add_module(self, module):
        self._modules.append(module)
        
    def forward(self, X):
        for module in self._modules:
            X = module.forward(X)
        return X

    def backward(self, output_grad):
        #print('Sequential - backward')
        #print(output_grad.shape)
        for module in reversed(self._modules):
            #print(type(module))
            output_grad = module.backward(output_grad)
        return output_grad
    
    def step(self, optimizer):
        #print('Sequential - step')
        for module in self._modules:
            #print(type(module))
            if isinstance(module, Linear) or isinstance(module, Conv2d):
                module.step(optimizer)
    
    def zero_grad(self):
        for module in self._modules:
            if isinstance(module, Linear) or isinstance(module, Conv2d):
                module.zero_grad()
                
    def parameters(self):
        parameters = []
        for module in self._modules:
            if isinstance(module, Linear) or isinstance(module, Conv2d):
                parameters.append(module._weight)
                parameters.append(module._bias)
        return parameters
                

class CrossEntropyLoss(object):
    def __init__(self):
        pass
    
    def __call__(self, Y, labels):
        loss = 0
        for i, y in enumerate(Y):
            loss += - y[labels[i]] + np.log(np.sum(np.exp(y)))
        return loss/len(labels)
    
    def grad(self, Y, labels):
        output_grad = np.empty_like(Y)
        for i, y in enumerate(Y):
            output_grad[i,:] = np.exp(y) / np.sum(np.exp(y))
            output_grad[i, labels[i]] -= 1
        return output_grad
        
class Variable(object):
    def __init__(self, data=None):
        self.data = data
        self.grad = None
    
#%%
class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 6, 5),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2,2)
        )
        
        self.classifier = Sequential(
                Linear(16*5*5, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 16*5*5)
        x = self.classifier(x)
        return x.reshape(x.shape[0],-1)
    
    def backward(self, output_grad):
        #print('MyNet - backward')
        #print(output_grad.shape)
        output_grad = self.classifier.backward(output_grad)
        output_grad = output_grad.reshape(-1, 16, 5, 5)
        return self.features.backward(output_grad)    
    
    def step(self, optimizer):
        #print('MyNet - step')
        self.classifier.step(optimizer)
        self.features.step(optimizer)
        
    def zero_grad(self):
        self.classifier.zero_grad()
        self.features.zero_grad()
    
    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

mynet = MyNet()

#%%
parameters = []
for parameter in net.parameters():
    print(type(parameter))
    parameters.append(parameter.data.numpy())
    
#%% Copy weights from torch CNN
mynet.features._modules[0]._weight = parameters[0].copy()
mynet.features._modules[0]._bias = parameters[1].copy()
mynet.features._modules[3]._weight = parameters[2].copy()
mynet.features._modules[3]._bias = parameters[3].copy()
mynet.classifier._modules[0]._weight = parameters[4].copy()
mynet.classifier._modules[0]._bias = parameters[5].copy()

#%% Load DATA
#import pandas as pd
#from sklearn.model_selection import train_test_split
#X = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1]
#X = X * 2
#y = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
#X_train_tensor = torch.FloatTensor(X_train).view(-1, 3, 32, 32)
#X_train = X_train_tensor.numpy()
#y_train_tensor = torch.LongTensor(y_train)
#y_train = y_train_tensor.numpy()

#%% Test entire net
n_d = 8
res_net = net(torch.autograd.Variable(X_train_tensor[0:n_d,:]))
criterion = torch.nn.CrossEntropyLoss()
print(criterion(res_net, torch.autograd.Variable(y_train_tensor[0:n_d])))

res_mynet = mynet(X_train[0:n_d,:])
criterion = CrossEntropyLoss()
print(criterion(res_mynet, y_train[0:n_d]))

#%% Train
import optim
#optimizer = optim.SGD(lr=0.001, momentum=0.9)
optimizer = optim.Adam(lr=0.001)

criterion = CrossEntropyLoss()

N = X_train.shape[0]
batch_size = 8
nb_batchs = int(N / batch_size)
for epoch in range(10): # loop over the dataset multiple times
    
    running_loss = 0.0
    for i in range(nb_batchs):
    #for i in np.random.permutation(nb_batchs):
        # get the inputs
        #inputs, labels = data
        inputs = X_train[i*batch_size:(i+1)*batch_size,:]
        labels = y_train[i*batch_size:(i+1)*batch_size]

        # zero the parameter gradients
        #optimizer.zero_grad()
        mynet.zero_grad()
        
        # forward + backward + optimize
        outputs = mynet(inputs)
        loss = criterion(outputs, labels)
        #print(round(loss,4))
        grad = criterion.grad(outputs, labels)
        mynet.backward(grad)  
        mynet.step(optimizer)
        # print statistics
        running_loss += loss
        if i % 100 == 99: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 100))
            running_loss = 0.0
        

print('Finished Training')

#%% Score on training and test datasets
correct_train = 0
total_train = 0
N = X_train.shape[0]
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = X_train[i*batch_size:(i+1)*batch_size,:]
    labels = y_train[i*batch_size:(i+1)*batch_size]
    outputs = mynet(inputs)
    predicted = np.argmax(outputs.data, axis=1)
    total_train += len(labels)
    correct_train += (predicted == labels).sum()

correct_test = 0
total_test = 0
N = X_test.shape[0]
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = X_test[i*batch_size:(i+1)*batch_size,:]
    labels = y_test[i*batch_size:(i+1)*batch_size]
    outputs = mynet(inputs)
    predicted = np.argmax(outputs.data, axis=1)
    total_test += len(labels)
    correct_test += (predicted == labels).sum()

print('Accuracy -- Train: {} | Test: {}'.format(
        round(100*correct_train/total_train,2), round(100*correct_test/total_test,2)))

#%% Get initialization via reverse engineering
#seq = []
#seq2 = []
#for i in range(1):
#    conv2 = torch.nn.Conv2d(6, 16, 5)
#    conv = Conv2d(6, 16, 5)
#    parameters = []
#    for p in conv2.parameters():
#        parameters.append(p.data.numpy())
#    seq.append(np.std(parameters[0]))
#    print(conv._weight)
#    seq2.append(np.std(conv._weight.reshape(-1)))
#print(np.mean(seq))
#print(np.mean(seq2))
##%% Test convolution layer
#conv = torch.nn.Conv2d(3, 6, 5)
#parameters = []
#for p in conv.parameters():
#    parameters.append(p.data.numpy())
#mynet.features._modules[0]._weight = parameters[0]
#mynet.features._modules[0]._bias = parameters[1]
#
#conv2 = torch.nn.Conv2d(6, 16, 5)
#parameters = []
#for p in conv2.parameters():
#    parameters.append(p.data.numpy())
#mynet.features._modules[3]._weight = parameters[0]
#mynet.features._modules[3]._bias = parameters[1]
#
#res_net = conv(torch.autograd.Variable(X_train[0:1,:]))
#relu = torch.nn.ReLU()
#res_net = relu(res_net)
#pool = torch.nn.MaxPool2d(2,2)
#res_net = pool(res_net)
#res_net = conv2(res_net)
#res_net = relu(res_net)
#res_net = pool(res_net).data.numpy()
#
#res_mynet = mynet.features._modules[0].forward(X_train.numpy()[0:1,:])
#res_mynet = mynet.features._modules[1].forward(res_mynet)
#res_mynet = mynet.features._modules[2].forward(res_mynet)
#res_mynet = mynet.features._modules[3].forward(res_mynet)
#res_mynet = mynet.features._modules[4].forward(res_mynet)
#res_mynet = mynet.features._modules[5].forward(res_mynet)
#sum((res_net - res_mynet).reshape(-1))
#
##%% Test features net
#n_d = 100
#res_net = net.features(torch.autograd.Variable(X_train[0:n_d,:])).data.numpy()
#res_mynet = mynet.features(X_train.numpy()[0:n_d,:])
##print(res_net[0,0])
##print(res_mynet[0,0])
#sum((res_net - res_mynet).reshape(-1))
##%% Test Linear Classifier
#test = np.random.randn(5, 16*5*5).astype(np.float32)
#res_net = net.classifier(torch.autograd.Variable(torch.FloatTensor(test))).data.numpy()
#
#res_mynet = mynet.classifier(test)
#res_net - res_mynet