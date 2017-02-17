#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:29:39 2017

@author: EmileMathieu
"""

import numpy as np

class Base_module(object):
     def __init__(self):
         raise NotImplementedError()
     def forward(self, x):
         raise NotImplementedError()
     def backward(self, x):
         raise NotImplementedError()
         
class ReLu(Base_module):
    def __init__(self):
         super().__init__()
         
    def forward(self, x):
        return np.maximum(x, 0)
    
    def backward(self, x):
        pass

class MaxPool2d(Base_module):
    def __init__(self,kernel_size,stride):
         super().__init__()
         self._kernel_size = kernel_size
         self._stride = stride
         
    def forward(self, x):
        pass
         
class Conv2d(Base_module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
         super().__init__()
         self._in_channels = in_channels
         self._out_channels = out_channels
         self._kernel_size = kernel_size
         self._stride = stride
         self._padding = padding
         
class Linear(Base_module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight = np.zeros((out_features, in_features))
        self._bias = np.zeros(out_features)
        
    def forward(self, x):
        return np.dot(self._weight, x) + self._bias