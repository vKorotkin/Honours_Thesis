#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:15:15 2019

@author: vassili
"""
import autograd.numpy as np
import autograd.numpy.random as npr

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b;
        inputs=np.tanh(outputs);
        #inputs = sigmoid(outputs);
    return outputs 
