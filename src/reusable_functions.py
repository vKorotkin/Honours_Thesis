#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:15:15 2019

@author: vassili
"""
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.extend import primitive, defvjp

from scipy.optimize import minimize
from autograd.misc import flatten
from autograd.wrap_util import wraps


@primitive
def relu(x):
    return np.maximum(0.,x);
    
def relu_vjp(ans, x):
    return lambda x: 1.*(x>=0)

@primitive
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_vjp(ans, x):
    return lambda ans, x: ans*(1-ans)

defvjp(relu, relu_vjp)
defvjp(sigmoid, sigmoid_vjp)

def init_random_params(scale, layer_sizes, rs=npr.RandomState()):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    
def init_ones_params(scale, layer_sizes, rs=npr.RandomState()):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * np.ones(m, n),   # weight matrix
             scale * np.ones(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b;
        inputs=np.tanh(outputs);
        #inputs=relu(outputs);
        #inputs = sigmoid(outputs);
    return outputs 


def my_unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(fun, grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _fun = lambda x: flatten(fun(unflatten(x)))[0]
        _grad = lambda x: flatten(grad(unflatten(x)))[0]
        
        if callback:
            _callback = lambda x: callback(unflatten(x))
        else:
            _callback = None
        result=optimize(_fun, _grad, _x0, _callback, *args, **kwargs)
        return unflatten(result.x), result.success, result.fun
        #return unflatten(optimize(_fun, _grad, _x0, _callback, *argss, **kwargs))

    return _optimize

@my_unflatten_optimizer
def modded_bfgs(fun, grad, x, callback=None, num_iters=3, sgd_step_size=10e-5):
    sgd_num_iters=500;
    for iter in range(num_iters):
        
        result=minimize(fun, x,method = 'BFGS', options={'disp': True}, callback=callback, jac=grad)
        #result=minimize(fun, x,method = 'L-BFGS-B', 
         #              options={'gtol':1e-6,'disp': True, 'maxiter':1500}, callback=callback, jac=grad)
        if (result.success==False):
             print("BFGS convergence failed at iter {}, func val: {} start SGD".format(iter, fun(result.x)))
             x=sgd(fun, grad, result.x,num_iters=sgd_num_iters,init_step_size=sgd_step_size)
             result.x=x
             #result=minimize(fun, x,method = 'L-BFGS-B', 
              #               options={'disp': True}, callback=callback, jac=grad, tol=1e-2)
        else:
            break
    return result


def sgd(fun, grad, x, num_iters=200, init_step_size=0.1, mass=0.1):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""    
    print("SGD start")
    eta = init_step_size
    beta = 0.01
    step_size_fun = lambda k: eta / (1 + beta*k)
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        #velocity = mass * velocity - (1.0 - mass) * g
        velocity=-g;

        x = x + step_size_fun(i) * velocity
        if i % 50 == 0:
            print("Iter:{}, Obj. Fun Val:{}".format(i, fun(x)[0]))
    return x



def adam(fun, grad, x, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
        if i % 50 == 0:
            print("Iter:{}, Obj. Fun Val:{}".format(i, fun(x)[0]))
    return x
