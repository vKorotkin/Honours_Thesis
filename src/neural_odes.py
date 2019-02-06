#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:34:58 2019

@author: vassili
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from autograd.misc.optimizers import adam
from functools import reduce

from reusable_functions import init_random_params, init_ones_params, neural_net_predict, modded_bfgs


def mass_spring_damper_resid_first_order(params, x, u, du, du2, forcing_fn): 
    c=0.;k=1.
    uv=u(params, x)[0]
    duv=du(params, x)[0]
    err=np.asarray([duv[0]-uv[1],duv[1]+k*uv[0]+c*uv[1]-forcing_fn(uv)]) 
    return np.square(err[0])+np.square(err[1])

def mass_spring_damper_resid_second_order(params, x, u, du,du2,forcing_fn): 
    c=0.;k=1.
    uv=u(params, x)
    duv=du(params, x)
    duv2=du2(params, x)
    return np.square(duv2+c*duv+k*uv-forcing_fn(x))

def diffusion_resid(params, x, u, du,du2,forcing_fn): 
    return np.square(du2(params,x)-forcing_fn(x))

class NeuralOde:
    def __init__(self, G, D, forcing_fn, layer_sizes, resid, x_space, batch_size):
        self.G=G;
        self.D=D;
        self.forcing_fn=forcing_fn;
        self.layer_sizes=layer_sizes;
        self.du=grad(self.u, 1);
        self.du2=grad(self.du,1);
        self.resid=lambda params, x: resid(params, x, self.u, self.du, self.du2, self.forcing_fn)
        self.objective_grad=grad(self.loss_function)
        self.x_space=x_space;
        self.batch_size=int(np.floor(batch_size*len(self.x_space)))
    def u(self, params, x):
        return self.G(x)+self.D(x)*neural_net_predict(params, x)
    
    def resid_batch(self, params, x_batch):
        resid_temp=list(map(lambda x: self.resid(params, x)**2, x_batch))
        resid_sum=reduce(lambda x,y:x+y, resid_temp)
        return resid_sum/len(x_batch)  
    def loss_function(self, params):
        x_batch=np.random.choice(self.x_space,size=self.batch_size) #random choice of points
        err=self.resid_batch(params, x_batch) 
        #for W, b in params:
         #   err=err+np.linalg.norm(W)+np.linalg.norm(b);
        return err 
    def accuracy(self, params):
        return self.resid_batch(params, self.x_space)    
    def phase_plots(self, params):
        nx=len(self.x_space)
        u=self.u;
        du=self.du
        x_space=self.x_space
        
        plt.figure(1)
        plt.clf()
        ax=plt.gca()
        pos = np.zeros(nx)
        vel = np.zeros(nx)
        
        for i,x in enumerate(x_space):
            pos[i]=u(optimized_params,x)
            vel[i]=du(optimized_params,x)
        plttitle='Mass spring damper response'
        plt.plot(x_space, pos, label='pos', marker='2')
        plt.plot(x_space, vel, label='vel', marker='2')
        plt.title(plttitle)
        ax.legend()
        
        plt.figure(2)
        plt.clf()
        ax=plt.gca()
        err = np.zeros(nx)
        for i,x in enumerate(x_space):
            err[i]=self.resid(optimized_params, np.array([x]))
        plttitle='PDE residual over time'
        plt.plot(x_space, err, label='error', marker='2')
        plt.title(plttitle)
        ax.legend()

    
    
myNNode2=NeuralOde(G=lambda x:1., 
                      D=lambda x: x**2/(1+x**2),
                      forcing_fn = lambda x: 0,
                      layer_sizes = [1,10,1],
                      resid=mass_spring_damper_resid_second_order,
                      x_space=np.linspace(0., 7., 50),
                      batch_size=1.)
init_params = init_random_params(1., myNNode2.layer_sizes)
i_rand_init_max=5;

opt_param_list=[]
opt_fun_val_arr=np.zeros([i_rand_init_max])

for i_rand_init in range(i_rand_init_max):
    init_params = init_random_params(1., myNNode2.layer_sizes)
    #init_params = init_ones_params(1., myNNode2.layer_sizes)
    
    
    #du=jacobian(myNNode.u, 1)
    optimized_params, success, fun_val=modded_bfgs(myNNode2.loss_function, myNNode2.objective_grad, init_params, callback=None, 
                                 num_iters=4)
    print("Random initialization {}, Success: {} End function value: {}\n".format(i_rand_init, success, fun_val))
    opt_param_list.append(optimized_params)
    print(fun_val)
    opt_fun_val_arr[i_rand_init]=fun_val
    
    
    myNNode2.phase_plots(optimized_params)
    
plt.figure()
plt.plot(opt_fun_val_arr)
plt.title("Optimized param function value vs random initialization")



























