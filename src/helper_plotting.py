

import os
import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import create_or_load_trained_f, get_functions_from_strings

import optimization_module_neural_network as rfc
from optimization_module_neural_network import get_parameters

def plot_targets(ax, g0,g1,f0,f1,t_max,L,N):
    """
    Plot the optimization targets on the boundary (initial and boundary conditions)

    Parameters
    -----------
    ax: Axes of figure on which to plot
    """
    t=np.linspace(0,t_max,N)
    x=np.linspace(0, L, N)
    
    ax.plot(x, np.zeros(t.shape), g0(x), label="Target $G(x,0)$")
    ax.plot(x, np.zeros(t.shape), g1(x),label="Target $G_t(x,0)$")
    ax.plot(0*np.ones(x.shape), t, f0(x),label="Target $G(0,t)$")
    ax.plot(L*np.ones(x.shape), t, f1(x),label="Target $G(L,t)$")