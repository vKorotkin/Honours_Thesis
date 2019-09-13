
import os
import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dill as pickle

import optimization_module_neural_network as rfc

"""
This module provides the boundary extension function for the wave equation neural network solution
in 1D. 
The wave eq. neural net solution is u(x,t)=G(x,t)+D(x,t)Phi(x,t)
with G the boundary extension, D the distance function, and Phi the neural network output. 
D(x,t) makes sure the second term doesn't influence the boundary/initial conditions. 
G(x,t) needs to satisfy initial/boundary conditions. 
A neural network is trained for G. 

Parameters common for all functions here
-----------
g0: Initial condition, u(x,0)
g1: Initial condition on derivative, u_t(x,0)
f0: Left boundary condition: u(0,t)
f1: Right boundary condition: u(L,t)
L: Length of spatial domain. x is on [0,L]
t_max: Maximum time. t is on [0,t_max]
N: Number of points taken (common for time and space)
,g1,f0,f1,t_max,L,N)
"""
FILE_TO_STORE_G="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/G_func"


mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;


def get_G_loss_function(G):
    """
    Returns loss function for G.
    """
    return 0

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

def set_labels_and_legends(ax):
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    ax.set_zlabel('$u$')
    plt.legend()


def plot_2D_function(ax, f, flabel, params, t_max,L, N):
    t=np.linspace(0,t_max,N)
    x=np.linspace(0, L, N)
    X,T=np.meshgrid(x, t)
    U=np.zeros(X.shape)
    dudt_0=np.zeros(x.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i,j]=f(params,X[i,j], T[i,j])

    
    surf = ax.plot_surface(X, T, U, cmap=mpl.cm.coolwarm,
                        linewidth=0, antialiased=False, label=flabel)
    #some kind of bug with legend if this isnt here
    #https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d

def get_parameters_G(g0,g1,f0,f1,t_max,L,N,layer_sizes):
    G=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    p0=rfc.init_random_params(1, layer_sizes)
    #Gradient of G w.r.t. time
    dGdt=grad(G,2)

    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)

    def loss_function(params):
        sum=0.
        for x in x_space:
            sum+=(G(params,x,0.)-g0(x))**2+(dGdt(params,x,0.)-g1(x))**2
        for t in t_space:
            sum+=(G(params,0.,t)-f0(x))**2+(G(params,L,t)-f1(x))**2
        return sum

    p, _=rfc.unflattened_lbfgs(loss_function, grad(loss_function,0),p0,\
        max_feval=400, max_iter=400, callback=None)

    return G,p;

def error_plots(G,p, g0,g1,f0,f1,t_max,L,N):

    fig=plt.figure()
    x_space=np.linspace(0,L,N)
    dudt_0=np.zeros(x_space.shape)
    dGdt=grad(G,2)
    for i,x in enumerate(x_space):
        dudt_0[i]=dGdt(p,x,0.)
    plt.plot(x_space, dudt_0, label="Actual")
    plt.plot(x_space, g1(x_space), label="Target")
    plt.legend()
    plt.title("$G_t(x,0)$")
    plt.show()

def get_functions_from_strings(g0expr,g1expr,f0expr,f1expr):
    g0=lambda x:eval(g0expr)
    g1=lambda x:eval(g1expr)
    f0=lambda t:eval(f0expr)
    f1=lambda t:eval(f1expr)
    return g0,g1,f0,f1

def save_G_to_file(G,p,g0expr,g1expr,f0expr,f1expr,fname):
    identifier="g0: %s, g1: %s, f0: %s, f1: %s" % (g0expr, g1expr, f0expr, f1expr)
    G_dict={}

    if os.path.isfile(fname):
        if os.path.getsize(fname) > 0:
            G_dict = pickle.load(open(fname, "rb"))
        G_dict[identifier]={'G':G, 'Parameters':p}
        pickle.dump(G_dict, open(fname, "wb"))
    else:
        G_dict = {identifier: {'G':G, 'Parameters':p}}
        pickle.dump(G_dict, open(fname, "wb"))

def get_G_from_file(fname, identifier):
    G_dict = pickle.load(open(fname, "rb"))
    G=G_dict[identifier]['G']
    p=G_dict[identifier]['Parameters']
    return G,p

def create_or_load_G(g0expr, g1expr, f0expr, f1expr,L, t_max, N, layer_sizes, fname, create_g=True):
    identifier="g0: %s, g1: %s, f0: %s, f1: %s" % (g0expr, g1expr, f0expr, f1expr)
    if create_g:
        g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
        G,p=get_parameters_G(g0,g1,f0,f1,t_max,L,N,layer_sizes)
        save_G_to_file(G,p,g0expr,g1expr,f0expr,f1expr,fname)
    else:
        G,p=get_G_from_file(fname, identifier)
    return G,p

def create_or_load_D(L):
    D=lambda params, x,t: 0.01*x*(L-x)*t**2
    return D

def test():
    
    #The eval stuff is done to be able to identify the functions that G was supposed to fit 
    #when saving and loading. 
    #Initial conditions
    fname=FILE_TO_STORE_G
    g0expr='np.sin(x)'
    g1expr='np.cos(x)'
    #Boundary conditions
    f0expr='0'
    f1expr='0'
    #Limits and number of points
    L=2*np.pi
    t_max=4
    N=15;
    #Network hyperparameters
    layer_sizes=[2,10,1]

    G,p=create_or_load_G(g0expr, g1expr, f0expr, f1expr,L, t_max, N, \
        layer_sizes, fname, create_g=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    plot_targets(ax, g0,g1,f0,f1,t_max,L,N)

    plot_2D_function(ax, G,"$G$", p, t_max,L,N)
    set_labels_and_legends(ax)
    plt.show()
    error_plots(G,p, g0,g1,f0,f1,t_max,L,N)

if __name__ == "__main__":
	test()





