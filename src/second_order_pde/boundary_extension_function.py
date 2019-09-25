
import os
import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import create_or_load_trained_f, get_functions_from_strings

import optimization_module_neural_network as rfc
from optimization_module_neural_network import get_parameters

"""
This module provides the boundary extension function for the wave equation neural network solution
in 1D. 
The wave eq. neural net solution is u(x,t)=G(x,t)+D(x,t)Phi(x,t)
with G the boundary extension, D the distance function, and Phi the neural network output. 
D(x,t) makes sure the second term doesn't influence the boundary/initial conditions. 
G(x,t) needs to satisfy initial/boundary conditions. 
A neural network is trained for G. 

Names common for all functions here
-----------
G(params,x,t): lambda function corresponding to G(x,t). Params are network weights'
D(params,x,t): lambda function corresponding to D(x,t). Params are network weights
g0: Initial condition, u(x,0)
g1: Initial condition on derivative, u_t(x,0)
f0: Left boundary condition: u(0,t)
f1: Right boundary condition: u(L,t)
L: Length of spatial domain. x is on [0,L]
t_max: Maximum time. t is on [0,t_max]
N: Number of points taken (common for time and space)
"""
FILE_TO_STORE_G="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/G_func"
FILE_TO_STORE_D="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/D_func"

mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;


def get_G_loss_function(G,g0,g1,f0,f1,t_max,L,N):
    """
    Returns loss function for G.
    """
    dGdt=grad(G,2)

    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)

    def loss_function(params):
        sum=0.
        for x in x_space:
            sum=sum+(G(params,x,0.)-g0(x))**2+(dGdt(params,x,0.)-g1(x))**2
        for t in t_space:
            sum=sum+(G(params,0.,t)-f0(x))**2+(G(params,L,t)-f1(x))**2
        
        return sum

    return loss_function

def get_D_loss_function(D,t_max,L,N):
    """
    Returns loss function for D.
    """
    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)
    X,T=np.meshgrid(x_space, t_space)
    dDdx=grad(D,1)
    dDdx2=grad(dDdx,1)
    def loss_function(params):
        sum=0.
        #int(np.floor(X.shape[1]/3)),
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                sum=sum+np.square(D(params, X[i,j], T[i,j])-1) \
                    +np.square(dDdx2(params, X[i,j], T[i,j]))
                #sum=sum+np.square(D(params, X[i,j], T[i,j])-np.tanh(T[i,j]))
        #for t in t_space:
        #    sum=sum+np.square(dDdx(params, 0., t))+np.square(dDdx(params, L, t))

        return sum/(X.shape[0]*X.shape[1])
    return loss_function

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

def set_labels_and_legends(ax, zlabel="$u$"):
    """
    Set labels and legend for axes. 
    NOTE: sets z axis to u by default. 
    Parameters
    -----------
    ax: Axes of figure on which to plot
    """
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    ax.set_zlabel('$u$')
    plt.legend()


def plot_2D_function(ax, f, flabel, params, t_max,L, N):
    """
    Uses axes ax to plot f(params, x,t) over x,t grid on [0,L]x[0,t_max] with N points in each direction. 
    Labels f with flabel. 
    Parameters
    -----------
    ax: Axes of figure on which to plot
    """
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


def error_plots(G,D,p_G, p_D, g0,g1,f0,f1,t_max,L,N):
    """
    Plot quamtities of interest for G
    """

    """
    fig=plt.figure()
    x_space=np.linspace(0,L,N)
    dudt_0=np.zeros(x_space.shape)
    dGdt=grad(G,2)
    for i,x in enumerate(x_space):
        dudt_0[i]=dGdt(p_G,x,0.)
    plt.plot(x_space, dudt_0, label="Actual")
    plt.plot(x_space, g1(x_space), label="Target")
    plt.legend()
    plt.title("$G_t(x,0)$")
    plt.show()
    """

    fig=plt.figure()
    x_space=np.linspace(0,L,N)
    dDdt_0_vals=np.zeros(x_space.shape)
    dDdt2_0_vals=np.zeros(x_space.shape)
    dDdt=grad(D,2)
    dDdt2=grad(dDdt,2)
    for i,x in enumerate(x_space):
        dDdt_0_vals[i]=dDdt(p_D,x,0.)
    for i,x in enumerate(x_space):
        dDdt2_0_vals[i]=dDdt2(p_D,x,0.)
    plt.plot(x_space, dDdt_0_vals, label="$D_t(x,0)$")
    plt.plot(x_space, dDdt_0_vals, label="$D_tt(x,0)$")
    plt.legend()
    plt.title("$D(x,0)$ properties")
    plt.show()

def test():
    
    #The eval stuff is done to be able to identify the functions that G was supposed to fit 
    #when saving and loading. 
    #Initial conditions
    g0expr='np.sin(x)'
    #g1expr='np.cos(x)'
    g1expr='0.'
    #Boundary conditions
    f0expr='0'
    f1expr='0'
    #Limits and number of points
    L=2*np.pi
    t_max=4
    N=15;
    #Network hyperparameters
    layer_sizes=[2,7,7,1]
    G=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    loss_function=get_G_loss_function(G,g0,g1,f0,f1,t_max,L,N)
    G,p_G=create_or_load_trained_f(G, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
        layer_sizes,fname=FILE_TO_STORE_G, create_f=False,maxiter=400,maxfuneval=400)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    #plot_targets(ax, g0,g1,f0,f1,t_max,L,N)

    plot_2D_function(ax, G,"$G$", p_G, t_max,L,N)
    plt.show()

    D=lambda params, x,t: x*(L-x)*t**2*rfc.neural_net_predict(params, np.array([x,t]))
    loss_function=get_D_loss_function(D,t_max,L,N)
    D,p_D=create_or_load_trained_f(D, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
        layer_sizes,maxiter=300,maxfuneval=300,fname=FILE_TO_STORE_D, create_f=True)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    #plot_targets(ax, g0,g1,f0,f1,t_max,L,N)

    plot_2D_function(ax, D,"$D$", p_D, t_max,L,N)
    set_labels_and_legends(ax)
    plt.show()
    error_plots(G,D,p_G,p_D,g0,g1,f0,f1,t_max,L,N)

if __name__ == "__main__":
	test()





