

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



def plot_vector_function_xt(ax, f, flabel, params, t_max,L, N, element_to_plot=0):
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
            U[i,j]=f(params,X[i,j], T[i,j])[element_to_plot]

    surf = ax.plot_surface(X, T, U, cmap=mpl.cm.coolwarm,
                        linewidth=0, antialiased=False, label=flabel)
    #some kind of bug with legend if this isnt here
    #https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    
def plot_vector_function_all_elements(f, p_f, identifier, g0expr,g1expr,f0expr,f1expr, t_max,L,N):
    fig = plt.figure()
    for element_to_plot in range(3):
        
        ax = fig.add_subplot(130+element_to_plot+1, projection='3d')
        g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
        #plot_targets(ax, g0,g1,f0,f1,t_max,L,20*N)

        plot_vector_function_xt(ax, f,identifier, p_f, t_max,L,N, element_to_plot=element_to_plot)
        #ax.set_zlim(-1,1)
        plt.title("$"+identifier+"_%d$" % element_to_plot+1)
    

def plot_2D_function_general_domain(ax, domain, f, flabel):
        #f_vals=domain 
        f_vals=[]
        for dom_idx, (X,Y) in enumerate(domain):
                f_vals.append(np.zeros(X.shape))
                for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                                f_vals[dom_idx][i][j]=f(X[i,j], Y[i,j])
        for dom_idx, (X,Y) in enumerate(domain):
                surf = ax.plot_surface(X, Y, f_vals[dom_idx], cmap=mpl.cm.coolwarm,
                                        linewidth=0, antialiased=False, label=flabel)
                #some kind of bug with legend if this isnt here
                #https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d

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