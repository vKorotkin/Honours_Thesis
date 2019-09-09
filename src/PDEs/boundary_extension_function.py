import reusable_functions_colab as rfc
import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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



mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;



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

def set_labels_and_show(ax):
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    ax.set_zlabel('$u$')
    plt.legend()
    plt.show()

def plot_G(ax, G, params, t_max,L, N):
    t=np.linspace(0,t_max,N)
    x=np.linspace(0, L, N)
    X,T=np.meshgrid(x, t)
    U=np.zeros(X.shape)
    dudt_0=np.zeros(x.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i,j]=G(params,X[i,j], T[i,j])

    
    surf = ax.plot_surface(X, T, U, cmap=mpl.cm.coolwarm,
                        linewidth=0, antialiased=False, label="$G$")
    #some kind of bug with legend if this isnt here
    #https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d

def get_parameters(g0,g1,f0,f1,t_max,L,N,layer_sizes):
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

    p, _, _=rfc.unflattened_lbfgs(loss_function, grad(loss_function,0),p0,callback=None)
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

def test():
    #Initial conditions
    g0=lambda x:np.sin(x)
    g1=lambda x:np.cos(x)
    #Boundary conditions
    f0=lambda t:0
    f1=lambda t:0
    #Limits and number of points
    L=2*np.pi
    t_max=2
    N=15;
    #Network hyperparameters
    layer_sizes=[2,10,1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_targets(ax, g0,g1,f0,f1,t_max,L,N)
    G,p=get_parameters(g0,g1,f0,f1,t_max,L,N,layer_sizes)
    plot_G(ax, G, p, t_max,L,N)
    set_labels_and_show(ax)
    error_plots(G,p, g0,g1,f0,f1,t_max,L,N)

if __name__ == "__main__":
	test()





