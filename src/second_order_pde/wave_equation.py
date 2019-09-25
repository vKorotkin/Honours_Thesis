
"""
    Obtain solution to wave equation in 1D. 
    Solution ansatz: u(x,t)=G(x,t)+D(x,t)Phi(x,t)
    -G(x,t) is boundary extension function
    -D(x,t) is distance function
    -Phi(x,t) is neural network trained to minimize residual
    Different IC's and BC's can be subbed in easily - maybe... unsure about this.
    Might be the case for some special ICs or BCs, check Berg/Nystrom paper. 

    Parameters common to all functions in this module
    -----------
    u: u(params, x,t) The function to be used as solution ansatz. u must take arguments u(params, x,t)
        where params are the neural network parameters, x is position (scalar), t is time (scalar)
    nx, nt: number of collocation points in x and t axes respectively. 
    L: Length of spatial domain
    t_max: Maximum time. 
    g0: Initial condition, position. u(x,0)=g0(x)
    g1: Initial condition, velocity du/dt(x,0)=g1(x)
    f0: Boundary condition, x=0 u(0,t)=f0(t)
    f1: Boundary condition, x=1 u(L,t)=f1(t)
"""

FILE_TO_STORE_G="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/G_func"
FILE_TO_STORE_D="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/D_func"

#Hack to be able to import modules from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import optimization_module_neural_network as rfc
import autograd.numpy as np

import matplotlib.pyplot as plt

from boundary_extension_function import get_G_loss_function ,plot_2D_function,set_labels_and_legends,get_D_loss_function
from helper import get_functions_from_strings, create_or_load_trained_f
from autograd import grad, value_and_grad


def get_resid_wave_eq(u):
    """
    Generate wave equation residual function from solution ansatz u
    
    Returns
    --------
    resid: function resid(params,x,t). Returns residual of PDE from neural network parameters and x,t space and time positions. 
    """
    ux=grad(u,1)
    uxx=grad(ux,1)
    ut=grad(u, 2)
    utt=grad(ut, 2)
    resid=lambda params,x,t: np.square(utt(params, x, t)-uxx(params, x, t))
    return resid

def get_loss_function(u,nx, nt, L, t_max):

    """
    Generate loss function to be used for optimization 

    Returns
    -----------
    loss_function: loss_function(params) returns sum of residual error (squared) over grid of evenly spaced collocation points
                    Positions and amount of collocation points specified by nx,nt,L, t_max
    """
    #todo: make x, t generation independent of this. Perhaps random generation? 
    t=np.linspace(0,t_max,nt)
    x=np.linspace(0, L, nx)

    #Get and vectorized resid to evaluate and sum over all the collocation points
    resid=get_resid_wave_eq(u)
    vresid=np.vectorize(resid,excluded=[0])
    def loss_function(params):
        res_arr=vresid(params,x[:,None], t[None,:])
        return np.sum(res_arr)/(res_arr.shape[0]*res_arr.shape[1])
    return loss_function
    
def optimize_u(G, params_G, D,params_D, layer_sizes, nx, nt, L, t_max, max_function_evals=100, max_iterations=100):
    """
    Returns u(x,t) to fit wave equation.
    """


    x0=rfc.init_random_params(1, layer_sizes)

 
    u=lambda params, x,t: G(params_G,x,t)+\
        D(params_D,x,t)*rfc.neural_net_predict(params, np.array([x,t]))

    #res_arr=vresid(x0,X,T)


    #plot_result(u,G,D,x0, params_G,params_D,t_max,L,nx*5,nt*5)

    loss_function=get_loss_function(u,nx, nt, L, t_max)
    loss_grad=grad(loss_function,0)
    p, fval=rfc.unflattened_lbfgs(loss_function, loss_grad, x0, \
        max_feval=max_function_evals, max_iter=max_iterations, callback=None)
   # plot_result(u,G,D,p, params_G,params_D,t_max,L,nx*5,nt*5)
    return u,p

def plot_result(u,G,D,param_u, param_g,param_D,t_max,L,nx,nt):
    """
    Plots result of simulation. 
    Generates four subplots over domain
        -u, the solution
        -Residual, error of solution as def'd by the residual function
        -G: boundary extension
        -D: distance function

    Parameters:
        u -- Solution function
        G -- Boundary extension function
        D -- Distance function 
            NOTE: All above functions are of form f(params,x,t)
        param_u,param_g,param_D -- neural network parameters for u,G,D respectively. 
        nx, nt: Amount of points used to plot over

    """

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    plot_2D_function(ax, u, "$u(x,t)$", param_u, t_max,L, nx)
    set_labels_and_legends(ax)
    print("u done")
    

    resid=get_resid_wave_eq(u)
    ax = fig.add_subplot(222, projection='3d')
    plot_2D_function(ax, resid, "Residual", param_u, t_max,L, nx)
    set_labels_and_legends(ax)
    ax.set_zlabel('Residual')
    print("Resid done")
    
    ax = fig.add_subplot(223, projection='3d')
    plot_2D_function(ax, G, "$G$", param_g, t_max,L, nx)
    set_labels_and_legends(ax)
    print("G done")
    ax = fig.add_subplot(224, projection='3d')
    plot_2D_function(ax, D, "$D$", param_D, t_max,L, nx)
    set_labels_and_legends(ax)
    print("D done")
    plt.show()

def test():
    """
    If first time running code set create_f=True in G and D creation
    """
    fname=FILE_TO_STORE_G

    #IC
    g0expr='np.sin(x)'
    #g1expr='np.cos(x)'
    g1expr='0.'
    #BC
    f0expr='0'
    f1expr='0'
    #Limits and number of points
    L=2*np.pi
    t_max=0.5
    nx=4
    nt=4
    #Network hyperparameters
    layer_sizes=[2,10,10,1]
    max_function_evals=100
    max_iterations=100

    G=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    loss_function=get_G_loss_function(G,g0,g1,f0,f1,t_max,L,N=15)
    G,param_G=create_or_load_trained_f(G, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max,  \
        layer_sizes, N=15,maxiter=300,maxfuneval=300,fname=FILE_TO_STORE_G, create_f=False)


    D=lambda params, x,t: x*(L-x)*t**2*rfc.neural_net_predict(params, np.array([x,t]))
    loss_function=get_D_loss_function(D,t_max,L,15)
    D,param_D=create_or_load_trained_f(D, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
        layer_sizes,N=15,maxiter=100,maxfuneval=100,fname=FILE_TO_STORE_D, create_f=False)
    u, param_u=optimize_u(G,param_G,D,param_D,layer_sizes, nx, nt, L, t_max, max_function_evals=200, max_iterations=200)
    print("Loss function:%.2f", loss_function(param_u))
    plot_result(u,G,D,param_u,param_G,param_D,t_max,L,10*nx,10*nt)
    
    

if __name__ == "__main__":
	test()