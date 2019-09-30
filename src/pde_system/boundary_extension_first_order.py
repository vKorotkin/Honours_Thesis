"""
This module provides the boundary extension function for the wave equation neural network solution
in 1D, in a first order system of PDEs formulation. 
The wave eq. neural net solution is 
[u(x,t), ut(x,t), ux(x,t)]=G(x,t)+D(x,t).Phi(x,t)]
where G, D, Phi are vector valued functions of x,t. 
They all have their parameters that are optimised for desired behaviour.
All map (x,t) to R3 (with components contributing to u, ut, and ux) 
G(x,t)=[G1(x,t), G2(x,t), G3(x,t)], sim. for D, Phi.
with G the boundary extension, D the distance function, and Phi the neural network output. 
D(x,t) makes sure the second term doesn't influence the boundary/initial conditions. 
G(x,t) needs to satisfy initial/boundary conditions. 
After finding G,D, Phi is optimised to make G+D.Phi fit the local equation. 

Because G,D,Phi are neural networks, their input is a vector v=[x,t]

Names common for the whole script
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
FILE_TO_STORE_G="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/G_func_first_order"
FILE_TO_STORE_D="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/D_func_first_order"

#Hack to be able to import modules from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import create_or_load_trained_f, get_functions_from_strings
from helper_plotting import plot_targets, plot_vector_function_xt, plot_vector_function_all_elements
import optimization_module_neural_network as rfc
from optimization_module_neural_network import get_parameters, init_random_params

mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;


def get_G_loss_function(G,g0,g1,f0,f1,t_max,L,N):
    """
    Returns loss function for G.
    BC's and ICs are specified on the network outputs (correspond to u, ut, ux)
    Note, ut doesn't have to be (and usually isn't) del u/del t at this stage
    It's a separate unknown. 
    """
    #G0:u G1:ut G2:ux (remember indexing starts at 0)
    #IC:g0,g1 BC:f0,f1
    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)

    X,T=np.meshgrid(x_space, t_space)
    #maybe vectorize, but is ok for now
    def loss_function(params):
        sum=0.
        
        for x in x_space:
            sum=sum+(G(params,x,0.)[0]-g0(x))**2+(G(params,x,0.)[1]-g1(x))**2
        for t in t_space:
            sum=sum+(G(params,0.,t)[0]-f0(x))**2+(G(params,L,t)[0]-f1(x))**2

        #for i in range(X.shape[0]):
        #    for j in range(X.shape[1]):
        #        sum=sum+0.2*(G(params,L,t)[2])**2 #want G2=0 everywhere just in case
        return sum

    return loss_function

def get_D_loss_function(D,g0,g1,f0,f1,t_max,L,N):
    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)
    #same as G but instead of g0 etc just want 0 at the boundaries
    #maybe vectorize, but is ok for now
    X,T=np.meshgrid(x_space, t_space)

    def loss_function(params):
        sum=0.
        for x in x_space:
            sum=sum+(D(params,x,0.)[0])**2+(D(params,x,0.)[1])**2
        for t in t_space:
            sum=sum+(D(params,0.,t)[0])**2+(D(params,L,t)[0])**2
        for i in range(1, X.shape[0]-1):
            for j in range(1, X.shape[1]-1):
                sum=sum+np.linalg.norm((D(params,X[i,j], T[i,j]))-1)**2
        return sum
    return loss_function
    

def test_G(g0expr,g1expr,f0expr,f1expr,t_max,L, N, maxiter, create_f,layer_sizes=[2,7,7,3]):

    
    G=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    loss_function=get_G_loss_function(G,g0,g1,f0,f1,t_max,L,N)
    
    #G
    G,p_G=create_or_load_trained_f(G, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
        layer_sizes,fname=FILE_TO_STORE_G, create_f=create_f,maxiter=maxiter,maxfuneval=maxiter)

    plot_vector_function_all_elements(G, p_G, "G", g0expr,g1expr,f0expr,f1expr, t_max,L,N)


    
def test_D(g0expr,g1expr,f0expr,f1expr,t_max,L, N, maxiter, create_f):

    layer_sizes=[2,7,7,3]
    #D=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    
    #Can just use this, want it to be zero at the boundaries. 
    #Can try to use similar trick for second order wave eq formilation somehow? 
    D=lambda params,x,t:(L-x)*x*t*(t_max-t)*np.ones(3)
    #D
    #loss_function=get_D_loss_function(D,g0,g1,f0,f1,t_max,L,N)
    #D,p_D=create_or_load_trained_f(D, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
    #    layer_sizes,fname=FILE_TO_STORE_D, create_f=create_f,maxiter=maxiter,maxfuneval=maxiter)
    plot_vector_function_all_elements(D, None, "D", g0expr,g1expr,f0expr,f1expr, t_max,L,N)
    


def test():
    #The eval stuff is done to be able to identify the functions that G was supposed to fit 
    #when saving and loading. 
    #Initial conditions
    g0expr='np.sin(x)'
    #g1expr='np.cos(x)'
    g1expr='0.'
    #Boundary conditions
    f0expr='0.'
    f1expr='0.'
    #Limits and number of points
    L=2*np.pi
    t_max=4
    N=15

    test_G(g0expr,g1expr,f0expr,f1expr, t_max,L, N, maxiter=400,create_f=True, layer_sizes=[2,7,7,3])
    test_D(g0expr,g1expr,f0expr,f1expr, t_max,L, N, maxiter=100,create_f=True)



    #Need to add test, see how ut corresponds to u? It probably won't. 

if __name__ == "__main__":
	test()