
LOCAL_PATH="/home/vassili/Desktop/Thesis/Honours_Thesis.git/"
FILE_TO_STORE_G=LOCAL_PATH+"data/G_func_first_order"
FILE_TO_STORE_D=LOCAL_PATH+"data/D_func_first_order"
FIGNAME_U=LOCAL_PATH+"figs/wave_first_order_U.eps"
FIGNAME_G=LOCAL_PATH+"figs/wave_first_order_G.eps"
FIGNAME_D=LOCAL_PATH+"figs/wave_first_order_D.eps"
####Hack to be able to import modules from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
######

import optimization_module_neural_network as rfc
import autograd.numpy as np

import matplotlib.pyplot as plt

from boundary_extension_first_order import get_G_loss_function,get_D_loss_function, plot_vector_function_xt
from autograd import grad, value_and_grad, jacobian


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
from helper_plotting import plot_targets, plot_vector_function_all_elements, plot_2D_function
from optimization_module_neural_network import get_parameters, init_random_params

mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;


def get_resid_wave_eq_first_order(u):
    """
    Generate wave equation residual function from solution ansatz u
    This is for first order form. 
    u must of be of form 
    [u1, u2, u3]=u(params, x,t) where params are going to be optimized over. 
    u1 is u, the position
    u2 is ut, unknown corresponding to derivative w.r.t. time. 
    u3 is ux, unknown corresponding to derivative w.r.t. time. 
        NOTE: derivative of u1 w.r.t. t i.e. du/dt is only equal to u2=ut if PDE system is solved exactly. 

    Returns
    --------
    resid: scalar valued function resid(params,x,t).
    Returns residual of PDE from neural network parameters and x,t (space and time positions). 
    """
    #See autograd docs for jacobian documentation. 
    #This code treats u as a vector valued function of the arguments x,t
    #So have to compute two jacobians, one for each. Consider changing depending on efficiency. 
    #Is one jacobian w.r.t single vector [x,t] much faster than two jacobians w.r.t. (x,t)?

    #Jx is vector valued function of params,x,t
    #Jx(params,x,t) is d([u,ut,ux])/dx(params,x,t)
    Jx=jacobian(u, 1)
    Jt=jacobian(u, 2)


    elementwise_error=lambda params,x,t: np.array([\
        Jx(params,x,t)[0]-Jt(params,x,t)[0]-u(params,x,t)[2]+u(params,x,t)[1], \
        Jx(params,x,t)[1]-Jt(params,x,t)[2], \
        Jx(params,x,t)[2]-Jt(params,x,t)[1]
        ])

    #elementwise_error=lambda params,x,t: np.array([\
     #   Jx(params,x,t)[0], 0., 0.])

    resid=lambda params,x,t: np.linalg.norm((elementwise_error(params,x,t)), ord=2)
    return resid

def get_loss_function_wave_eq_first_order(u,nx, nt, L, t_max):

    """
    Generate loss function to be used for optimization 

    Returns
    -----------
    loss_function: loss_function(params) returns sum of residual error (squared) over grid of evenly spaced collocation points
                    Positions and amount of collocation points specified by nx,nt,L, t_max
    """
    #todo: make x, t generation independent of this. Perhaps random generation? 
    t=np.linspace(0,t_max,nt)
    #The displacement on boundaries prescribed. Leave those alone.
    x=np.linspace(L/10, L*9/10, nx)

    #Get and vectorized resid to evaluate and sum over all the collocation points
    resid=get_resid_wave_eq_first_order(u)
    vresid=np.vectorize(resid,excluded=[0])
    def loss_function(params):
        res_arr=vresid(params,x[:,None], t[None,:])
        return np.sum(res_arr)/(res_arr.shape[0]*res_arr.shape[1])
    return loss_function


def test():
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
    layer_sizes=[2,10,10,10,3]
    max_function_evals=100
    max_iterations=100

    #Create or load G
    G=lambda params, x,t: rfc.neural_net_predict(params, np.array([x,t]))
    g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
    loss_function=get_G_loss_function(G,g0,g1,f0,f1,t_max,L, N=5*nx)

    G,p_G=create_or_load_trained_f(G, loss_function, g0expr, g1expr, f0expr, f1expr,L, t_max, \
        layer_sizes,fname=FILE_TO_STORE_G, create_f=False,maxiter=max_iterations,maxfuneval=max_function_evals)
    
    #Create or load D
    D=lambda params,x,t:(L-x)*x*t*(t_max-t)*np.ones(3)

    #Train u
    x0=rfc.init_random_params(1, layer_sizes)
    u=lambda params, x,t: G(p_G,x,t)+\
        D(None,x,t)*rfc.neural_net_predict(params, np.array([x,t]))

    #Plots
    plot_vector_function_all_elements(G, p_G, "G", g0expr,g1expr,f0expr,f1expr, t_max,L,N=nx*5)
    plt.savefig(FIGNAME_G, bbox_inches="tight")
    plot_vector_function_all_elements(D, None, "D", g0expr,g1expr,f0expr,f1expr, t_max,L,N=nx*5)
    plt.savefig(FIGNAME_D, bbox_inches="tight")   

    loss_function=get_loss_function_wave_eq_first_order(u,nx, nt, L, t_max)
    loss_grad=grad(loss_function,0)
    p_U, fval=rfc.unflattened_lbfgs(loss_function, loss_grad, x0, \
        max_feval=max_function_evals, max_iter=max_iterations, callback=None)
    plot_vector_function_all_elements(u, p_U, "U", g0expr,g1expr,f0expr,f1expr, t_max,L,N=nx*5)
    plt.savefig(FIGNAME_U, bbox_inches="tight")
    resid=get_resid_wave_eq_first_order(u)
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_2D_function(ax,resid, "Residual", p_U, t_max, L, nx*5)
    plt.title("Residual")
    plt.show(block=True)
if __name__ == "__main__":
	test()