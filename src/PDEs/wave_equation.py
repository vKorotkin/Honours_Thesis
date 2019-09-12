from boundary_extension_function import create_or_load_G, create_or_load_D, plot_2D_function, set_labels_and_legends
import optimization_module_neural_network as rfc
import autograd.numpy as np
import matplotlib.pyplot as plt

from autograd import grad

"""
    Different IC's and BC's can be subbed in easily - maybe... unsure about this.
    Might be the case for some special ICs or BCs, check Berg/Nystrom paper. 
"""

#BCs arent met
#plt.show seems to break. Need to fix plots. 

FILE_TO_STORE_G="/home/vassili/Desktop/Thesis/Honours_Thesis.git/data/G_func"

def get_resid_wave_eq(u):
    ux=grad(u,1)
    uxx=grad(ux,1)
    ut=grad(u, 2)
    utt=grad(ut, 2)
    resid=lambda params,x,t: np.square(utt(params, x, t)-uxx(params, x, t))
    return resid

def get_loss_function(u,nx, nt, L, t_max):
    #todo: make x, t generation independent of this. 
    t=np.linspace(t_max/8,t_max-t_max/8,nt)
    x=np.linspace(L/8, L-L/8, nx)
    X,T=np.meshgrid(x, t)

    resid=get_resid_wave_eq(u)
    def loss_function(params):
        sum=0.
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                sum=sum+resid(params, X[i,j], T[i,j])
        return sum/(X.shape[0]*X.shape[1])
    return loss_function
    
def optimize_u(G, params_G, D,params_D, layer_sizes, nx, nt, L, t_max, max_function_evals, max_iterations):
    """
    Returns u(x,t) to fit wave equation.
    """
    x0=rfc.init_random_params(1, layer_sizes)
    u=lambda params, x,t: G(params_G,x,t)+\
        D(params_D,x,t)*rfc.neural_net_predict(params, np.array([x,t]))

    loss_function=get_loss_function(u,nx, nt, L, t_max)
    loss_grad=grad(loss_function,0)
    p, fval=rfc.unflattened_lbfgs(loss_function, loss_grad, x0, \
        max_feval=max_function_evals, max_iter=max_iterations, callback=None)
    
    return u,p

def plot_result(u,G,D,param_u, param_g,param_D,t_max,L,nx,nt):
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

    fname=FILE_TO_STORE_G

    #IC
    g0expr='np.sin(x)'
    g1expr='np.cos(x)'
    #BC
    f0expr='0'
    f1expr='0'
    #Limits and number of points
    L=2*np.pi
    t_max=4
    nx=5
    nt=5
    #Network hyperparameters
    layer_sizes=[2,10,1]
    max_function_evals=100
    max_iterations=100

    G,param_G=create_or_load_G(g0expr, g1expr, f0expr, f1expr,L, t_max, 15, layer_sizes, fname, create_g=False)
    D=create_or_load_D(L)
    u, param_u=optimize_u(G,param_G,D,None,layer_sizes, nx, nt, L, t_max, max_function_evals, max_iterations)
    plot_result(u,G,D,param_u,param_G,None,t_max,L,10*nx,10*nt)
    
    

if __name__ == "__main__":
	test()