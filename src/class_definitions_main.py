import autograd.numpy as np
from autograd import grad
import optimization_module_neural_network as optnn
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
mpl.rcParams['font.size']=15;

class SingleVariable_PDE_Solver():
    def __init__(self, nx, L, get_resid,forcing_function_expr, fig_dir):
        self.L=L
        self.nx=nx
        self.G=None 
        self.D=None
        self.U=None
        self.p_U=None
        self.fig_dir=fig_dir
        self.get_resid=get_resid
        self.resid=None
        self.forcing_function_expr=forcing_function_expr
    def initialize_net_fun_from_params(self, layer_sizes):
        return lambda params, x: optnn.neural_net_predict(np.array(x))
    def initialize_g_d_dirichlet(self, f0,fL):
        self.G=lambda x: (fL-f0)/self.L*x+f0
        self.D=lambda x: x*(self.L-x)
    def initialize_g_d_ode(self, f0,fp0):
        self.G=lambda x: fp0*x+f0
        self.D=lambda x: 0.3*1/self.L*x**2
    def set_residual(self, U):
        forcing_function=lambda x: eval(self.forcing_function_expr)
        self.resid=self.get_resid(U, forcing_function)
    def get_loss_function(self):
        vresid=np.vectorize(self.resid,excluded=[0])
        x=np.linspace(0,self.L, self.nx)
        sum=0.
        def loss_function(params):
            res_arr=vresid(params, x)
            return np.sum(res_arr)/(res_arr.shape[0])
        return loss_function
    def solve(self, ls_phi, max_feval):
        Phi=lambda params, x: optnn.neural_net_predict(params, np.array(x))
        U=lambda params,x: self.G(x)+self.D(x)*Phi(params, x)
        x0=optnn.init_random_params(1,ls_phi)
        self.set_residual(U)
        loss_function=self.get_loss_function()
        loss_grad=grad(loss_function,0)
        p_U, _=optnn.unflattened_lbfgs(loss_function, loss_grad, x0, \
                max_feval=max_feval, max_iter=max_feval, callback=None)
        self.U=lambda x: U(p_U, x)
        self.p_U=p_U

    def plot_save_results(self, id):
        plt.figure()
        x=np.linspace(0,self.L,self.nx)
        
        u=np.zeros(x.shape)
        g=np.zeros(x.shape)
        d=np.zeros(x.shape)
        for i,x_entry in enumerate(x):
            u[i]=self.U(x_entry)
            g[i]=self.G(x_entry)
            d[i]=self.D(x_entry)
        plt.plot(x, u, label="$U(x)$")
        plt.plot(x, g, label="$G(x)$")
        plt.plot(x, d, label="$D(x)$")
        plt.xlabel("$x$")
        plt.legend()
        plt.savefig(self.fig_dir+id+"_ugd.eps")
        plt.show(block=True)
        vresid=np.vectorize(self.resid,excluded=[0])
        plt.figure()
        plt.plot(x, vresid(self.p_U,x), label="Residual")
        plt.xlabel("$x$")
        plt.legend()
        plt.savefig(self.fig_dir+id+"_resid.eps")
        plt.show(block=True)
        return 0