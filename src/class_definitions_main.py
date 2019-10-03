import autograd.numpy as np
from autograd import grad
import optimization_module_neural_network as optnn
import matplotlib as mpl
import matplotlib.pyplot as plt
from helper import save_var_to_file, get_var_from_file
from helper_plotting import plot_2D_function_general_domain

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
    @staticmethod
    def initialize_net_fun_from_params(layer_sizes):
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

class TwoVariablesOneUnknown_PDE_Solver():
    """A class to solve two variable one unknown PDEs, e.g. 1D wave EQ, 2D Laplace
    Attributes:
        domain: list of lists that each contain two coordinate matrices. 
            wave eq: [[X,T]] (single rectangular domain, space-time)
            laplace 2d: [[X1,Y1], [X2, Y2]] (union of two rectangular domains to make l-shaped plate)
        var_ids: list of strings, identifying each variable. [var1, var2, unknown]
    """
    def __init__(self,domain, plot_domain, id, local_path, var_ids):
        self.domain=domain
        self.plot_domain=plot_domain
        self.local_path=local_path
        self.data_file=local_path+"/data/trained_functions_"+id
        self.var_ids=var_ids
        self.id=id
        self.G=None
        self.D=None 
        self.U=None
        self.resid=None
        self.pstar_U=None
    def set_g_d(self, G_ansatz, D_ansatz, G_loss, D_loss, layer_sizes, max_fun_evals=200, create_G=True, create_D=True):
        p0=optnn.init_random_params(1, layer_sizes)

        if G_loss is not None:
            if create_G==True:
                self.G, _=self.optimize_func(G_ansatz, p0, G_loss, max_fun_evals)
                save_var_to_file(self.G, self.id+"G", self.data_file)
            else:
                self.G=get_var_from_file(self.id+"G", self.data_file)
        else: 
            self.G=lambda x,y: G_ansatz(None, x,y)

        if D_loss is not None:
            if create_D==True:
                self.D, _=self.optimize_func(D_ansatz, p0, D_loss, max_fun_evals)
                save_var_to_file(self.D, self.id+"D", self.data_file)
            else:
                self.D=get_var_from_file(self.id+"D", self.data_file)
        else: 
            self.D=lambda x,y: D_ansatz(None, x,y)
    @staticmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    def optimize_func(f_ansatz, x0, loss_function, max_fun_evals):
        loss_grad=grad(loss_function,0)
        p, _=optnn.unflattened_lbfgs(loss_function, loss_grad, x0, \
            max_feval=max_fun_evals, max_iter=max_fun_evals, callback=None)
        fstar=lambda x,y: f_ansatz(p,x,y)
        return fstar, p
    def get_local_eq_loss_function(self, resid):
        vresid=np.vectorize(resid,excluded=[0])
        def loss_function(params):
            sum=0.
            for (X,Y) in self.domain:
                res_arr=vresid(params,X, Y)
                sum = sum + np.sum(res_arr)/(res_arr.shape[0]*res_arr.shape[1])
            return sum
        return loss_function
    def solve(self, get_resid, layer_sizes, max_fun_evals, create_U=True):
        U=lambda params,x,y:self.G(x,y)+self.D(x,y)*optnn.neural_net_predict(params,np.array([x,y]).reshape(1,2))
        resid=get_resid(U)
        if create_U:
            x0=optnn.init_random_params(1, layer_sizes)    
            self.U, self.pstar_U=self.optimize_func(U, x0, self.get_local_eq_loss_function(resid), max_fun_evals)
            save_var_to_file(self.U, self.id+"U", self.data_file)
            save_var_to_file(self.pstar_U, self.id+"pstar_U", self.data_file)
        else:
            self.U=get_var_from_file(self.id+"U", self.data_file)
            self.pstar_U=get_var_from_file(self.id+"pstar_U", self.data_file)
        self.resid=resid
        return 0
        
    def plot_quantity(self, ax, quantity_fun, fun_id):
        plot_2D_function_general_domain(ax, self.plot_domain, quantity_fun, "$%s$" % fun_id)
        #plt.title("%s: $G(%s, %s)$" % (self.id, self.var_ids[0], self.var_ids[1]))
        plt.title("%s: $%s(%s,%s)$" % (self.id, fun_id, self.var_ids[0], self.var_ids[1]))
        plt.xlabel("$"+self.var_ids[0]+"$")
        plt.ylabel("$"+self.var_ids[1]+"$")
        ax.set_zlabel("$"+self.var_ids[2]+"$")
    def plot_results(self):
        fig=plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        self.plot_quantity(ax, self.G, "G")
        ax = fig.add_subplot(222, projection='3d')
        self.plot_quantity(ax, self.D, "D")
        if self.U is not None:
            ax = fig.add_subplot(223, projection='3d')
            self.plot_quantity(ax, self.U, "U")
            ax = fig.add_subplot(224, projection='3d')
            self.plot_quantity(ax, lambda x,y: self.resid(self.pstar_U, x,y), "Resid")
        plt.show(block=True)