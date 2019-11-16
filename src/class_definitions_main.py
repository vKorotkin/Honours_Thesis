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
    def __init__(self, nx, L, get_resid,forcing_function_expr, fig_dir, var_id, unknown_id, id, local_path):
        """
        Parameters
        -----------------
        nx: Number of collocation points in x direction
        L: length of domain. x in [0,L]
        get_resid: A function which returns a another function which is the residual of the PDE. 
            Note: This "function to return a function" is needed to get residuals for different geometries (number of points, etc..)
        fig_dir: Directory to save figures to. Should include the "/" at the end. 
        var_id: Id for independent variable e.g. x, or t. Used to plot in graphs.
        unknown_id: Id for unknown, e.g. u. 
            NOTE: For ids do not include the latex $$. 
        """
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
        self.var_id=var_id
        self.unknown_id=unknown_id
        self.data_file=local_path+"/data/trained_functions_"+id
        self.id=id
        self.local_path=local_path

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
    def solve(self, ls_phi, max_feval, create_U):
        Phi=lambda params, x: optnn.neural_net_predict(params, np.array(x))
        U=lambda params,x: self.G(x)+self.D(x)*Phi(params, x)
        x0=optnn.init_random_params(1,ls_phi)
        self.set_residual(U)
        loss_function=self.get_loss_function()
        loss_grad=grad(loss_function,0)
        if create_U:
            p_U, _=optnn.unflattened_lbfgs(loss_function, loss_grad, x0, \
                    max_feval=max_feval, max_iter=max_feval, callback=None)
            self.U=lambda x: U(p_U, x)
            self.p_U=p_U
            save_var_to_file(self.U, self.id+"U", self.data_file)
            save_var_to_file(self.p_U, self.id+"pstar_U", self.data_file)
        else:
            self.U=get_var_from_file(self.id+"U", self.data_file)
            self.p_U=get_var_from_file(self.id+"pstar_U", self.data_file)

    def plot_save_results(self, id):
        Phi=lambda x:(self.U(x)-self.G(x))/self.D(x)
        plt.figure()
        x=np.linspace(0,self.L,self.nx)
        
        u=np.zeros(x.shape)
        g=np.zeros(x.shape)
        d=np.zeros(x.shape)
        phi=np.zeros(x.shape)
        for i,x_i in enumerate(x):
            u[i]=self.U(x_i)
            g[i]=self.G(x_i)
            d[i]=self.D(x_i)
            phi[i]=Phi(x_i)
        # PLOT COMPONENTS: G,D, Phi
        plt.plot(x, g, label="$G(%s)$" % self.var_id, color='g')
        plt.plot(x, d, label="$D(%s)$" % self.var_id, color='tab:olive')
        plt.plot(x, phi, label="$\Phi(%s)$" % self.var_id, color='r')
        plt.xlabel("$%s$" % self.var_id)
        plt.legend()
        plt.savefig(self.fig_dir+id+"_ugd.eps")
        #plt.show(block=True)
        # PLOT U, the solution
        plt.figure()
        plt.plot(x, u, label="$U(%s)$" % self.var_id, color='k')
        plt.xlabel("$%s$" % self.var_id)
        #plt.ylabel("$U(%s)$" % self.var_id)
        plt.legend()
        plt.savefig(self.fig_dir+id+"_u.eps")
        #plt.show(block=True)
        #PLOT RESIDUAL
        vresid=np.vectorize(self.resid,excluded=[0])
        plt.figure()
        
        plt.plot(x, vresid(self.p_U,x),label="$R(%s)$" % self.var_id)
        plt.ylabel("$R(%s)$" % self.var_id)
        plt.xlabel("$%s$" % self.var_id)
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
        self.fig_dir=local_path+"figs/"
        self.id=id
        self.G=None
        self.D=None 
        self.U=None
        self.resid=None
        self.get_resid=None
        self.pstar_U=None
        self.exact_solution=None
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
    def solve(self, get_resid, layer_sizes, max_fun_evals, create_U=True, neural_net_basis=True, U_ansatz_if_not_neural=None, x0=None):
        """
        Optimizes the solution ansatz U to minimize the local equation defined by get_resid. 
        The boundary and initial conditions are assumed to already have been fit by self.G and self.D
        By default, a neural network basis is used i.e. neural_net_basis=True. 
        If a different basis is used, U_ansatz_if_not_neural and x0 need to be given. 
        """
        if neural_net_basis:
            U=lambda params,x,y:self.G(x,y)+self.D(x,y)*optnn.neural_net_predict(params,np.array([x,y]).reshape(1,2))
            x0=optnn.init_random_params(1, layer_sizes) 
        else:
            U=U_ansatz_if_not_neural

        resid=get_resid(U)
        if create_U:
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
        plt.title("$%s(%s,%s)$" % (fun_id, self.var_ids[0], self.var_ids[1]))
        plt.xlabel("$"+self.var_ids[0]+"$")
        plt.ylabel("$"+self.var_ids[1]+"$")
        ax.set_zlabel("$"+self.var_ids[2]+"$")
    def plot_results(self):
        num_rows=2
        if self.exact_solution is not None and self.get_resid is not None:
            num_rows=3
        fig=plt.figure()
        if self.G is not None:
            ax = fig.add_subplot(num_rows*100+21, projection='3d')
            self.plot_quantity(ax, self.G, "G")
        if self.D is not None:
            ax = fig.add_subplot(num_rows*100+22, projection='3d')
            self.plot_quantity(ax, self.D, "D")
        if self.U is not None:
            #self.resid=self.get_resid(self.U)
            ax = fig.add_subplot(num_rows*100+23, projection='3d')
            self.plot_quantity(ax, self.U, self.var_ids[2])
            ax = fig.add_subplot(num_rows*100+24, projection='3d')
            self.plot_quantity(ax, lambda x,y: self.resid(self.pstar_U, x,y), "r")
        if self.exact_solution is not None and self.get_resid is not None: 
            resid=self.get_resid(self.exact_solution)
            ax = fig.add_subplot(num_rows*100+25, projection='3d')
            self.plot_quantity(ax, lambda x,y: self.exact_solution(None, x,y), "u_{exact}")
            ax = fig.add_subplot(num_rows*100+26, projection='3d')
            self.plot_quantity(ax, lambda x,y: resid(None, x,y), "r_{exact}")
        plt.savefig(self.fig_dir+self.id+"_ugd.eps")
        plt.show(block=True)