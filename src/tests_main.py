import class_definitions_main as cdm 
from autograd import grad
import autograd.numpy as np
import optimization_module_neural_network as optnn
from domain_generator import generate_L_shaped_domain, generate_square_domain
import loss_functions as lf
from helper import get_var_from_file
from matplotlib import pyplot as plt
import sys
LOCAL_PATH="/home/vassili/Desktop/Thesis/Honours_Thesis.git/"
FIG_PATH=LOCAL_PATH+"figs/"
def get_diffusion_residual(u, forcing_function):
    ux=grad(u,1)
    uxx=grad(ux,1)
    resid=lambda params,x: np.square(uxx(params,x)-forcing_function(x)) 
    return resid

def get_mass_spring_damper_residual(u, forcing_function):
    ux=grad(u,1)
    uxx=grad(ux,1)
    resid=lambda params,x: np.square(uxx(params,x)+u(params,x)-forcing_function(x)) 
    return resid

def test_diffusion_1D():
    forcing_function_expr='np.sin(x)'
    forcing_function_expr='0'
    diff1D=cdm.SingleVariable_PDE_Solver(nx=10, L=1, get_resid=get_diffusion_residual, \
        forcing_function_expr=forcing_function_expr, fig_dir=LOCAL_PATH+"figs/",var_id="x", unknown_id="u")
    diff1D.initialize_g_d_dirichlet(0,1)
    diff1D.solve(ls_phi=[1,10,1], max_feval=100)
    diff1D.plot_save_results("Diffusion1D")

def test_undamped_mass_spring_1D():
    nx=20
    forcing_function_expr='0'
    mass_spring1D=cdm.SingleVariable_PDE_Solver(nx=nx, L=2*np.pi, \
        get_resid=get_mass_spring_damper_residual, \
        forcing_function_expr=forcing_function_expr, fig_dir=LOCAL_PATH+"figs/", var_id="t", unknown_id="u", id="mass-spring",\
            local_path=LOCAL_PATH)
    mass_spring1D.initialize_g_d_ode(1,0)
    mass_spring1D.solve(ls_phi=[1,10,10, 1], max_feval=400, create_U=False)
    mass_spring1D.plot_save_results("MassSpring1D_Undamped_nx%d" % nx)


def run_and_save_mass_spring_runs_varying_nx(nx_range):
    for nx in nx_range:
        forcing_function_expr='0'
        mass_spring1D=cdm.SingleVariable_PDE_Solver(nx=nx, L=2*np.pi, \
            get_resid=get_mass_spring_damper_residual, \
            forcing_function_expr=forcing_function_expr, fig_dir=LOCAL_PATH+"figs/", var_id="t", unknown_id="u", id="mass-spring-nx%d" % nx,\
                local_path=LOCAL_PATH)
        mass_spring1D.initialize_g_d_ode(1,0)
        mass_spring1D.solve(ls_phi=[1,10,10, 1], max_feval=400, create_U=True)
        #mass_spring1D.plot_save_results("MassSpring1D_Undamped_nx%d" % nx)

def plot_convergence_mass_spring_runs_varying_nx(nx_range):
    exact_sln=lambda t: np.cos(t)

    L=2*np.pi
    fine_nx=30
    x_arr=np.linspace(0,L,fine_nx)
    err_arr=np.zeros(np.size(nx_range))
    
    for nx_idx, nx in enumerate(nx_range):
        id="mass-spring-nx%d" % nx
        data_file=LOCAL_PATH+"/data/trained_functions_"+id
        U=get_var_from_file(id+"U", data_file)
        
        for x in x_arr:
            err_arr[nx_idx]=err_arr[nx_idx]+\
                (exact_sln(x)-U(x))**2
        err_arr[nx_idx]=err_arr[nx_idx]/nx
    fig=plt.figure()
    ax=fig.gca()
    plt.plot(nx_range, err_arr)
    plt.yscale(value="log")
    plt.ylabel("$(u_{exact}-u_{approx})^2$")
    plt.xlabel("$N_t$")
    ax.set_aspect(aspect=0.5)
    plt.savefig(FIG_PATH+id+"_convergence_mass_spring.eps", bbox_inches="tight")
    plt.show(block=True)


def test_wave_1D():
    L=2*np.pi 
    t_max=3
    X, T=generate_square_domain(x_min=0., x_max=L, y_min=0., y_max=t_max, nx=5, ny=5)
    X_plot, T_plot=generate_square_domain(x_min=0., x_max=L, y_min=0., y_max=t_max, nx=30, ny=30)

    G_ansatz=lambda params, x,t: optnn.neural_net_predict(params, np.array([x,t]))
    G_ansatz=lambda params, x,t: np.sin(x)

    D_ansatz=lambda params, x,t: x*(L-x)*t**2*optnn.neural_net_predict(params, np.array([x,t]))
    G_loss=lf.get_G_loss_function_wave_eq(G_ansatz,g0=lambda x: np.sin(x),g1=lambda x:0.,\
        f0=lambda x: 0.,f1=lambda x:0,t_max=t_max,L=L,N=30)
    D_loss=lf.get_D_loss_function_wave_eq(D_ansatz,t_max=t_max,L=L,N=5)

    #sys.stdout = open(LOCAL_PATH+"shelloutput_wave1D", 'w')
    wave1D=cdm.TwoVariablesOneUnknown_PDE_Solver(\
        domain=[[X,T]],id="Wave1DNoExact", plot_domain=[[X_plot, T_plot]], \
        local_path=LOCAL_PATH,var_ids=["x","t","u"])
    #wave1D.exact_solution=lambda params,x,t: np.sin(x)*np.cos(t)
    wave1D.get_resid=lf.get_resid_wave_eq_1D
    wave1D.set_g_d(G_ansatz, D_ansatz, G_loss=None, D_loss=D_loss, layer_sizes=[2,10,10,1], \
        create_G=False, create_D=False, max_fun_evals=200)
    wave1D.solve(lf.get_resid_wave_eq_1D, [2,10,10,10,10,1], max_fun_evals=400, create_U=True)
    wave1D.plot_results()

def test_wave_1D_sine_function_basis():
    L=2*np.pi 
    t_max=3
    X, T=generate_square_domain(x_min=0., x_max=L, y_min=0., y_max=t_max, nx=5, ny=5)
    X_plot, T_plot=generate_square_domain(x_min=0., x_max=L, y_min=0., y_max=t_max, nx=30, ny=30)

    sys.stdout = open(LOCAL_PATH+"shelloutput_wave1D_sine", 'w')
    wave1D=cdm.TwoVariablesOneUnknown_PDE_Solver(\
        domain=[[X,T]],id="Wave1DSineFnBasis", plot_domain=[[X_plot, T_plot]], \
        local_path=LOCAL_PATH,var_ids=["x","t","u"])
    wave1D.exact_solution=lambda params,x,t: np.sin(x)*np.cos(t)
    wave1D.get_resid=lf.get_resid_wave_eq_1D
    x0=optnn.init_params_sine_series(scale=1., num_terms=3)
    U_ansatz=lambda params, x,t: optnn.sine_series_predict(params,[x,t])
    wave1D.solve(get_resid=lf.get_resid_wave_eq_1D, layer_sizes=None, max_fun_evals=200, create_U=True, \
        neural_net_basis=False, U_ansatz_if_not_neural=U_ansatz, x0=x0)
    wave1D.plot_results()


def test_laplace_2D_L_shape():
    #THIS DOESNT WORK FOR NOW. NEED LOSS FUNCTION FOR D. 
    #TODO: MAKE D LOSS. 
    L_shape=generate_L_shaped_domain(nx=10, ny=10, Lx1=1.,Lx2=2.,Ly1=1.,Ly2=2.)
    L_shape_plot=generate_L_shaped_domain(nx=30, ny=30, Lx1=1.,Lx2=2.,Ly1=1.,Ly2=2.)
    G_ansatz=lambda params, x,t: optnn.neural_net_predict(params, np.array([x,t]))
    D_ansatz=lambda params, x,t: optnn.neural_net_predict(params, np.array([x,t]))
    laplace2d=cdm.TwoVariablesOneUnknown_PDE_Solver(domain=L_shape,plot_domain=L_shape_plot, \
        id="Laplace2D", local_path=LOCAL_PATH, var_ids=["x","y","T"])
    laplace2d.set_g_d(G_ansatz, D_ansatz, G_loss=None, D_loss=None, layer_sizes=[2,10,10,1], \
        create_G=False, create_D=True)
    laplace2d.plot_results()

def test_laplace_square():
    Lx=4
    Ly=4
    X, Y=generate_square_domain(x_min=0., x_max=Lx, y_min=0., y_max=Ly, nx=4, ny=4)
    X_plot, Y_plot=generate_square_domain(x_min=0., x_max=Lx, y_min=0., y_max=Ly, nx=30, ny=30)

    G_ansatz=lambda params, x,y: 0
    D_ansatz=lambda params, x,y: x*(Lx-x)*y*(Ly-y)/(Lx*Ly)

    laplace2Dsquare=cdm.TwoVariablesOneUnknown_PDE_Solver(\
        domain=[[X,Y]],id="Laplace2D", plot_domain=[[X_plot, Y_plot]], \
        local_path=LOCAL_PATH,var_ids=["x","y","T"])

    laplace2Dsquare.set_g_d(G_ansatz, D_ansatz, G_loss=None, D_loss=None, layer_sizes=[2,10,10,1], \
        create_G=None, create_D=None, max_fun_evals=200)
    laplace2Dsquare.solve(lf.get_resid_Laplace_2D, [2,7,7,7,1], max_fun_evals=100, create_U=True)
    laplace2Dsquare.plot_results()

def test_neural_network_symbolic_derivative():
    return 0
def test():
    #test_diffusion_1D()
    #test_undamped_mass_spring_1D()
    #run_and_save_mass_spring_runs_varying_nx(nx_range=range(5,30))
    #plot_convergence_mass_spring_runs_varying_nx(nx_range=range(5,30))

    test_wave_1D()
    #test_laplace_2D_L_shape()
    #test_laplace_square()
    #test_wave_1D_sine_function_basis()
    return 0

if __name__ == "__main__":
	test()
