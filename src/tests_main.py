import class_definitions_main as cdm 
from autograd import grad
import autograd.numpy as np
LOCAL_PATH="/home/vassili/Desktop/Thesis/Honours_Thesis.git/"

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
    diff1D=cdm.SingleVariable_PDE_Solver(nx=10, L=1, get_resid=get_diffusion_residual, forcing_function_expr=forcing_function_expr, fig_dir=LOCAL_PATH+"figs/")
    diff1D.initialize_g_d_dirichlet(0,1)
    diff1D.solve(ls_phi=[1,10,1], max_feval=100)
    diff1D.plot_save_results("Diffusion1D")

def test_undamped_mass_spring_1D():
    forcing_function_expr='0'
    diff1D=cdm.SingleVariable_PDE_Solver(nx=10, L=2*np.pi, \
        get_resid=get_mass_spring_damper_residual, \
        forcing_function_expr=forcing_function_expr, fig_dir=LOCAL_PATH+"figs/")
    diff1D.initialize_g_d_ode(1,0)
    diff1D.solve(ls_phi=[1,10,10, 1], max_feval=200)
    diff1D.plot_save_results("MassSpring1D_Undamped")

def test():
    #test_diffusion_1D()
    test_undamped_mass_spring_1D()


if __name__ == "__main__":
	test()
