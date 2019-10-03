
import autograd.numpy as np
from autograd import grad
from domain_generator import generate_L_shaped_domain

def get_G_loss_function_wave_eq(G,g0,g1,f0,f1,t_max,L,N):
    """
    Returns loss function for G.
    g0: Initial condition, position. u(x,0)=g0(x)
    g1: Initial condition, velocity du/dt(x,0)=g1(x)
    f0: Boundary condition, x=0 u(0,t)=f0(t)
    f1: Boundary condition, x=1 u(L,t)=f1(t)
    """
    dGdt=grad(G,2)

    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)

    def loss_function(params):
        sum=0.
        for x in x_space:
            sum=sum+(G(params,x,0.)-g0(x))**2+(dGdt(params,x,0.)-g1(x))**2
        for t in t_space:
            sum=sum+(G(params,0.,t)-f0(x))**2+(G(params,L,t)-f1(x))**2
        
        return sum

    return loss_function

def get_D_loss_function_wave_eq(D,t_max,L,N):
    """
    Returns loss function for D.
    """
    t_space=np.linspace(0,t_max,N)
    x_space=np.linspace(0, L, N)
    X,T=np.meshgrid(x_space, t_space)
    dDdx=grad(D,1)
    dDdx2=grad(dDdx,1)
    dDdt=grad(D,2)
    dDdt2=grad(dDdt,2)
    #Leave the first few times alone, because the ansatz for D has the t^2 term that defines the behaviour for small t. 
    min_t_idx=int(np.floor(X.shape[1]/10))
    def loss_function(params):
        sum=0.
        #int(np.floor(X.shape[1]/3)),
        for i in range(X.shape[0]):
            for j in range(min_t_idx, X.shape[1]):
                sum=sum+np.square(D(params, X[i,j], T[i,j])-X[i,j]*(L-X[i,j]))
                    #+np.square(dDdt2(params, X[i,j], T[i,j]))+np.square(dDdx2(params, X[i,j], T[i,j]))
                    #+np.square(dDdx2(params, X[i,j], T[i,j]))
                #+np.square(D(params, X[i,j], T[i,j])-1) \
                    #
                #sum=sum+np.square(D(params, X[i,j], T[i,j])-np.tanh(T[i,j]))
        #for t in t_space:
        #    sum=sum+np.square(dDdx(params, 0., t))+np.square(dDdx(params, L, t))

        return sum/(X.shape[0]*X.shape[1])
    return loss_function


#TODO
def get_smallest_distance_L_shape(x,y,Lx1,Lx2,Ly1,Ly2):
    #crap this is for infinite lines, might have problems. 
    point_to_line_smallest_dist=lambda a,b,c,x,y:np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)
    #d=min(point_to_line_smallest_dist())
    return 0
def get_D_loss_function_laplace_2D_L_shape(nx=30, ny=30, Lx1=1.,Lx2=2.,Ly1=1.,Ly2=2.):
    L_shape=generate_L_shaped_domain(nx, ny, Lx1,Lx2,Ly1,Ly2)
    smallest_dist=[]
    # Get array containing minimum distance to border
    for (X,Y) in L_shape:
        smallest_dist.append(np.zeros(X.shape))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                smallest_dist[i][j]=get_smallest_distance_L_shape(X[i,j],Y[i,j],Lx1,Lx2,Ly1,Ly2)
    #Optimize D to fit the minimum distance to border
    return 0

def get_resid_wave_eq_1D(u):
    """
    Generate wave equation residual function from solution ansatz u
    
    Returns
    -------
    resid: function resid(params,x,t). Returns residual of PDE from neural network parameters and x,t space and time positions. 
    """
    ux=grad(u,1)
    uxx=grad(ux,1)
    ut=grad(u, 2)
    utt=grad(ut, 2)
    resid=lambda params,x,t: np.square(utt(params, x, t)-uxx(params, x, t))
    return resid

def get_resid_Laplace_2D(u):
    """
    Generate wave equation residual function from solution ansatz u
    
    Returns
    -------
    resid: function resid(params,x,t). Returns residual of PDE from neural network parameters and x,t space and time positions. 
    """
    ux=grad(u,1)
    uxx=grad(ux,1)
    uy=grad(u, 2)
    uyy=grad(uy, 2)
    resid=lambda params,x,t: np.square(uxx(params, x, t)+uyy(params, x, t)+1.)
    return resid

