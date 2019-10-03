import autograd.numpy as np 

def generate_square_domain(x_min, x_max, y_min, y_max, nx, ny):
    x=np.linspace(x_min,x_max,nx)
    y=np.linspace(y_min,y_max,ny)
    X, Y=np.meshgrid(x,y)
    return X,Y
def generate_L_shaped_domain(nx, ny, Lx1,Lx2,Ly1,Ly2):
     """ 
     Generate L-shape domain, consists of two pieces X1,Y1 (bottom rectangle), X2,Y2 (top rectangle)
     X,Y are coordinate matrices. 
     Ly2 --- Lx1
     | X2,Y2  |
     |        |
     |------Ly1-----Lx2
     |       X1,Y1    |
     0,0----Lx1-----Lx2
     """
     X1,Y1=generate_square_domain(x_min=0, x_max=Lx2, y_min=0, y_max=Ly1,\
         nx=nx, ny=np.floor(ny/2))
     X2,Y2=generate_square_domain(x_min=0, x_max=Lx1, y_min=Ly1, y_max=Ly2,\
         nx=np.floor(nx/2), ny=ny)
     return [[X1, Y1], [X2,Y2]]