import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.extend import primitive, defvjp

from scipy.optimize import minimize
from autograd.misc import flatten
from autograd.wrap_util import wraps
from autograd.misc.optimizers import adam, sgd
from matplotlib import pyplot as plt

from scipy.optimize import basinhopping

@primitive
def relu(x):
  return x * (x > 0)

def relu_vjp(ans, x):
  return lambda x:(x>0).astype(float)

defvjp(relu, relu_vjp)

def init_random_params(scale, layer_sizes, rs=npr.RandomState()):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    
def init_ones_params(scale, layer_sizes, rs=npr.RandomState()):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * np.ones(m, n),   # weight matrix
             scale * np.ones(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b;
        inputs=np.tanh(outputs);
        #inputs=relu(outputs);
        #inputs = sigmoid(outputs);
    return outputs 


def my_unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(fun, grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _fun = lambda x: flatten(fun(unflatten(x)))[0]
        _grad = lambda x: flatten(grad(unflatten(x)))[0]
        
        if callback:
            _callback = lambda x: callback(unflatten(x))
        else:
            _callback = None
        result=optimize(_fun, _grad, _x0, _callback, *args, **kwargs)
        return unflatten(result.x), result.fun, result.func_values
        #return unflatten(optimize(_fun, _grad, _x0, _callback, *argss, **kwargs))

    return _optimize

@my_unflatten_optimizer
def my_lbfgs(fun, grad, x, minimizer_kwargs, num_iter=200, max_bfgs_iter=50, gtol=1e-03, ftol=1e-3):    
    func_values=[]

    def callback(x):
        func_values.append(np.log(fun(x)))


        
    minimizer_kwargs = {"method": "L-BFGS-B","options":{'disp': True,'maxiter':max_bfgs_iter, 
                                                         'gtol':gtol, 'ftol':ftol}, "jac":grad, 'callback':callback}
    #OR JUST BFGS
    #minimizer_kwargs = {"method": "BFGS","options":{'disp': True,'maxiter':max_bfgs_iter, 
     #                                                    'gtol':1e-08}, "jac":grad}
    
    
    ret= basinhopping(fun, x, minimizer_kwargs=minimizer_kwargs, niter=num_iter)
    
    ret.func_values=np.asarray(func_values)
    return ret