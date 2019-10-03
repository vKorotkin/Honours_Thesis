import os
import autograd.numpy as np
import dill as pickle
import optimization_module_neural_network as rfc
from optimization_module_neural_network import get_parameters
from autograd import grad



def get_functions_from_strings(g0expr,g1expr,f0expr,f1expr):
    """
    Used to have identifiers when storing/retrieving specific G's,
    as we need a string to identify the BCs. 
    So instead of directly specifying g0 we specify a string g0expr that evaluates to it. 
    """
    g0=lambda x:eval(g0expr)
    g1=lambda x:eval(g1expr)
    f0=lambda t:eval(f0expr)
    f1=lambda t:eval(f1expr)
    return g0,g1,f0,f1

def save_fun_to_file(fun,p,g0expr,g1expr,f0expr,f1expr,fname):
    """
    Save function fun and its parameters p into a dict entry, with dict stored in fname. 
    fun_dict[identifier]['Fun'] has fun 
    fun_dict[identifier]['Parameters'] has its parameters (as its usually a neural net.)
    """
    identifier="g0: %s, g1: %s, f0: %s, f1: %s" % (g0expr, g1expr, f0expr, f1expr)
    fun_dict={}

    if os.path.isfile(fname):
        if os.path.getsize(fname) > 0:
            fun_dict = pickle.load(open(fname, "rb"))
        fun_dict[identifier]={'Fun':fun, 'Parameters':p}
        pickle.dump(fun_dict, open(fname, "wb"))
    else:
        fun_dict = {identifier: {'Fun':fun, 'Parameters':p}}
        pickle.dump(fun_dict, open(fname, "wb"))

def save_var_to_file(var, identifier, fname):
    if os.path.isfile(fname):
        if os.path.getsize(fname) > 0:
            fun_dict = pickle.load(open(fname, "rb"))
        fun_dict[identifier]=var
        pickle.dump(fun_dict, open(fname, "wb"))
    else:
        fun_dict = {identifier: var}
        pickle.dump(fun_dict, open(fname, "wb"))

def get_var_from_file(identifier, fname):
    fun_dict = pickle.load(open(fname, "rb"))
    var=fun_dict[identifier]
    return var

def get_fun_from_file(fname, identifier):
    """
    See save_fun_to_file
    """
    fun_dict = pickle.load(open(fname, "rb"))
    fun=fun_dict[identifier]['Fun']
    p=fun_dict[identifier]['Parameters']
    return fun,p



def create_or_load_trained_f(f,loss_function,g0expr, g1expr, f0expr, f1expr,L, t_max, layer_sizes, fname, N=15, maxiter=100,maxfuneval=100, create_f=True):
    """
    Either optimize a function f to minimize loss function and return f and parameters, or load an existing one from file 
    Parameters
    ----------
    f: function f(params,x,t) to train. 
    loss_function: loss_function(params) to optimize over
    g0,g1,f0,f1expr's: expressions for the boundary/initial conditions. 
    """
    identifier="g0: %s, g1: %s, f0: %s, f1: %s" % (g0expr, g1expr, f0expr, f1expr)
    if create_f:
        g0,g1,f0,f1=get_functions_from_strings(g0expr,g1expr,f0expr,f1expr)
        p=get_parameters(loss_function,layer_sizes, maxiter,maxfuneval)
        save_fun_to_file(f,p,g0expr,g1expr,f0expr,f1expr,fname)
    else:
        f,p=get_fun_from_file(fname, identifier)
    return f,p