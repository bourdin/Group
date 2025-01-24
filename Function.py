#%%
import numpy as np

## Interpolation

def interpolation_node_to_edge(v):
    '''
    :v: eta (node)
    :return: interpolation of two points edge
    '''
    return linear_interpolation(v,np.insert([v[0]],0,v[1:]))

def interpolation_edge_to_node(v):
    '''
    :v: u (edge)
    :return: interpolation of two points node
    '''
    return linear_interpolation(v,np.insert(v[0:-1],0,v[-1]))
    # return np.insert(linear_interpolation(v[0:-1],v[1:]),0,[0])

def linear_interpolation(v1,v2):
    '''
    :v1: u or eta
    :v2: u or eta which next to v1
    :return: Calculate linear interpolation: mean
    '''
    return 0.5*(v1+v2)

## 1D version divergence

def divergence(f1,f2,dx):
    '''
    :f1: vector such as u or eta
    :f2: vector such as u or eta
    :dx: increment of x
    :return: used to illustrate divergence
    '''
    return derivative(f1,f2,dx)

def grad(f1,f2,dx):
    '''
    :f1: vector such as u or eta
    :f2: vector such as u or eta
    :return: used to illustrate gradient
    \n same with divergence (since it's 1D version)
    '''
    return derivative(f1,f2,dx)

def Laplacian(f1,dx):
    '''
    :f1: vector such as u or eta
    :dx: increment of x
    :return: used to illustrate Laplacian
    '''
    return divergence(grad(f1,dx))

def derivative(v1,v2,dx):
    '''
    :v1: vector such as u or eta
    :v2: vector such as u or eta next time step
    :dx: increment of x
    :return: used to illustrate derivative
    '''
    return (v2-v1)/dx


# Flux and Bernoulli Function

def Flux(v1,v3,c,dx,state):
    '''
    :v1: u vector
    :v3: h vector
    :c: constant H (mean depth)
    :dx: increment of x
    :return: used to illustrate flux
    '''
    match state:                                                        ## match need python 3.10
        case "non_linear":
            v3=interpolation_edge_to_node(v3)
            f=-np.multiply(v1,v3)
            return divergence(np.insert(f[0:-1],0,f[-1]),f,dx)
        case "linear":
            f=-np.dot(v1,c)
            return divergence(np.insert(f[0:-1],0,f[-1]),f,dx)

def Bernoulli(v1,v2,c,dx,state):
    '''
    :v1: u vector
    :v2: eta vector
    :c: constant g (acceleration of gravity)
    :dx: increment of x
    :return: used to illustrate Bernoulli
    '''
    match state:
        case "non_linear":
            v1=interpolation_node_to_edge(v1)
            f=-0.5*np.square(v1)+np.dot(v2,c)
            return grad(f,np.insert([f[0]],0,f[1:]),dx)
        case "linear":
            f=-np.dot(v2,c)
            return grad(f,np.insert([f[0]],0,f[1:]),dx)
                
#%% integrater
# use scipy libary
# import scipy.integrate
