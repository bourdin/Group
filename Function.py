#%%
import numpy as np

## Interpolation

def interpolation_node_to_edge(v, BoundaryCondition):
    '''
    :v: eta (node)
    :BoundaryCondition: Periodic, Infinite, SolidWall
    :return: interpolation of two points edge
    '''
    match BoundaryCondition:
        case "Periodic": return linear_interpolation(v,np.append(v[1:],v[0]))        
        case "SolidWall": return np.append(linear_interpolation(v[0:-1],v[1:]),v[-1])
        case "Open": return linear_interpolation(v,np.append(v[1:],v[-1]))
        #np.append(linear_interpolation(v[0:-1],v[1:]),v[-1])
    
def interpolation_edge_to_node(v, BoundaryCondition):
    '''
    :v: u (edge)
    :BoundaryCondition: Periodic, Infinite, SolidWall
    :return: interpolation of two points node
    '''
    match BoundaryCondition:
        case "Periodic": return linear_interpolation(v,np.append(v[-1],v[0:-1]))
        case "SolidWall": return linear_interpolation(v,np.append([0],v[0:-1]))
        case "Open": return linear_interpolation(v,np.append(v[0],v[0:-1]))
        #np.append(v[0],linear_interpolation(v[0:-1],v[1:]))
    
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
    :v1: vector such as u or eta in i-1/2
    :v2: vector such as u or eta in i+1/2
    :dx: increment of x
    :return: used to illustrate derivative
    '''
    return (v2-v1)/dx


# Flux and Bernoulli Function

def Flux(v1,v2,c,nu,dx,state,BoundaryCondition):
    '''
    :v1: u vector
    :v2: eta vector
    :c: constant H (mean depth)
    :dx: increment of x
    :return: used to illustrate flux
    '''
    match state:
        case "linear":
            f=-np.dot(v1,c)
            match BoundaryCondition:
                case "Periodic": return divergence(np.append(f[-1],f[0:-1]),f,dx)
                case "SolidWall": return divergence(np.append([0],f[0:-1]),f,dx)
                case "Open":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[1:-1],f[2:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
        case "non_linear":
            v2=interpolation_node_to_edge(v2, BoundaryCondition)
            f=-np.multiply(v1,v2+c)
            match BoundaryCondition:
                case "Periodic": return divergence(np.append(f[-1],f[0:-1]),f,dx)
                case "SolidWall": return divergence(np.append([0],f[0:-1]),f,dx)
        case "non_linear_s":
            match BoundaryCondition:
                case "Periodic":
                    hx=v1*2
                    hx[1:-1] = (v2[2:] - 2*v2[1:-1] + v2[:-2]) / dx**2
                    hx[0] = (v2[1] - 2*v2[0] + v2[-1]) / dx**2
                    hx[-1] = (v2[0] - 2*v2[-1] + v2[-2]) / dx**2
                    v2=interpolation_node_to_edge(v2, BoundaryCondition)
                    f=-np.multiply(v1,v2+c)
                    return divergence(np.append(f[-1],f[0:-1]),f,dx)+nu*hx
                case "SolidWall": 
                    hx=v1*2
                    hx[1:-1] = (v2[2:] - 2*v2[1:-1] + v2[:-2]) / dx**2
                    hx[0] = 0
                    hx[-1] = 0
                    v2=interpolation_node_to_edge(v2, BoundaryCondition)
                    f=-np.multiply(v1,v2+c)
                    return divergence(np.append([0],f[0:-1]),f,dx)+nu*hx

def Bernoulli(v1,v2,c,nu,dx,state,BoundaryCondition):
    '''
    :v1: u vector
    :v2: eta vector
    :c: constant g (acceleration of gravity)
    :dx: increment of x
    :return: used to illustrate Bernoulli
    '''
    match state:
        case "linear":
            f=-np.dot(v2,c)
            match BoundaryCondition:
                case "Periodic": return grad(f,np.append(f[1:],f[0]),dx)
                case "SolidWall": return np.append(grad(f[0:-1],f[1:],dx),[0])
                case "Open":
                    u_sta = c*derivative(v1[0],v1[1],dx)
                    u_end = -c*derivative(v1[-2],v1[-1],dx)
                    B = grad(f[0:-2],f[1:-1],dx)
                    return np.append(u_sta, np.append(B,u_end))
        case "non_linear":
            v1=interpolation_edge_to_node(v1, BoundaryCondition)
            f=-0.5*np.dot(v2,c)-v1**2
            match BoundaryCondition:
                case "Periodic": return grad(f,np.append(f[1:],f[0]),dx)
                case "SolidWall": return np.append(grad(f[0:-1],f[1:],dx),[0])
        case "non_linear_s":
            match BoundaryCondition:
                case "Periodic":
                    ux=(np.append(v1[-1],v1[0:-1])-2*v1+np.append(v1[1:],v1[0]))/dx**2
                    ux[-1]=0
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*np.dot(v2,c)-v1**2
                    return grad(f,np.append(f[1:],f[0]),dx)#+nu*ux
                case "SolidWall": 
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*np.dot(v2,c)-v1**2
                    return np.append(grad(f[0:-1],f[1:],dx),[0])
<<<<<<< HEAD
        
=======
        
>>>>>>> Infinite-BC
