#%%
import numpy as np

## Function for the interpolation

def interpolation_node_to_edge(v, BoundaryCondition):
    '''
    :v: eta (node)
    :BoundaryCondition: Periodic, Open, SolidWall
    :return: interpolation of two points edge
    '''
    if (BoundaryCondition=="Periodic"):
        v[-1]=v[0]
    return linear_interpolation(v[:-1],v[1:])

def interpolation_edge_to_node(v, BoundaryCondition):
    '''
    :v: u (edge)
    :BoundaryCondition: Periodic, Open, SolidWall
    :return: interpolation of two points node
    '''
    match BoundaryCondition:
        case "Periodic":
            return linear_interpolation(np.append(v[-1],v),np.append(v,v[0]))
        case "SolidWall":
            return linear_interpolation(np.append(0,v),np.append(v,0))
        case "Open":
            return linear_interpolation(np.append(v[0],v),np.append(v,v[-1]))
        case "L_Solid_R_Open":
            return linear_interpolation(np.append(0,v),np.append(v,v[-1]))
        case "L_Open_R_Solid":
            return linear_interpolation(np.append(v[0],v),np.append(v,0))
    
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

def Flux(v1,v2,c,b,nu,dx,state,BoundaryCondition):
    '''
    :v1: u vector
    :v2: eta vector
    :c: constant H (mean depth)
    :dx: increment of x
    :return: used to illustrate flux
    '''
    match state:
        case "linear":
            f=-v1*c
            match BoundaryCondition:
                case "Periodic": return divergence(np.append(f[-1],f),np.append(f,f[0]),dx)
                case "SolidWall": return divergence(np.append(0,f),np.append(f,0),dx)
                case "Open":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Solid_R_Open":
                    eta_sta = derivative(0,f[0],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Open_R_Solid":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = derivative(f[-1],0,dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
        case "non_linear":
            v2=interpolation_node_to_edge(v2, BoundaryCondition)
            b=interpolation_node_to_edge(b, BoundaryCondition)
            f=-np.multiply(v1,v2+c-b)
            match BoundaryCondition:
                case "Periodic": return divergence(np.append(f[-1],f),np.append(f,f[0]),dx)
                case "SolidWall": return divergence(np.append(0,f),np.append(f,0),dx)
                case "Open":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Solid_R_Open":
                    eta_sta = derivative(0,f[0],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Open_R_Solid":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = derivative(f[-1],0,dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
        case "non_linear_s":
            v2=interpolation_node_to_edge(v2, BoundaryCondition)
            b=interpolation_node_to_edge(b, BoundaryCondition)
            f=-np.multiply(v1,v2+c-b)
            match BoundaryCondition:
                case "Periodic": return divergence(np.append(f[-1],f),np.append(f,f[0]),dx)
                case "SolidWall": return divergence(np.append(0,f),np.append(f,0),dx)
                case "Open":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Solid_R_Open":
                    eta_sta = derivative(0,f[0],dx)
                    eta_end = -c*derivative(v2[-2],v2[-1],dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))
                case "L_Open_R_Solid":
                    eta_sta = c*derivative(v2[0],v2[1],dx)
                    eta_end = derivative(f[-1],0,dx)
                    F = divergence(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F,eta_end))

def Bernoulli(v1,v2,c,b,nu,dx,state,BoundaryCondition):
    '''
    :v1: u vector
    :v2: eta vector
    :c: constant g (acceleration of gravity)
    :dx: increment of x
    :return: used to illustrate Bernoulli
    '''
    match state:
        case "linear":
            f=-v2*c
            if (BoundaryCondition=="Periodic"):
                f[-1]=f[0]
            return grad(f[:-1],f[1:],dx)
        case "non_linear":
            v1=interpolation_edge_to_node(v1, BoundaryCondition)
            f=-0.5*v2*c-v1**2
            return grad(f[:-1],f[1:],dx)
        case "non_linear_s": 
            eta_x = v1*2
            match BoundaryCondition:
                case "Periodic":
                    eta_x [1:-1] = (v1[2:] - 2*v1[1:-1] + v1[:-2]) / dx**2
                    eta_x[0] = (v1[1] - 2*v1[0] + v1[-1]) / dx**2
                    eta_x[-1] = (v1[0] - 2*v1[-1] + v1[-2]) / dx**2
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*v2*c-v1**2
                    return grad(f[:-1],f[1:],dx)+nu*eta_x
                case "SolidWall":
                    eta_x [1:-1] = (v1[2:] - 2*v1[1:-1] + v1[:-2]) / dx**2
                    eta_x[0] = 0
                    eta_x[-1] = 0
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*v2*c-v1**2
                    return grad(f[:-1],f[1:],dx)+nu*eta_x
                case "Open":
                    eta_x [1:-1] = (v1[2:] - 2*v1[1:-1] + v1[:-2]) / dx**2
                    eta_x[0] = 0
                    eta_x[-1] = 0
                    eta_sta = c*derivative(v1[0],v1[1],dx)
                    eta_end = -c*derivative(v1[-2],v1[-1],dx)
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*v2*c-v1**2
                    F=grad(f[:-1],f[1:],dx)
                    return np.append(eta_sta, np.append(F[1:-1],eta_end))+nu*eta_x
                case "L_Solid_R_Open":
                    eta_x = v1*2
                    eta_x [1:-1] = (v1[2:] - 2*v1[1:-1] + v1[:-2]) / dx**2
                    eta_x[0] = 0
                    eta_x[-1] = 0                    
                    eta_end = -c*derivative(v1[-2],v1[-1],dx)
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*v2*c-v1**2
                    F=grad(f[:-1],f[1:],dx)
                    return np.append(F[:-1],eta_end)+nu*eta_x
                case "L_Open_R_Solid":
                    eta_x = v1*2
                    eta_x [1:-1] = (v1[2:] - 2*v1[1:-1] + v1[:-2]) / dx**2
                    eta_x[0] = 0
                    eta_x[-1] = 0
                    eta_sta = c*derivative(v1[0],v1[1],dx)
                    v1=interpolation_edge_to_node(v1, BoundaryCondition)
                    f=-0.5*v2*c-v1**2
                    F=grad(f[:-1],f[1:],dx)
                    return np.append(eta_sta, F[1:])+nu*eta_x
            
        

# %%
