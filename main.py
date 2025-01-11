#%%
## Library
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import integrate
from sklearn.metrics import mean_squared_error
import Function

from matplotlib.animation import FuncAnimation
from matplotlib import rc

#%%
### Parameter
#numerical parameter
N = 200                              
''' number of cell'''


#pysical parameter
x_end = 5                           ## end point of the end_domain
x_sta = -x_end                      ## start point of the end_domain
t_end = 2                           ## end point of the time

sigma = 0.5                         ## std for the initial surface perturvation
H = 1                               ## mean depth of the water
g = 1                               ## gravity


dx = (x_end-x_sta)/N                ## difference between two points of x
x = np.arange(x_sta,x_end,dx)       ## list of x axis

c=np.sqrt(g*H)                      ## wave speed
dt = dx/2*c                         ## step size (cfl criteria: less than )
nu = 1                              ## stability constant
t_step=int(t_end/dt)                ## number of time steps
t = np.linspace(0,t_end,t_step)     ## list of t axis

### Variable
eta = np.zeros((N))                 ## eta is the surface perturvation
eta_0 = np.exp(-np.square(x)/sigma**2)      ## eta_0 is the initial eta
eta = eta_0  

eta_b = np.zeros((N))               ## eta_b is the bottom bathymetry
u = np.zeros((N))                   ## u is the horizontal verocity
u_0 = np.zeros((N))                 ## u_0 is the initial u
u = u_0                             
h = H + eta - eta_b                 ## h is depth of each point

#%% Plot the initial condition


f, axes = plt.subplots(figsize = (6,4))
plt.plot(x,eta_0,color = 'r',label='eta')
plt.plot(x+dx*0.5,u_0,color = 'b',label='u')
# plt.scatter(x,eta_0, color = 'r',label='point eta',s=5) #plot the exact points of eta
# plt.scatter(x+dx*0.5,u_0, color = 'b',label='point u',s=5) #plot the exact points of eta
plt.ylim(0, 1.5)
axes.set_title(f"Initial Condition with sigma = {sigma}, N = {N}, t_step = {t_step}")
axes.set_xlabel(f"$x$")
axes.set_ylabel(f"$eta$")
axes.legend()
axes.grid()
plt.show()


#%% Plot the solution graph

eta_s = 0.5*np.exp(-np.square(x-t_end*c)/sigma**2)+0.5*np.exp(-np.square(x+t_end*c)/sigma**2)
u_s = (0.5*np.exp(-np.square(x-t_end*c)/sigma**2)-0.5*np.exp(-np.square(x+t_end*c)/sigma**2))/c

f, axes = plt.subplots(figsize = (6,4))
plt.plot(x,eta_s,color = 'r',label='eta')
plt.plot(x+dx*0.5,u_s,color = 'b',label='u')
axes.set_title(f"Reference Solution with sigma = {sigma}, N = {N}, t_step = {t_step}")
axes.set_xlabel(r"$x$")
axes.set_ylabel(r"$eta$")
axes.legend()
axes.grid()
plt.show()


#%% Plot after the integration
eta_tmp=eta_0
eta=eta_0
u=u_0
u_tmp=u_0


def flux_ode(t, eta):
    return Function.Flux(u_tmp, h, H, dx, "linear") 
def bernoulli_ode(t, u):
    return Function.Bernoulli(u_tmp,eta_tmp,g,dx,"linear")
# solve_ivp 호출
for i in range(0,t_step):
    eta_tmp=eta
    u_tmp=u
    # Euler Method
    u=u+dt*bernoulli_ode(i*dt, u_tmp)
    eta = eta+dt*flux_ode(i*dt,eta_tmp)
    # sol = integrate.solve_ivp(bernoulli_ode,[i*dt,(i+1)*dt], u_tmp,method='RK23',t_eval=[dt*(i+1)])
    # u=sol.y.flatten()
    # sol = integrate.solve_ivp(flux_ode,[i*dt,(i+1)*dt], eta_tmp,method='RK23',t_eval=[dt*(i+1)])
    # eta=sol.y.flatten()
    h = H + eta - eta_b 


f, axes = plt.subplots(figsize = (6,4))

# Draw numerical results
plt.plot(x,eta,color = 'r',label='eta')
plt.plot(x+dx*0.5,u,color = 'b',label='u')

axes.set_title(f"Numerical Result with sigma = {sigma}, N = {N}, t_step = {t_step}")
axes.set_xlabel(r"$x$")
axes.set_ylabel(r"$eta$")
axes.legend()
axes.grid()
plt.show()

#%% Plot after the integration
eta_tmp=eta_0
eta=eta_0
u=u_0
u_tmp=u_0


def flux_ode(t, eta):
    return Function.Flux(u_tmp, h, H, dx, "linear") 
def bernoulli_ode(t, u):
    return Function.Bernoulli(u_tmp,eta_tmp,g,dx,"linear")
# solve_ivp 호출
for i in range(0,t_step):
    eta_tmp=eta
    u_tmp=u
    # Euler Method
    u=u+dt*bernoulli_ode(i*dt, u_tmp)
    eta = eta+dt*flux_ode(i*dt,eta_tmp)
    # sol = integrate.solve_ivp(bernoulli_ode,[i*dt,(i+1)*dt], u_tmp,method='RK23',t_eval=[dt*(i+1)])
    # u=sol.y.flatten()
    # sol = integrate.solve_ivp(flux_ode,[i*dt,(i+1)*dt], eta_tmp,method='RK23',t_eval=[dt*(i+1)])
    # eta=sol.y.flatten()
    h = H + eta - eta_b 


f, axes = plt.subplots(figsize = (6,4))

# Draw reference solution
plt.plot(x,eta_s,ls = '--',color = 'm',label='eta reference')
plt.plot(x+dx*0.5,u_s,ls = '--',color = 'c',label='u reference')

# Draw numerical results
plt.plot(x,eta,ls = '-',color = 'r',label='eta')
plt.plot(x+dx*0.5,u,ls = '-',color = 'b',label='u')

axes.set_title(f"Numerical Result with sigma = {sigma}, N = {N}, t_step = {t_step}")
axes.set_xlabel(r"$x$")
axes.set_ylabel(r"$eta$")
axes.legend()
axes.grid()
plt.show()

#%%
# Calculate Error
error_u = u_s - u
error_eta = eta_s - eta

# Plot the error
f, axes = plt.subplots(figsize = (6,4))
plt.plot(x,error_eta,color = 'r',label='eta error')
plt.plot(x+dx*0.5,error_u,color = 'b',label='u error')
axes.set_title(f"Error with sigma = {sigma}, N = {N}, t_step = {t_step}")
axes.set_xlabel(r"$x$")
axes.set_ylabel(r"$eta$")
axes.legend()
axes.grid()
plt.ylim(-1,1)
plt.show()

#%% Calculate RMSE
mse = mean_squared_error(eta_s, eta)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
# %%
data = [[0 for col in range(N)] for row in range(t_step)]

eta_tmp=eta_0
eta=eta_0
u=u_0
u_tmp=u_0


def flux_ode(t, eta):
    return Function.Flux(u_tmp, h, H, dx, "linear") 
def bernoulli_ode(t, u):
    return Function.Bernoulli(u_tmp,eta_tmp,g,dx,"linear")
# solve_ivp 호출
for i in range(0,t_step):
    eta_tmp=eta
    u_tmp=u
    # Euler Method
    u=u+dt*bernoulli_ode(i*dt, u_tmp)
    eta = eta+dt*flux_ode(i*dt,eta_tmp)
    # sol = integrate.solve_ivp(bernoulli_ode,[i*dt,(i+1)*dt], u_tmp,method='RK23',t_eval=[dt*(i+1)])
    # u=sol.y.flatten()
    # sol = integrate.solve_ivp(flux_ode,[i*dt,(i+1)*dt], eta_tmp,method='RK23',t_eval=[dt*(i+1)])
    # eta=sol.y.flatten()
    data[i] = eta
    h = H + eta - eta_b 


#%%
# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(x_sta, x_end, N)
ln, = plt.plot([], [], )

def init():    
    ax.set_xlim(x_sta - 1, x_end + 1)
    ax.set_ylim(-0.2, 1.2)

    # grid(True)를 해야 에니메이션에 grid가 나옴. 그냥 grid()는 안됨
    ax.grid(True)
      
    return ln,

def update(i):    
    # 주기: 2pi
    # y = np.sin(x-i) # i만클 shift하는 것임
    
    #  주기: pi
    y = data[i]
    
    ln.set_data(x, y)
    
    return ln,

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(199,t_step,200))),
                    init_func=init, interval=20, blit=True)

rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)
# %%
