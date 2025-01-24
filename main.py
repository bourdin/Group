#%%
## Library
from SWE import SWE
test = SWE()
test.set("N", 200)
test.set("t_end", 10)
test.set("eta_s", test.get("eta_0"))
test.set("u_s", test.get("u_0"))
test.pflist()
#test.plot_sol()
test.numerical()
test.plot_result()


#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import integrate
from sklearn.metrics import mean_squared_error
import Function
from SWE import SWE

from matplotlib.animation import FuncAnimation
from matplotlib import rc
### Parameter
#numerical parameter
N = 200                              

#pysical parameter
x_end = 5                           ## end point of the end_domain
x_sta = -x_end                      ## start point of the end_domain
t_end = 10                           ## end point of the time

sigma = 1                           ## std for the initial surface perturvation
H = 1                               ## mean depth of the water
g = 1                               ## gravity


dx = (x_end-x_sta)/N                ## difference between two points of x
x = np.arange(x_sta,x_end,dx)       ## list of x axis

c=np.sqrt(g*H)                      ## wave speed
dt = dx/(2*c)                       ## step size (cfl criteria: less than )
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

### Boundary Condition
x_sol = np.append(x,x_end)          ## Add the end point(= initial point)
eta_sol = np.append(eta,eta[0])     ## Add the end point(= initial point)

    
# %%
# Experiment for the convergence test
import numpy as np
dx_data = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
N_data = [int((test.get("x_end") - test.get("x_sta")) / x) for x in dx_data]

# change dt, t_step also np.arange
data = [0 for row in range(np.size(N_data))]

for j in range(0,np.size(N_data)):
    test.set("N", N_data[j])
    test.set("dx", dx_data[j])
    test.numerical()
    test.error()

    data[j] = test.get("Nerror")

#%%
# Plot the convergence test result
import matplotlib as mpl
import matplotlib.pyplot as plt
f, axes = plt.subplots(figsize = (6,4))

plt.xscale("log")
plt.yscale("log")
plt.plot(dx_data,data)
plt.scatter(dx_data,data,label='L2 norm error',s=10)
axes.set_title(f"Convergence test result")
axes.set_xlabel(r"$dx$")
axes.set_ylabel(r"$Error$")
axes.legend()
axes.grid()
plt.show()

# %%
## Developed Function Test

# Parameter setting
test = SWE()
test.pflist()
# Plot the initial condition
test.plot_ini()
# Plot the solution graph
test.plot_sol()
# Plot after the integration
test.numerical()
test.plot_result()
# Calculate Error
test.error()
# Plot the error
test.plot_error()
# Make animation
test.animation()
