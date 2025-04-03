#%%
## Library
from SWE import SWE
import numpy as np
test = SWE()
test.set("BoundaryCondition","L_Open_R_Solid")
test.set("t_end",17)
test.set("cnu",0.2)
test.set("eta_b",np.linspace(-0.3,.0,200))
# test.numerical(linearity = "linear")
# test.plot_result()
test.animation(linearity = "linear")

# test.set("BoundaryCondition","SolidWall")
# test.set("BoundaryCondition","Open")
# test.set("N",600)
# test.set("x_end",15)
# test.set("x_sta",-15)
# test.set("eta_0",np.exp(-np.square(test.get('x')-2/test.get('sigma')**2)))
# test.set("eta_0",np.append(np.append(np.zeros(285),np.zeros(30)+0.5),np.zeros(285)))
# test.plot_ini()
# test.set("t_end",4)
# test.set("cnu",0.2)
# test.pflist()
# test.numerical(linearity = "non_linear_s")
# test.numerical(linearity = "non_linear")
# test.numerical(linearity = "linear")
# test.plot_result()
# test.ConvergenceTest()
# test.animation()
# test.plot_sol()
#%% Infinite boudary condition check
from SWE import SWE
Btest =SWE()
Btest.set("stabilizer","off")
Btest.set("t_end",4)
Btest.pflist()
Btest.set("BoundaryCondition","Infinite")
Btest.numerical(linearity = "linear")
# Btest.set("BoundaryCondition","SolidWall")
# Btest.numerical(linearity = "non_linear")
Btest.plot_result()


#%%
import Function as F
import numpy as np
import matplotlib.pyplot as plt
#%%


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
