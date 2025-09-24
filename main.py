#%%
## Library
from SWE import SWE
import numpy as np
test = SWE()
b = 1
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
test.set("BoundaryCondition",boundary)
test.set("t_end",1)
test.set("cnu",0.3)
test.set("eta_0",np.exp(-np.square(test.get('x')+1)/test.get('sigma')**2))
test.set("eta_b",0.08*(test.get('x')-5))
test.numerical(linearity = "non_linear_s")
test.plot_result()
# test.animation(linearity = "non_linear_s")
test.plot_ini(["eta","b"])
# test.set("eta_s", test.get("eta_0"))
# test.set("u_s", test.get("u_0"))


#%%
test.set("BoundaryCondition","SolidWall")
test.set("H",5)
test.set("eta_0",np.exp(-np.square(test.get('x')+1)/test.get('sigma')**2))
test.set("t_end",18)
test.set("cnu",0.3)
# test.set("eta_b",-np.ones(200))
test.set("eta_b",0.05*(test.get('x')-5))

# test.animation(linearity = "linear")

test.numerical(linearity = "linear")
test.plot_result()
test.plot_ini(list = ["eta","b"])

#%%
test.set("N",600)
test.set("x_end",15)
test.set("x_sta",-15)
test.set("eta_0",np.exp(-np.square(test.get('x')+10)/test.get('sigma')**2))
test.set("t_end",2)
test.set("cnu",0.3)

test.set("eta_b",np.append(np.append(np.zeros(200),np.zeros(200)+0.5),np.zeros(200))-1)
# test.set("eta_b",np.append(np.append(np.zeros(275),np.zeros(50)+1),np.zeros(275))-1)
# test.set("eta_b",np.zeros(test.get("N"))-1)

# test.numerical(linearity = "non_linear_s")
test.numerical(linearity = "linear")
test.plot_result()
test.plot_ini(list = ["eta","b"], size_a= 15)

# test.animation(linearity = "non_linear_s",size_a= 15)

# test.set("eta_b",np.linspace(-0.3,.0,200))
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




#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test_linear = SWE()
test_nonlinear = SWE()
test_nonlinear_b = SWE()


# Parameter
b = 2
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
         
t_end = 5
cnu = 0.3
eta_0 = np.exp(-np.square(test_linear.get('x')+1)/test_linear.get('sigma')**2)
eta_b = 0.08*(test_linear.get('x'))


# test setting
test_linear.set("BoundaryCondition",boundary)
test_linear.set("t_end",t_end)
test_linear.set("cnu",cnu)
test_linear.set("eta_0",eta_0)
test_linear.set("eta_b",eta_b)

test_nonlinear.set("BoundaryCondition",boundary)
test_nonlinear.set("t_end",t_end)
test_nonlinear.set("cnu",cnu)
test_nonlinear.set("eta_0",eta_0)

test_nonlinear_b.set("BoundaryCondition",boundary)
test_nonlinear_b.set("t_end",t_end)
test_nonlinear_b.set("cnu",cnu)
test_nonlinear_b.set("eta_0",eta_0)
test_nonlinear_b.set("eta_b",eta_b)

test_linear.numerical(linearity="linear")
test_nonlinear.numerical(linearity="non_linear_s")
test_nonlinear_b.numerical(linearity="non_linear_s")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test_linear.get("x_sta"), test_linear.get("x_end"), test_linear.get("N"))
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ of linear equation with Bathymetry')
ln2, = plt.plot([], [], 'g--', label=r'$\eta$ of nonlinear equation without Bathymetry')
ln3, = plt.plot([], [], 'b-', label=r'$\eta$ of nonlinear equation with Bathymetry')


# m = ( -0.5 - (-0.8) ) / (5 - (-5))
# b = -0.8 - m * (-5)
y4 = test_nonlinear_b.get("eta_b") - test_nonlinear_b.get("H")

wall = patches.Rectangle((5, -2), 1, 3, color='gray', zorder=5)

ax.set_xlim(test_linear.get("x_sta") - 1, test_linear.get("x_end") + 1)
ax.set_ylim(-2, 1.2)
ax.set_title(f"Animation with sigma = {test_linear.get("sigma")}, N = {test_linear.get("N")}, t = {test_linear.get("t_end")}")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$eta$")
ax.legend(loc='upper right')
ax.grid(True)

Rwall = patches.Rectangle((test_linear.get("x_end"), -2), 1, 3, color='gray', zorder=5)
Lwall = patches.Rectangle((test_linear.get("x_sta") - 1, -2), 1, 3, color='gray', zorder=5)

match b:
    case 2: # boundary = "SolidWall"        
        ax.add_patch(Lwall)
        ax.add_patch(Rwall)
    case 4: # boundary = "L_Solid_R_Open"
        ax.add_patch(Lwall)
    case 5: # boundary = "L_Open_R_Solid"
        ax.add_patch(Rwall)

ax.plot(x, y4, 'k--', linewidth=1.0)

y1 = test_linear.get("eta")
y2 = test_nonlinear.get("eta")
y3 = test_nonlinear_b.get("eta")
ln1.set_data(x, y1)
ln2.set_data(x, y2)
ln3.set_data(x, y3)

fill = ax.fill_between(x, y4, y3, color='blue', alpha=0.3)
fill2 = ax.fill_between(x, y4, -2, color='yellow', alpha=0.3)

#%% Convergence Test
from SWE import SWE
import numpy as np
test = SWE()
dxList = [0.1, 0.05, 0.01, 0.005, 0.001] #, 0.0005, 0.0001
test.ConvergenceTest()


#%% Experiment abount u and n bathymetry
from SWE import SWE
import numpy as np
test = SWE()
test_b = SWE()

N=test.get('N')
t_end = 8
b = 2
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"

test.set("BoundaryCondition",boundary)
test.set("t_end",t_end)
test.set("cnu",0.3)
test.set("eta_0",np.exp(-np.square(test.get('x')-5)/test.get('sigma')**2))

test_b.set("BoundaryCondition",boundary)
test_b.set("t_end",t_end)
test_b.set("cnu",0.3)
test_b.set("eta_0",np.exp(-np.square(test.get('x')-5)/test.get('sigma')**2))
test_b.set("eta_b",np.concatenate((+0.3*np.ones(140),np.zeros(60)),axis=0))

test.plot_ini(["eta","b"])
test.numerical(linearity = "non_linear_s")
test.plot_result()

test_b.plot_ini(["eta","b"])
test_b.numerical(linearity = "non_linear_s")
test_b.plot_result()
# test.animation(linearity = "non_linear_s")