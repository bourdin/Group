#%%
## Library
from SWE import SWE
import numpy as np
test = SWE()
b = 5
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
test.set("BoundaryCondition",boundary)
test.set("t_end",8)
test.set("cnu",0.3)
# test.set("eta_0",np.exp(-np.square(test.get('x')+1)/test.get('sigma')**2))
# test.set("eta_b",0.08*(test.get('x')-5))
test.numerical(linearity = "non_linear_s")
test.plot_result()
# test.animation(linearity = "non_linear_s")
# test.plot_ini(["eta","b"])
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

test1 = SWE()
test2 = SWE()

test1.set("BoundaryCondition","L_Open_R_Solid")
test1.set("t_end",20)
test1.set("cnu",0.3)
test1.set("eta_0",np.exp(-np.square(test1.get('x')+1)/test1.get('sigma')**2))

test2.set("BoundaryCondition","L_Open_R_Solid")
test2.set("t_end",20)
test2.set("cnu",0.3)
test2.set("eta_0",np.exp(-np.square(test1.get('x')+1)/test1.get('sigma')**2))
test2.set("eta_b",0.08*(test1.get('x')-5))

data = np.zeros((test1.get("N"),test1.get("t_step"),2))

for i in range(0, test1.get("t_step")):
    test1.numerical(linearity="non_linear_s", time = "one")
    test2.numerical(linearity="non_linear_s", time = "one")
    data[:,i,0] = test1.get("eta")
    data[:,i,1] = test2.get("eta")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test1.get("x_sta"), test1.get("x_end"), test1.get("N"))
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ without Bathymetry')
ln2, = plt.plot([], [], 'b-', label=r'$\eta$ with Bathymetry')

m = ( -0.5 - (-0.8) ) / (5 - (-5))
b = -0.8 - m * (-5)
y3 = m * x + b

wall = patches.Rectangle((5, -1), 1, 2, color='gray', zorder=5)

def init():
    ax.set_xlim(test1.get("x_sta") - 1, test1.get("x_end") + 1)
    ax.set_ylim(-1, 1.2)
    ax.set_title(f"Animation with sigma = {test1.get("sigma")}, N = {test1.get("N")}, t = {test1.get("t_end")}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$eta$")
    ax.legend(loc='upper right')
    ax.grid(True)

    ax.add_patch(wall)  # 벽 그리기 (한 번만)
    ax.plot(x, y3, 'k--', linewidth=1.0)
    
    return ln1, ln2

def update(i):    
    y1 = data[:, i, 0]
    y2 = data[:, i, 1]
    ln1.set_data(x, y1)
    ln2.set_data(x, y2)

    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    # y2와 -0.5 사이 영역 채우기
    fill = ax.fill_between(x, y3, y2, color='blue', alpha=0.3)
    fill2 = ax.fill_between(x, y3, -1, color='yellow', alpha=0.3)

    return ln1, ln2, fill, fill2

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test1.get("t_step"), 2))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)
rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)

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

#%%
# Figure for linear vs nonlinear
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test_linear = SWE()
test_nonlinear = SWE()

# Parameter
b = 5
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
         
t_end = 8
cnu = 0.3
eta_0 = np.exp(-np.square(test_linear.get('x')+1)/test_linear.get('sigma')**2)
# eta_b = 0.08*(test_linear.get('x'))


# test setting
test_linear.set("BoundaryCondition",boundary)
test_linear.set("t_end",t_end)
test_linear.set("cnu",cnu)
test_linear.set("eta_0",eta_0)
# test_linear.set("eta_b",eta_b)

test_nonlinear.set("BoundaryCondition",boundary)
test_nonlinear.set("t_end",t_end)
test_nonlinear.set("cnu",cnu)
test_nonlinear.set("eta_0",eta_0)

test_linear.numerical(linearity="linear")
test_nonlinear.numerical(linearity="non_linear_s")

# Make the Animation
fig, ax = plt.subplots(figsize=(8,2))

x,y = [],[]
x = np.linspace(test_linear.get("x_sta"), test_linear.get("x_end"), test_linear.get("N"))
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ of linear equation with Bathymetry')
ln2, = plt.plot([], [], 'b', label=r'$\eta$ of nonlinear equation without Bathymetry')

ax.set_xlim(test_linear.get("x_sta") - 1, test_linear.get("x_end") + 1)
ax.set_ylim(-0.1, 1)
# ax.set_title(f"Linear and Nonlinear Results when t = {test_linear.get("t_end")}")
ax.set_xlabel(f"$x$ (t = {test_linear.get("t_end")})")
ax.set_ylabel(r"$\eta$")
# ax.legend(loc='upper right')
ax.grid(True)

y1 = test_linear.get("eta")
y2 = test_nonlinear.get("eta")
ln1.set_data(x, y1)
ln2.set_data(x, y2)

#%%
# Nonlinear vs nonlinear bathymetry

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test_nonlinear = SWE()
test_nonlinear_b = SWE()


# Parameter
b = 5
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
         
t_end = 10
cnu = 0.3
eta_0 = np.exp(-np.square(test_nonlinear_b.get('x')+1)/test_nonlinear_b.get('sigma')**2)
eta_b = 0.08*(test_nonlinear_b.get('x'))


# test setting

test_nonlinear.set("BoundaryCondition",boundary)
test_nonlinear.set("t_end",t_end)
test_nonlinear.set("cnu",cnu)
test_nonlinear.set("eta_0",eta_0)

test_nonlinear_b.set("BoundaryCondition",boundary)
test_nonlinear_b.set("t_end",t_end)
test_nonlinear_b.set("cnu",cnu)
test_nonlinear_b.set("eta_0",eta_0)
test_nonlinear_b.set("eta_b",eta_b)

test_nonlinear.numerical(linearity="non_linear_s")
test_nonlinear_b.numerical(linearity="non_linear_s")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test_nonlinear_b.get("x_sta"), test_nonlinear_b.get("x_end"), test_nonlinear_b.get("N"))
ln1, = plt.plot([], [], 'g--', label=r'$\eta$ of nonlinear equation without Bathymetry')
ln2, = plt.plot([], [], 'b-', label=r'$\eta$ of nonlinear equation with Bathymetry')

# m = ( -0.5 - (-0.8) ) / (5 - (-5))
# b = -0.8 - m * (-5)
y3 = test_nonlinear_b.get("eta_b") - test_nonlinear_b.get("H")

wall = patches.Rectangle((5, -2), 1, 3, color='gray', zorder=5)

ax.set_xlim(test_nonlinear_b.get("x_sta"), test_nonlinear_b.get("x_end") + 1)
ax.set_ylim(-1.5, 1.3)
ax.set_title(f"Nonlinear result with and without bathymetry when t = {test_nonlinear_b.get("t_end")}")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$eta$")
ax.legend(loc='upper right')
ax.grid(True)

Rwall = patches.Rectangle((test_nonlinear_b.get("x_end"), -2), 1, 3, color='gray', zorder=5)
Lwall = patches.Rectangle((test_nonlinear_b.get("x_sta") - 1, -2), 1, 3, color='gray', zorder=5)

match b:
    case 2: # boundary = "SolidWall"        
        ax.add_patch(Lwall)
        ax.add_patch(Rwall)
    case 4: # boundary = "L_Solid_R_Open"
        ax.add_patch(Lwall)
    case 5: # boundary = "L_Open_R_Solid"
        ax.add_patch(Rwall)

ax.plot(x, y3, 'k--', linewidth=1.0)

y1 = test_nonlinear.get("eta")
y2 = test_nonlinear_b.get("eta")
ln1.set_data(x, y1)
ln2.set_data(x, y2)

fill = ax.fill_between(x, y3, y2, color='blue', alpha=0.3)
fill2 = ax.fill_between(x, y3, -2, color='yellow', alpha=0.3)

#%% Convergence Test
from SWE import SWE
import numpy as np
test = SWE()
dxList = [0.1, 0.05, 0.01, 0.005, 0.001] #, 0.0005, 0.0001
test.ConvergenceTest()
