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
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ of linear equation with Bathymetry', linewidth = 3)
ln2, = plt.plot([], [], 'b', label=r'$\eta$ of nonlinear equation without Bathymetry', linewidth = 3)

ax.set_xlim(test_linear.get("x_sta") - 1, test_linear.get("x_end") + 1)
ax.set_ylim(-0.1, 1)
# ax.set_title(f"Linear and Nonlinear Results when t = {test_linear.get("t_end")}")
ax.set_xlabel(f"$x$ (t = {test_linear.get("t_end")})", fontsize = 20)
ax.set_ylabel(r"$\eta$", fontsize = 20)
# ax.legend(loc='upper right')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax.grid(True)

y1 = test_linear.get("eta")
y2 = test_nonlinear.get("eta")
ln1.set_data(x, y1)
ln2.set_data(x, y2)

filename = f"l_vs_n_{boundary}_{t_end}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#%%
# Animation for linear vs nonlinear animation
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
b = 1
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
         
t_end = 10
cnu = 0.3
eta_0 = np.exp(-np.square(test_linear.get('x')+1)/test_linear.get('sigma')**2)

# test setting
test_linear.set("BoundaryCondition",boundary)
test_linear.set("t_end",t_end)
test_linear.set("cnu",cnu)
test_linear.set("eta_0",eta_0)

test_nonlinear.set("BoundaryCondition",boundary)
test_nonlinear.set("t_end",t_end)
test_nonlinear.set("cnu",cnu)
test_nonlinear.set("eta_0",eta_0)

data = np.zeros((test_linear.get("N"),test_linear.get("t_step"),2))

for i in range(0, test_linear.get("t_step")):
    test_linear.numerical(linearity="linear", time = "one")
    test_nonlinear.numerical(linearity="non_linear", time = "one")
    data[:,i,0] = test_linear.get("eta")
    data[:,i,1] = test_nonlinear.get("eta")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test_linear.get("x_sta"), test_linear.get("x_end"), test_linear.get("N"))
ln1, = plt.plot([], [], 'b-', label=r'$\eta$ for linear equations')
ln2, = plt.plot([], [], 'b-', label=r'$\eta$ for nonlinear equations')

bottom0 = test_linear.get("eta_b")-test_linear.get("H")
bottom1 = test_nonlinear.get("eta_b")-test_nonlinear.get("H")

Rwall = patches.Rectangle((test_nonlinear.get("x_end"), -2), 1, 3, color='gray', zorder=5)
Lwall = patches.Rectangle((test_nonlinear.get("x_sta") - 1, -2), 1, 3, color='gray', zorder=5)

match b:
    case 2: # boundary = "SolidWall"        
        ax.add_patch(Lwall)
        ax.add_patch(Rwall)
    case 4: # boundary = "L_Solid_R_Open"
        ax.add_patch(Lwall)
    case 5: # boundary = "L_Open_R_Solid"
        ax.add_patch(Rwall)

def init():
    ax.set_xlim(test_linear.get("x_sta") - 1, test_linear.get("x_end") + 1)
    ax.set_ylim(-1.5, 1.2)
    # ax.set_title(f"Animation with sigma = {test_linear.get("sigma")}, N = {test_linear.get("N")}, t = {test_linear.get("t_end")}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\eta$")
    ax.legend(loc='upper right')
    ax.grid(True)

    # ax.add_patch(Lwall)
    ax.add_patch(Rwall)
    ax.plot(x, bottom0, 'k--', linewidth=1.0)
    
    return ln1, ln2

def update(i):
    y1 = data[:, i, 0]
    y2 = data[:, i, 1]
    ln1.set_data(x, y1)
    ln2.set_data(x, y2)

    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    # y2와 -0.5 사이 영역 채우기
    fill = ax.fill_between(x, bottom0, y2, color='blue', alpha=0.3)
    fill1 = ax.fill_between(x, bottom1, -1.5, color='orange', alpha=0.3)

    return ln1, ln2, fill, fill1

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test_linear.get("t_step")))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)
rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)


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

font = 26

# Parameter
b = 1
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
ln1, = plt.plot([], [], 'b--', label=r'$\eta$ of nonlinear equation without Bathymetry', linewidth = 3)
ln2, = plt.plot([], [], 'b-', label=r'$\eta$ of nonlinear equation with Bathymetry', linewidth = 3)

# m = ( -0.5 - (-0.8) ) / (5 - (-5))
# b = -0.8 - m * (-5)
y3 = test_nonlinear_b.get("eta_b") - test_nonlinear_b.get("H")

wall = patches.Rectangle((5, -2), 1, 3, color='gray', zorder=5)

ax.set_xlim(test_nonlinear_b.get("x_sta"), test_nonlinear_b.get("x_end") + 1)
ax.set_ylim(-1.5, 1.3)
#ax.set_title(f"Nonlinear result with and without bathymetry when t = {test_nonlinear_b.get("t_end")}")
ax.set_xlabel(r"$x$", fontsize = font)
ax.set_ylabel(r"$\eta$", fontsize = font)
plt.xticks(fontsize = font)
plt.yticks(fontsize = font)
#ax.legend(loc='upper right')
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

filename = f"n_vs_nb_mix_{t_end}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")

#%%
# Nonlinear vs nonlinear bathymetry animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test1 = SWE()
test2 = SWE()
t = 16

test1.set("BoundaryCondition","L_Open_R_Solid")
test1.set("t_end",t)
test1.set("cnu",0.3)
test1.set("eta_0",np.exp(-np.square(test1.get('x')+1)/test1.get('sigma')**2))

test2.set("BoundaryCondition","L_Open_R_Solid")
test2.set("t_end",t)
test2.set("cnu",0.3)
test2.set("eta_0",np.exp(-np.square(test1.get('x')+1)/test1.get('sigma')**2))
test2.set("eta_b",0.08*test1.get('x'))

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
ln1, = plt.plot([], [], 'b--', label=r'$\eta$ with constant bathymetry')
ln2, = plt.plot([], [], 'b-', label=r'$\eta$ with linear bathymetry')

m = 0.08 #( -1 - (-0.8) ) / (5 - (-5))
b = -1 #-0.8 - m * (-5)
y3 = m * x + b

wall = patches.Rectangle((5, -1.5), 1, 2.5, color='gray', zorder=5)

def init():
    ax.set_xlim(test1.get("x_sta") - 1, test1.get("x_end") + 1)
    ax.set_ylim(-1.5, 1.2)
    # ax.set_title(f"Animation with sigma = {test1.get("sigma")}, N = {test1.get("N")}, t = {test1.get("t_end")}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\eta$")
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
    fill2 = ax.fill_between(x, y3, -1.5, color='orange', alpha=0.3)

    return ln1, ln2, fill, fill2

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test1.get("t_step"), 2))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)
rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)

#%%
# Animation for linear
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test_linear = SWE()

# Parameter
b = 1
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"
         
t_end = 3
cnu = 0.3
eta_0 = np.exp(-np.square(test_linear.get('x')+1)/test_linear.get('sigma')**2)

# test setting
test_linear.set("BoundaryCondition",boundary)
test_linear.set("t_end",t_end)
test_linear.set("eta_0",eta_0)

data = np.zeros((test_linear.get("N"), test_linear.get("t_step")))

for i in range(0, test_linear.get("t_step")):
    test_linear.numerical(linearity="non_linear", time = "one")
    data[:,i] = test_linear.get("eta")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test_linear.get("x_sta"), test_linear.get("x_end"), test_linear.get("N"))
ln, = plt.plot([], [], 'b-', label=r'$\eta$ of nonlinear equation without Bathymetry')

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

def init():
    ax.set_xlim(test_linear.get("x_sta") - 1, test_linear.get("x_end") + 1)
    ax.set_ylim(-1.5, 1.2)
    #ax.set_title(f"Animation with sigma = {self.__sigma}, N = {self.__N}, t = {self.__t_end}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\eta$")
    # ax.legend()
    ax.grid(True)
    
    return ln,

def update(i):    
    y = data[:,i]
    ln.set_data(x, y)

    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    bottom0 = test_linear.get("eta_b")-test_linear.get("H")

    # y2와 -0.5 사이 영역 채우기
    fill = ax.fill_between(x, bottom0, y, color='blue', alpha=0.3)
    fill1 = ax.fill_between(x, bottom0, -1.5, color='orange', alpha=0.3)
    
    return ln,fill,fill1

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test_linear.get("t_step"), 2))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)

rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)