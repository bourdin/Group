#%% Experiment abount u and n bathymetry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test1 = SWE()
test2 = SWE()
test3 = SWE()

N=test1.get('N')
t_end = 30
b = 2
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"

test1.set("BoundaryCondition",boundary)
test1.set("t_end",t_end)
test1.set("cnu",0.3)
test1.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))

test2.set("BoundaryCondition",boundary)
test2.set("t_end",t_end)
test2.set("cnu",0.3)
test2.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))
test2.set("eta_b",np.concatenate((+0.3*np.ones(140),np.zeros(60)),axis=0))

test3.set("BoundaryCondition",boundary)
test3.set("t_end",t_end)
test3.set("cnu",0.3)
test3.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))
test3.set("eta_b",np.concatenate((-0.3*np.ones(140),np.zeros(60)),axis=0))

data = np.zeros((test1.get("N"),test1.get("t_step"),3))

for i in range(0, test1.get("t_step")):
    test1.numerical(linearity="non_linear_s", time = "one")
    test2.numerical(linearity="non_linear_s", time = "one")
    test3.numerical(linearity="non_linear_s", time = "one")
    data[:,i,0] = test1.get("eta")
    data[:,i,1] = test2.get("eta")
    data[:,i,2] = test3.get("eta")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test1.get("x_sta"), test1.get("x_end"), test1.get("N"))
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ without Bathymetry')
ln2, = plt.plot([], [], 'b--', label=r'$\eta$ with orange Bathymetry')
ln3, = plt.plot([], [], 'b-', label=r'$\eta$ with yellow Bathymetry')

bottom0 = test1.get("eta_b")-test1.get("H")
bottom1 = test2.get("eta_b")-test2.get("H")
bottom2 = test3.get("eta_b")-test3.get("H")

Lwall = patches.Rectangle((test1.get("x_sta")-1, -1.5), 1, 3, color='gray', zorder=5)
Rwall = patches.Rectangle((test1.get("x_end"), -1.5), 1, 3, color='gray', zorder=5)

def init():
    ax.set_xlim(test1.get("x_sta") - 1, test1.get("x_end") + 1)
    ax.set_ylim(-1.5, 1.2)
    ax.set_title(f"Animation with sigma = {test1.get("sigma")}, N = {test1.get("N")}, t = {test1.get("t_end")}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$eta$")
    ax.legend(loc='upper right')
    ax.grid(True)

    ax.add_patch(Lwall)
    ax.add_patch(Rwall)
    ax.plot(x, bottom0, 'k--', linewidth=1.0)
    
    return ln1, ln2

def update(i):
    y1 = data[:, i, 0]
    y2 = data[:, i, 1]
    y3 = data[:, i, 2]
    ln1.set_data(x, y1)
    ln2.set_data(x, y2)
    ln3.set_data(x, y3)

    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    # y2와 -0.5 사이 영역 채우기
    fill = ax.fill_between(x, bottom2, y3, color='blue', alpha=0.3)
    fill1 = ax.fill_between(x, bottom1, -1.5, color='orange', alpha=0.3)
    fill2 = ax.fill_between(x, bottom2, -1.5, color='yellow', alpha=0.3)

    return ln1, ln2, fill, fill2

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test1.get("t_step"), 2))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)
rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)

#%% Experiment abount u and n bathymetry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches    
from matplotlib import rc
from SWE import SWE
import numpy as np

test1 = SWE()
test2 = SWE()
test3 = SWE()

N=test1.get('N')
t_end = 30
b = 2
match b:
    case 1: boundary = "Periodic"
    case 2: boundary = "SolidWall"
    case 3: boundary = "Open"
    case 4: boundary = "L_Solid_R_Open"
    case 5: boundary = "L_Open_R_Solid"

test1.set("BoundaryCondition",boundary)
test1.set("t_end",t_end)
test1.set("cnu",0.3)
test1.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))

test2.set("BoundaryCondition",boundary)
test2.set("t_end",t_end)
test2.set("cnu",0.3)
test2.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))
test2.set("eta_b",np.concatenate((np.zeros(60),+0.3*np.ones(80),np.zeros(60)),axis=0))

test3.set("BoundaryCondition",boundary)
test3.set("t_end",t_end)
test3.set("cnu",0.3)
test3.set("eta_0",np.exp(-np.square(test1.get('x')-test1.get('x_end'))/test1.get('sigma')**2))
test3.set("eta_b",np.concatenate((np.zeros(60),-0.3*np.ones(80),np.zeros(60)),axis=0))

data = np.zeros((test1.get("N"),test1.get("t_step"),3))

for i in range(0, test1.get("t_step")):
    test1.numerical(linearity="non_linear_s", time = "one")
    test2.numerical(linearity="non_linear_s", time = "one")
    test3.numerical(linearity="non_linear_s", time = "one")
    data[:,i,0] = test1.get("eta")
    data[:,i,1] = test2.get("eta")
    data[:,i,2] = test3.get("eta")

# Make the Animation
fig, ax = plt.subplots(figsize=(10,6))

x,y = [],[]
x = np.linspace(test1.get("x_sta"), test1.get("x_end"), test1.get("N"))
ln1, = plt.plot([], [], 'r--', label=r'$\eta$ without Bathymetry')
ln2, = plt.plot([], [], 'b--', label=r'$\eta$ with orange Bathymetry')
ln3, = plt.plot([], [], 'b-', label=r'$\eta$ with yellow Bathymetry')

bottom0 = test1.get("eta_b")-test1.get("H")
bottom1 = test2.get("eta_b")-test2.get("H")
bottom2 = test3.get("eta_b")-test3.get("H")

Lwall = patches.Rectangle((test1.get("x_sta")-1, -1.5), 1, 3, color='gray', zorder=5)
Rwall = patches.Rectangle((test1.get("x_end"), -1.5), 1, 3, color='gray', zorder=5)

def init():
    ax.set_xlim(test1.get("x_sta") - 1, test1.get("x_end") + 1)
    ax.set_ylim(-1.5, 1.2)
    ax.set_title(f"Animation with sigma = {test1.get("sigma")}, N = {test1.get("N")}, t = {test1.get("t_end")}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$eta$")
    ax.legend(loc='upper right')
    ax.grid(True)

    ax.add_patch(Lwall)
    ax.add_patch(Rwall)
    ax.plot(x, bottom0, 'k--', linewidth=1.0)
    
    return ln1, ln2

def update(i):
    y1 = data[:, i, 0]
    y2 = data[:, i, 1]
    y3 = data[:, i, 2]
    ln1.set_data(x, y1)
    ln2.set_data(x, y2)
    ln3.set_data(x, y3)

    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    # y2와 -0.5 사이 영역 채우기
    fill = ax.fill_between(x, bottom2, y3, color='blue', alpha=0.3)
    fill1 = ax.fill_between(x, bottom1, -1.5, color='orange', alpha=0.3)
    fill2 = ax.fill_between(x, bottom2, -1.5, color='yellow', alpha=0.3)

    return ln1, ln2, fill, fill2

ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(0, test1.get("t_step"), 3))), #self.__t_step - 400
                    init_func=init, interval=20, blit=True)
rc('animation', html='html5')
ani
ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)