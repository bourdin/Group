# import the class name shallow water eqution

class SWE:
    def __init__(self):
        '''
        :fuction: Initialize the parameters
        '''
        import numpy as np
        self.__N = 200
        ''' Number of cell in x '''

        ### Pysical parameter
        self.__x_end = 5
        ''' End point of domain x '''
        self.__x_sta = -self.__x_end
        ''' Start point of domain x '''
        self.__t_end = 2
        ''' End point of the time '''

        self.__sigma = 1
        ''' Standard deviation for the initial surface perturvation eta '''
        self.__H = 1
        ''' Mean depth of the water '''
        self.__g = 1
        ''' Gravitational acceleration '''

        ### Parameter about x and t
        self.__dx = (self.__x_end - self.__x_sta)/self.__N
        ''' Difference of x '''
        self.__x = np.arange(self.__x_sta, self.__x_end, self.__dx)
        ''' List of x '''
        self.__c=np.sqrt(self.__g*self.__H)
        ''' Wave speed '''
        self.__dt = self.__dx/(2*self.__c)
        ''' Step size '''
        self.__nu = 1
        ''' Stability constant '''
        self.__t_step = int(self.__t_end/self.__dt)
        ''' Number of time steps '''
        self.__t = np.linspace(0,self.__t_end,self.__t_step)
        ''' List of t axis '''

        ### Variable
        self.__eta_0 = np.exp(-np.square(self.__x)/self.__sigma**2)
        ''' Initial eta '''
        self.__eta = self.__eta_0
        ''' Surface perturvation eta '''
        self.__eta_b = np.zeros((self.__N))
        ''' Bottom bathymetry '''
        self.__u_0 = np.zeros((self.__N))
        ''' Initial u '''
        self.__u = self.__u_0
        ''' Amplitude of the wave '''
        self.__h = self.__H + self.__eta - self.__eta_b
        ''' Depth of each point '''

        ### Boundary Condition
        self.__x_sol = np.append(self.__x, self.__x_end)
        ''' Add the end point(= initial point) x '''
        self.__eta_sol = np.append(self.__eta, self.__eta[0])
        ''' Add the end point(= initial point) eta '''

        ### Get the solution
        self.__eta_s = 0.5*np.exp(-np.square(self.__x - self.__t_end * self.__c)/self.__sigma**2)+0.5*np.exp(-np.square(self.__x + self.__t_end * self.__c)/self.__sigma**2)
        ''' Exact solution of eta '''
        self.__u_s = (0.5*np.exp(-np.square(self.__x - self.__t_end * self.__c)/self.__sigma**2)-0.5*np.exp(-np.square(self.__x + self.__t_end * self.__c)/self.__sigma**2))/self.__c
        ''' Exact solution of u '''

        ## Get the error
        self.__error_u = self.__u_s - self.__u
        ''' Difference of u '''
        self.__error_eta = self.__eta_s - self.__eta
        ''' Difference of eta '''
        self.__Nerror = 0
        ''' Norm error '''

    def set_var(self):
        import numpy as np
        ### Parameter about x and t
        self.__dx = (self.__x_end - self.__x_sta)/self.__N
        ''' Difference of x '''
        self.__x = np.arange(self.__x_sta, self.__x_end, self.__dx)
        ''' List of x '''
        self.__c=np.sqrt(self.__g*self.__H)
        ''' Wave speed '''
        self.__dt = self.__dx/(2*self.__c)
        ''' Step size '''
        self.__nu = 1
        ''' Stability constant '''
        self.__t_step=int(self.__t_end/self.__dt)
        ''' Number of time steps '''
        self.__t = np.linspace(0,self.__t_end,self.__t_step)
        ''' List of t axis '''

        ### Variable
        self.__eta_0 = np.exp(-np.square(self.__x)/self.__sigma**2)
        ''' Initial eta '''
        self.__eta = self.__eta_0
        ''' Surface perturvation eta '''
        self.__eta_b = np.zeros((self.__N))
        ''' Bottom bathymetry '''
        self.__u_0 = np.zeros((self.__N))
        ''' Initial u '''
        self.__u = self.__u_0
        ''' Amplitude of the wave '''
        self.__h = self.__H + self.__eta - self.__eta_b
        ''' Depth of each point '''

        ### Boundary Condition
        self.__x_sol = np.append(self.__x, self.__x_end)
        ''' Add the end point(= initial point) x '''
        self.__eta_sol = np.append(self.__eta, self.__eta[0])
        ''' Add the end point(= initial point) eta '''

        ### Get the solution
        self.get_sol()

        ## Get the error
        self.get_error()
    
    def get_sol(self):
        import numpy as np
        self.__eta_s = 0.5*np.exp(-np.square(self.__x - self.__t_end * self.__c)/self.__sigma**2)+0.5*np.exp(-np.square(self.__x + self.__t_end * self.__c)/self.__sigma**2)
        ''' Exact solution of eta '''
        self.__u_s = (0.5*np.exp(-np.square(self.__x - self.__t_end * self.__c)/self.__sigma**2)-0.5*np.exp(-np.square(self.__x + self.__t_end * self.__c)/self.__sigma**2))/self.__c
        ''' Exact solution of u '''
    
    def get_error(self):
        self.__error_u = self.__u_s - self.__u
        ''' Difference of u '''
        self.__error_eta = self.__eta_s - self.__eta
        ''' Difference of eta '''
        self.__Nerror = 0
        ''' Norm error ''' 

    def set_const(self):
        self.__N = int((self.__x_end-self.__x_sta)/self.__dx)

    def set_t(self):
        import numpy as np
        self.__t_step=int(self.__t_end/self.__dt)
        ''' Number of time steps '''
        self.__t = np.linspace(0,self.__t_end,self.__t_step)
        ''' List of t axis '''

    def get(self,component):
        '''
        :self: SWE Class name
        :component: Variables name to check
        :fuction: Check the parameters
        '''
        match component:
            case "N": return self.__N
            case "x_end": return self.__x_end
            case "x_sta": return self.__x_sta
            case "sigma": return self.__sigma
            case "H": return self.__H
            case "g": return self.__g
            case "dx": return self.__dx
            case "dt": return self.__dt
            case "t_end": return self.__t_end
            case "t_step": return self.__t_step
            case "x": return self.__x
            case "t": return self.__t
            case "c": return self.__c
            case "eta_0": return self.__eta_0
            case "eta": return self.__eta
            case "eta_b": return self.__eta_b
            case "u": return self.__u
            case "u_0": return self.__u_0
            case "h": return self.__h
            case "x_sol": return self.__x_sol
            case "eta_s": return self.__eta_s
            case "u_s": return self.__u_s
            case "eta_sol": return self.__eta_sol
            case "error_u": return self.__error_u
            case "error_eta": return self.__error_eta
            case "Nerror": return self.__Nerror
            case _: raise InputError

    def set(self,component,number):
        '''
        :self: SWE Class name
        :component: Variables name to change
        :number: Number to change
        :fuction: Change the parameters
        '''
        # constant
        match component:
            case "N":
                self.__N = number
                self.set_var()
            case "x_end":
                self.__x_end = number
                self.set_var()
            case "x_sta":
                self.__x_sta = number
                self.set_var()
            case "sigma":
                self.__sigma = number
                self.set_var()
            case "H":
                self.__H = number
                self.set_var()
            case "g":
                self.__g = number
                self.set_var()

            # Set x and t
            case "dx":
                self.__dx = number
                self.set_const()
                self.set_var()
            case "dt": 
                self.__dt = number
                self.set_t()
                self.get_sol()
            case "t_end": 
                self.__t_end = number
                self.set_t()
                self.get_sol()
            case "t_step":
                import numpy as np
                self.__t_step = number
                self.__dt=int(self.__t_end/self.__t_step)
                self.__t = np.linspace(0,self.__t_end,self.__t_step)

            # Set variable
            case "c": 
                self.__c = number
                self.__dt = self.__dx/(2*self.__c)
                self.set_t()
                self.get_sol()
            case "eta_0": 
                self.__eta_0 = number
                self.__eta = self.__eta_0
                self.__h = self.__H + self.__eta - self.__eta_b
            case "eta_b":
                self.__eta_b = number
                self.__h = self.__H + self.__eta - self.__eta_b
            case "u_0":
                self.__u_0 = number
                self.__u = self.__u_0
                self.__h = self.__H + self.__eta - self.__eta_b
            case "eta_s": self.__eta_s = number
            case "u_s": self.__u_s = number
            case _: raise InputError

    def pflist(self):
        '''
        :self: SWE Class name
        :fuction: Print the list of class
        '''
        print("--------  X  --------")
        print("N : ", self.__N)
        print("dx : ", self.__dx)
        print("x : (", self.__x_sta,',', self.__x_end,')')
        print("--------  T  --------")
        print("t_step : ", self.__t_step)
        print("dt : ", self.__dt)
        print("t : (", 0,',', self.__t_end,')')
        print("------ Constant ------")
        print("sigma : ", self.__sigma)
        print("H : ", self.__H)
        print("g : ", self.__g)
        print("c : ", self.__c)

    def numerical(self, linearity = "linear", method = "euler", time = "all"):
        '''
        :self: SWE Class name
        :linearity: Check the linearity
        :method: Integration method
        :time: Getting it from the beginning to the end = all, just one step = one.
        :fuction: Get the numerical result
        '''
        eta=self.__eta
        u=self.__u

        import Function
        def flux_ode(t, eta):
            return Function.Flux(u, self.__h, self.__H, self.__dx, linearity) 
        def bernoulli_ode(t, u):
            return Function.Bernoulli(u, eta, self.__g, self.__dx, linearity)
        
        if (time == "all"):
            timestep = self.__t_step
        elif (time == "one"):
            timestep = 1

        # Call solve_ivp
        for i in range(0, timestep):
            if (method == "euler"):
                # Euler Method
                u = u + self.__dt * bernoulli_ode((i + 1) * self.__dt, u)
                eta = eta + self.__dt * flux_ode((i + 1) * self.__dt, eta)
            else:
                from scipy import integrate
                sol = integrate.solve_ivp(bernoulli_ode,[i*self.__dt,(i+1)*self.__dt], u, method=method, t_eval=[self.__dt*(i+1)])
                u=sol.y.flatten()
                sol = integrate.solve_ivp(flux_ode,[i*self.__dt,(i+1)*self.__dt], eta, method=method, t_eval=[self.__dt*(i+1)])
                eta=sol.y.flatten()

        self.__u = u
        self.__eta = eta
        self.__h = self.__H + self.__eta - self.__eta_b
        self.__error_u = self.__u_s - self.__u
        self.__error_eta = self.__eta_s - self.__eta

    def error(self):
        '''
        :self: SWE Class name
        :fuction: Calculate(Update) the error
        '''
        import numpy as np
        L2sq = self.__error_eta**2/self.__eta_s**2
        L2norm = np.mean(np.sqrt(L2sq))
        self.__Nerror = L2norm
        print(f"L2 norm of eta: {L2norm}")

    def plot_ini(self, size_a = 6, size_b = 4, size_c = 5):
        '''
        :self: SWE Class name
        :size_a: Horizontal length of plot
        :size_b: Vertical length of plot
        :size_c: size of the dot
        :fuction: Plot the initial condtion (IC)
        '''
        import matplotlib.pyplot as plt
        f, axes = plt.subplots(figsize = (size_a, size_b))

        # plot the exact line
        plt.plot(self.__x_sol, self.__eta_sol, color = 'r', label = 'eta')
        plt.plot(self.__x + self.__dx * 0.5, self.__u_0, color = 'b', label = 'u')

        # plot the exact points
        plt.scatter(self.__x, self.__eta_0, color = 'r', label = 'point eta', s = size_c)
        plt.scatter(self.__x + self.__dx * 0.5, self.__u_0, color = 'b', label = 'point u', s = size_c)
        plt.ylim(-0.2, 1.5)
        axes.set_title(f"Initial Condition with sigma = {self.__sigma}, N = {self.__N}, t = {self.__t_end}")
        axes.set_xlabel(f"$x$")
        axes.set_ylabel(f"$eta$")
        axes.legend()
        axes.grid()
        plt.show()

    def plot_sol(self, size_a = 6, size_b = 4):
        '''
        :self: SWE Class name
        :size_a: Horizontal length of plot
        :size_b: Vertical length of plot
        :fuction: Plot the exact solution graph
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        eta_sol = np.append(self.__eta_s, self.__eta_s[0])
        f, axes = plt.subplots(figsize = (size_a, size_b))
        plt.plot(self.__x_sol,eta_sol, color = 'r', label = 'eta')
        plt.plot(self.__x + self.__dx * 0.5, self.__u_s, color = 'b', label = 'u')
        axes.set_title(f"Reference Solution with sigma = {self.__sigma}, N = {self.__N}, t = {self.__t_end}")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$eta$")
        axes.legend()
        axes.grid()
        plt.show()

    def plot_result(self, size_a = 6, size_b = 4):
        '''
        :self: SWE Class name
        :size_a: Horizontal length of plot
        :size_b: Vertical length of plot
        :fuction: Plot the numerical result graph
        '''
        import matplotlib.pyplot as plt
        f, axes = plt.subplots(figsize = (size_a, size_b))

        # Draw reference solution
        plt.plot(self.__x, self.__eta_s, ls = '-', color = 'm', label = 'eta reference')
        plt.plot(self.__x + self.__dx * 0.5, self.__u_s, ls = '-', color = 'c', label = 'u reference')

        # Draw numerical results
        plt.scatter(self.__x, self.__eta, color = 'r', s = 5, label = 'eta')
        plt.scatter(self.__x + self.__dx * 0.5, self.__u, color = 'b', s = 5, label = 'u')
        # plt.plot(self.__x, self.__eta, ls = '-', color = 'r', label = 'eta')
        # plt.plot(self.__x + self.__dx * 0.5, self.__u, ls = '-', color = 'b', label = 'u')

        axes.set_title(f"Numerical Result with sigma = {self.__sigma}, N = {self.__N}, t = {self.__t_end}")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$eta$")
        axes.legend()
        axes.grid()
        plt.show()

    def plot_error(self, size_a = 6, size_b = 4):
        '''
        :self: SWE Class name
        :size_a: Horizontal length of plot
        :size_b: Vertical length of plot
        :fuction: Plot the error graph
        '''
        import matplotlib.pyplot as plt
        f, axes = plt.subplots(figsize = (size_a, size_b))
        plt.plot(self.__x, self.__error_eta, color = 'r', label = 'eta error')
        plt.plot(self.__x + self.__dx * 0.5, self.__error_u, color = 'b', label = 'u error')
        axes.set_title(f"Error with sigma = {self.__sigma}, N = {self.__N}, t = {self.__t_end}")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$eta$")
        axes.legend()
        axes.grid()
        plt.ylim(-1,1)
        plt.show()

    def animation(self, size_a = 10, size_b = 6):
        '''
        :self: SWE Class name
        :size_a: Horizontal length of plot
        :size_b: Vertical length of plot
        :fuction: Plot the animation result
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib import rc

        data = [[0 for col in range(self.__N + 1)] for row in range(self.__t_step)]

        for i in range(0, self.__t_step):
            self.numerical(time = "one")
            data[i] = np.append(self.__eta, self.__eta[0])

        # Make the Animation
        fig, ax = plt.subplots(figsize=(size_a,size_b))

        x,y = [],[]
        x = np.linspace(self.__x_sta, self.__x_end, self.__N)
        ln, = plt.plot([], [], )

        def init():    
            ax.set_xlim(self.__x_sta - 1, self.__x_end + 1)
            ax.set_ylim(-0.2, 1.2)
            ax.grid(True)
            
            return ln,

        def update(i):    
            y = data[i]
            
            ln.set_data(self.__x_sol, y)
            
            return ln,

        ani = FuncAnimation(fig=fig, func=update, frames=np.array(list(range(self.__t_step))),
                            init_func=init, interval=20, blit=True)

        rc('animation', html='html5')
        ani
        ani.save('fig.gif', writer='imagemagick', fps=15, dpi=100)

    def ConvergenceTest(self, dxList = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
        # Experiment for the convergence test
        import numpy as np
        import matplotlib.pyplot as plt

        dx_data = dxList
        N_data = [int((self.get("x_end") - self.get("x_sta")) / x) for x in dx_data]

        # change dt, t_step also np.arange
        data = [0 for row in range(np.size(N_data))]

        for j in range(0,np.size(N_data)):
            self.set("N", N_data[j])
            self.set("dx", dx_data[j])
            self.numerical()
            self.error()

            data[j] = self.get("Nerror")

        # Plot the convergence test result
        f, axes = plt.subplots(figsize = (6,4))
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(dx_data,data)
        plt.scatter(dx_data,data,label='L2 Norm error',s=10)
        axes.set_title(f"Convergence test result")
        axes.set_xlabel(r"$dx$")
        axes.set_ylabel(r"$Error$")
        axes.legend()
        axes.grid()
        plt.show()




class InputError(Exception):
     def __init__(self):
        super().__init__('Input variable name is invalid.')