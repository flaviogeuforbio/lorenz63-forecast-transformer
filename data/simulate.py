import numpy  as np
import matplotlib.pyplot as plt

def lorenz_field(state, rho, sigma, beta):
    #unpacking the state
    x, y, z = state

    #calculating time derivatives 
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz], dtype = np.float64)

#Runge-Kutta4 integration method
def rk4(f, y, dt, *args):
    k1 = f(y, *args)
    k2 = f(y + (dt/2.0) * k1, *args)
    k3 = f(y + (dt/2.0) * k2, *args)
    k4 = f(y + dt * k3, *args)

    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_traj(
        rho: float, 
        sigma: float = 10.0, 
        beta: float = 8.0 / 3.0, 
        dt: float = 0.01, 
        n_steps: int = 25000, 
        burn_in: int = 400, 
        x0 = None
):
    #imposing default initial conditions (if not specified)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype = np.float64)

    #creating trajectories and position var
    traj = np.empty((n_steps, 3), dtype = np.float64)
    y = x0.copy()

    #running simulation
    for n in range(n_steps):
        y = rk4(lorenz_field, y, dt, rho, sigma, beta) #updating position according to RK4
        traj[n] = y #updating trajectories array

    #output full trajectories except for the initial transient 
    return traj[burn_in:]

#function to plot 3D trajectory + x(t) curve
def plot_traj(traj, title, x_plot=False):
    #extracting x(t), y(t), z(t)
    x = traj[:,0]; y = traj[:,1]; z = traj[:,2]

    if x_plot: #1D plot x(t) is optional
        t = np.arange(len(traj))

        plt.figure(figsize=(8,3))
        plt.plot(t, x)
        plt.title(title + " — x(t)")
        plt.xlabel("time step")
        plt.ylabel("x")
        plt.tight_layout()

    # 3D plot
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=0.5)
    ax.set_title(title + " — phase space")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout() 
