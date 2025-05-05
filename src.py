import numpy as np
from numba_kdtree import KDTree
from scipy.spatial import distance_matrix
from numba import njit
import matplotlib.pyplot as plt


# constants

R = 1 # swimmer radius
v0 = 1 # swimmer speed
eta = 1 # viscosity
R1 = 5 * R # inner radius
R2 = 50 * R # outer radius
L = 200 * R # length
#Omega = 0.1 # inner rotation speed 
#Omega = 5 # inner rotation speed 

beta = 200

mu = 1 / (6 * np.pi * eta * R)
print(mu)

#alpha = 0.0331355721
alpha = 0.001656

D_T  = alpha / (6 * np.pi)
D_R = alpha / (8 * np.pi)

k = np.array([1, 0, 0]) # up vector

sigma0 = 0.1 # variance 

print(f'1/D_R*beta: {1 / D_R / beta}')

sigma_wall = 2 ** (- 1 / 6) * R
sigma_part = 2 ** (- 1 / 6) * 2 * R

Omega_crit = ((R2 / R1) ** 2 - 1) / 2 / beta

def get_params(Omega):
    A = - Omega / (R2 ** 2 / R1 ** 2 - 1)
    B = Omega * R2 ** 2 / (R2 ** 2 / R1 ** 2 - 1)

    Omega_crit = ((R2 / R1) ** 2 - 1) / 2 / beta

    #print(Omega_crit)

    if Omega > Omega_crit:
        T_tumb = 4 * np.pi * beta / np.sqrt(4 * A **2 * beta ** 2 - 1)
        print(f'Tumbling period: {T_tumb}')
    else:
        print('No tumbling')
        
    omega = np.array([0, 0, 2 * A]) # flow curl
    print(f'omega/D_R: {- 2 * A / D_R}')
    
    return A, B, omega

def get_v_flow(A, B, poses):
    R_norm = np.linalg.norm(poses, axis = 1)
    v = A * R_norm + B / R_norm
    vs = np.array([- v * poses[:,1] / R_norm, v * poses[:,0] / R_norm]).T
    return np.hstack((vs, np.zeros((1, v.shape[0])).T))

@njit
def force(r, epsilon, sigma, delta=1e-6):
    return 24 * epsilon * (2 * sigma ** 12 / (r + delta) ** 13 - sigma ** 6 / (r + delta) ** 7)

def get_wall_force(poses):
    R_norm = np.linalg.norm(poses, axis = 1)
    epsilon = 0.01
    f = np.empty_like(R_norm)
    for i in range(f.shape[0]):
        if (R_norm[i] < R1 + R):
            if (R_norm[i] < R1 + 0.59):
                f[i] = 1e2
            else:
                f[i] = force(R_norm[i] - R1, epsilon, sigma_wall)          
        elif (R_norm[i] > R2 - R):
            if (R_norm[i] > R2 - 0.59):
                f[i] = - 1e2
            else:
                f[i] = - force(R2 - R_norm[i], epsilon, sigma_wall)
        else:
            f[i] = 0
    fs = np.array([f * poses[:,0] / R_norm, f * poses[:,1] / R_norm]).T
    return np.hstack((fs, np.zeros((1, f.shape[0])).T))

@njit
def get_interaction(Rs, epsilon):
    Rs_search = Rs.copy()
    Rs_plus = Rs[Rs[:,2]<=3]
    Rs_minus = Rs[Rs[:,2]>=197]
    if Rs_plus.size != 0:
        Rs_plus[:,2] += 200
        Rs_search  = np.concatenate((Rs_search, Rs_plus), axis = 0)
    if Rs_minus.size != 0:
        Rs_minus[:,2] -= 200
        Rs_search = np.concatenate((Rs_minus, Rs_search), axis = 0)
    search = KDTree(Rs_search)
    dist, inds, nn = search.query(Rs, k = 13, distance_upper_bound = 3 * sigma_part)
    F = np.zeros_like(Rs)
    if epsilon != 0:
        for i in range(Rs.shape[0]):
            for j in range(13):
                if inds[i, j] != -1 and dist[i, j] > 0:
                    f = force(dist[i,j], epsilon, sigma_part)
                    if f > 200:
                        f = 200
                    F[i] += f * (Rs[i] - Rs_search[inds[i, j]]) / dist[i,j]
    return F

def dot(ps, Rs, t, beta, A, B, omega, epsilon, k, v0):
    ps = (ps.T / np.linalg.norm(ps, axis = 1)).T
    p_dot = (k - (ps.T * (k @ ps.T)).T ) / (2 * beta) + 1 / 2 * np.cross(omega, ps)
    R_dot = v0 * ps + get_v_flow(A, B, Rs[:, :2]) + mu * (get_wall_force(Rs[:, :2]) + get_interaction(Rs, epsilon))
    #print(f'in dot: {R_dot}')
    return p_dot, R_dot

def euler(ps, Rs, params, epsilon, k, dot, dt, t = 0, noise = True):    
    dpdt, dRdt = dot(ps, Rs, t, beta, *params, epsilon, k, v0)
    dps_d = dpdt * dt
    dRs_d = dRdt * dt
    if not noise:
        #print(f'in Euler: {dRs_d}')
        return ((ps + dps_d).T/np.linalg.norm(ps + dps_d, axis = 1)).T, Rs + dRs_d
    nT = np.random.normal(0, sigma0, Rs.shape)
    nR = np.random.normal(0, sigma0, ps.shape)
    dRs_s = np.sqrt(2 * D_T * dt) * nT
    dps_s = - 2 * D_R * dt * ps + np.sqrt(2 * D_R * dt) * np.cross(nR, ps)
    return ((ps + dps_d + dps_s).T/np.linalg.norm(ps + dps_d + dps_s, axis = 1)).T, Rs + dRs_d + dRs_s

def solver(Omega, epsilon, k, new, N, p0s, R0s, dt, steps, noise = False):
    params = get_params(Omega)
    ps = np.zeros((steps, N, 3))
    Rs = np.zeros((steps, N, 3))
    ps[0] = p0s
    Rs[0] = R0s
    time = np.zeros(steps)
    step = 1
    while step < steps:
        ps_n, Rs_n = new(ps[step-1], Rs[step-1], params, epsilon, k, dot, dt, time[step], noise)
        Rs_n[:, 2] = Rs_n[:, 2] % L
        ps[step] = ps_n.copy()
        Rs[step] = Rs_n.copy()
        time[step] = step * dt
        step += 1
    return ps, Rs, time

## Plotting

def data_for_cylinder_along_z(radius, height_z, null_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z + null_z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

def makesphere(x, y, z, radius, resolution=10):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def plot_state(coords):
    ax = plt.figure().add_subplot(projection='3d')

    for i in range(len(coords)):
        X, Y, Z = makesphere(coords[i][0], coords[i][1], coords[i][2], R)
        ax.plot_surface(X, Y, Z, color="r")

    X1,Y1,Z1 = data_for_cylinder_along_z(R1, L, 0)
    ax.plot_surface(X1, Y1, Z1, alpha=0.5)

    X2,Y2,Z2 = data_for_cylinder_along_z(R2, L, 0)
    ax.plot_surface(X2, Y2, Z2, alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal');

    plt.show()

def plot_dist(sol, N, threshold, bins):
    dist = distance_matrix(sol[-1, :, :],sol[-1,:,:])
    ds = dist[np.triu_indices(N, k = 1)]
    ds_short = [d for d in ds if d < threshold]

    fig, axs = plt.subplots(1, 2, figsize = (12,5))
    axs[0].hist(ds, bins = bins,  density=False);
    axs[1].hist(ds_short, bins = bins,  density=False);

    axs[0].set_xlabel('distance between particles')
    axs[1].set_xlabel('distance between particles')

    plt.show()

def plot_angle(A, sol, bins):
    psi = np.arctan2(sol[-1, :, 1], sol[-1, :, 0]) * 180 / np.pi
    if -1 < (2 * A * beta) and (2 * A * beta) < 1:
        phi_star = np.arcsin(2 * A * beta) * 180 / np.pi
        plt.axvline(phi_star, color = 'r')

    plt.hist(psi,  bins = bins, density=False);
    plt.xlabel(f'psi')
    plt.ylabel(f'# particles')
    plt.show()

def plot_2d(coords):
    fig, axs = plt.subplots(figsize = (10,10))
    inner = plt.Circle((0, 0), R1, color='b', fill=False)
    outer = plt.Circle((0, 0), R2, color='b', fill=False)
    axs.add_patch(inner)
    axs.add_patch(outer)

    ind=np.argsort(coords[:,-1])
    b=coords[ind]

    for i in range(N):
        plt.scatter(b[i][0], b[i][1], s = 300 * (1 - b[i][2]/L), c = b[i][2], cmap = 'jet', vmin = 0, vmax = L)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.grid(True)
    axs.axis('equal')
    plt.colorbar()
    plt.show()    

def plot_R_mean(sol):
    Rads = np.mean(np.linalg.norm(sol[:, :, :2], axis = 2), axis = 1)

    fig, axs = plt.subplots()
    axs.plot(t, Rads)
    axs.set_xlabel('Time')
    axs.set_ylabel('Mean radial coordinate')
    axs.grid(True)

    plt.show()

def plot_trajectory(sol_i):
    fig, axs = plt.subplots()
    axs.plot(sol_i[:,0], sol_i[:,1])
    axs.plot(sol_i[0,0], sol_i[0,1], 'o')

    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.grid(True)
   
    axs.axis('equal')
    plt.show() 