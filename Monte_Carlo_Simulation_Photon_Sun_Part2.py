import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy import constants as const
from tqdm import tqdm

# This code was made in conjunction with Pycharm Co-Pilot, particularly for the visualization which uses AI (Mellum) to help generate code snippets.

def simulation_setup(plot_sim=False, save_sim=False):
    """
        Description:
        ------------
        Function Setting up the Monte Carlo Simulation of Photon Scattering in the Sun (3D)
    """
    R_sun = 6.9634e10       # cm
    sigma_T = const.sigma_T.cgs.value  # Thomson cross-section [cm^2]
    c = const.c.cgs.value   # speed of light [cm/s]
    n_photons = 5  # Number of photons to simulate
    if (plot_sim is True) and (save_sim is True):
        paths = simulate_photons(R_sun, sigma_T, c, n_photons)
        anim = animate_photons_3d(paths, R_sun, n_photons)
        plt.show()
        print("Saving animation to 'photon_slab.mp4'...")
        anim.save("Photons_Near_Sun_Simulation.mp4", writer='ffmpeg', fps=30, dpi=150)
    elif (plot_sim is True) and (save_sim is False):
        paths = simulate_photons(R_sun, sigma_T, c, n_photons)
        anim = animate_photons_3d(paths, R_sun, n_photons)
        plt.show()
    else:
        simulate_photons(R_sun, sigma_T, c, n_photons)
        print("Simulation complete. No plot generated sadly :( ")


def n_e(r, R_sun):
    """
        Description:
        ------------
        Function Returning Electron Number Density Profile in the Sun
    """
    return 2.5e26 * np.exp(-r / (0.096 * R_sun))


def mean_free_path(r, sigma_T, R_sun):
    """
        Description:
        ------------
        Calculate Mean Free Path at radius r in the Sun
    """
    ne = n_e(r, R_sun)
    if ne < 1e-30:  # avoid zero division
        ne = 1e-30
    return 1.0 / (ne * sigma_T)


def free_path(lmbda):
    """
        Description:
        ------------
        Function Generating Free Path Lengths for Photons Based on Exponential Distribution
    """
    u = random.random()
    u = max(u, 1e-16)  # avoid log(0)
    return -lmbda * np.log(u)


def isotropic_direction():
    """
        Description:
        ------------
           Function Generating Isotropic Direction Arrays for Photon Scattering
    """
    mu = 2.0 * random.random() - 1.0
    phi = 2.0 * np.pi * random.random()
    sin_theta = np.sqrt(1 - mu**2)
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = mu
    direction_array = np.array([dx, dy, dz])
    return direction_array


def simulate_photon(R_sun, sigma_T, max_steps):
    """
        Description:
        ------------
        Function Simulating the Path of a Single Photon through the Sun in 3D
    """
    pos = np.array([0.0, 0.0, 0.0])
    path = [pos.copy()]
    tot_dis = 0.0

    for i in range(max_steps):
        r = np.linalg.norm(pos)
        if r >= 0.9 * R_sun:
            break
        lmbda = mean_free_path(r * 1e9, sigma_T, R_sun)
        s = free_path(lmbda)
        s_viz = s * 1e10  # scaling for visualization
        direction = isotropic_direction()
        pos += s * direction
        tot_dis += s
        path.append(pos.copy() * 1e10)  # scaling for visualization

    return np.array(path),tot_dis


def simulate_photons(R_sun, sigma_T, speed_light, n_photons, max_steps=10**7):
    """
        Description:
        ------------
        Function Simulating the Path of Multiple Photons through the Sun in 3D
    """
    paths = []
    for n in tqdm(range(n_photons), desc="Simulating Photons"):
        path, dist = simulate_photon(R_sun, sigma_T, max_steps)
        paths.append(path)

    # Analytical estimate of escape time since values from simulation did not appear realistic based on the setup for visualization
    lambda_core = mean_free_path(0, sigma_T, R_sun)
    R = 0.9 * R_sun
    s_total = (R ** 2) / lambda_core  # random walk estimate
    t_escape_sec = s_total / speed_light
    print(f"Approximate photon escape time: {t_escape_sec / 3.15e7:.2e} years")

    return paths


def animate_photons_3d(paths, R_sun, n_photons):
    """
        Description:
        ------------
        Function animating the 3D Photon Paths through the Sun
    """

    scale = 1e10  # scale to make plot clearer
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Photon Random Walk Through the Sun (3D)")
    ax.set_xlabel("x (scaled)")
    ax.set_ylabel("y (scaled)")
    ax.set_zlabel("z (scaled)")

    # Setting plot limits based on scaling
    lim = 5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


    # Plotting escape boundary (0.9 R_sun)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_b = 0.9 * R_sun * np.cos(u) * np.sin(v) * scale
    y_b = 0.9 * R_sun * np.sin(u) * np.sin(v) * scale
    z_b = 0.9 * R_sun * np.cos(v) * scale
    ax.plot_wireframe(x_b, y_b, z_b, color='orange', alpha=0.6, linewidth=0.5)

    # Assign colors for each photon with a colormap
    colors = plt.cm.magma(np.linspace(0, 1, n_photons))
    lines = [ax.plot([], [], [], lw=1, color=c, alpha=1)[0] for c in colors]
    dots = [ax.plot([], [], [], 'o', color=c, markersize=6, alpha=1)[0] for c in colors]

    # Determine max number of points to set animation frames
    max_len = max(len(p) for p in paths)

    def init():
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            line.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
        return lines + dots

    def photo_func(frame):
        for i, p in enumerate(paths):
            if frame < len(p):
                xs, ys, zs = p[:frame+1, 0], p[:frame+1, 1], p[:frame+1, 2]
                lines[i].set_data(xs, ys)
                lines[i].set_3d_properties(zs)
                dots[i].set_data([p[frame, 0]], [p[frame, 1]])
                dots[i].set_3d_properties([p[frame, 2]])
        return lines + dots

    # Creating the actual animation
    animation = FuncAnimation(fig, photo_func, frames=max_len, init_func=init, interval=50, blit=False)
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation of Photon Scattering in the Sun (3D)")
    parser.add_argument('-plot_sim', action='store_true',
                        help="Include this flag if you want to make a plot of the simulation results")
    parser.add_argument('-save_sim', action='store_true',
                        help="Include this flag if you want to save the animation of the simulation results")
    args = parser.parse_args()
    simulation_setup(plot_sim=args.plot_sim, save_sim=args.save_sim)
