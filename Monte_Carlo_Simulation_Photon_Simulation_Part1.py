# Necessary Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from matplotlib.animation import FuncAnimation
import random


def simulation_setup(plot_sim=False, save_sim=False):
    # Physical constants and parameters
    sigma_t = const.sigma_T.cgs.value  # Thomson cross-section [cm^2]
    n_electron = 1e20  # electron number density [cm^-3]
    slab_width = 1e5  # 1 km = 1e5 cm
    mean_free_path = 1.0 / (n_electron * sigma_t)  # cm  (~150 m)
    max_scatters = 1000  # Maximum number of scatters per photon
    if (plot_sim is True) and (save_sim is True):
        paths, outcomes, n_photons = photon_simulations(mean_free_path, slab_width, max_scatters)
        anim = photon_animation(paths, slab_width, n_photons)
        plt.show()
        print("Saving animation to 'photon_slab.mp4'...")
        anim.save("Photon_Path_Simulation.mp4", writer='ffmpeg', fps=30, dpi=150)
    elif (plot_sim is True) and (save_sim is False):
        paths, outcomes, n_photons = photon_simulations(mean_free_path, slab_width, max_scatters)
        anim = photon_animation(paths,slab_width, n_photons)
        plt.show()
    else:
        photon_simulations(mean_free_path, slab_width, max_scatters)
        print("Simulation complete. No plot generated sadly :( ")


def free_path(lmbda):
    u = random.random()
    return -lmbda * np.log(u)


def isotropic_mu():
    return 2.0 * random.random() - 1.0


def simulate_photon_path(mean_free_path, slab_width, max_scatters, mu0=1.0):

    # Initializing photon position and direction, along with mu
    x = 0.0
    z = 0.0
    mu = mu0
    n_scatters = 0
    path = [(x, z)]

    while True:
        s = free_path(mean_free_path)
        n_scatters += 1
        z_new = z + s * mu
        # Providing random azimuthal angle
        phi = 2 * np.pi * random.random()
        # Converting to cartesian coordinates and ensuring realistic values
        dx = s * np.sqrt(max(0.0, 1 - mu**2)) * np.cos(phi)
        x_new = x + dx
        # Append new position to path
        path.append((x_new, z_new))
        x, z = x_new, z_new
        # Determine if photon has reflected, transmitted, or if max scatters reached
        if z < 0:
            return path, 'reflected'
        elif z > slab_width:
            return path, 'transmitted'
        elif n_scatters > max_scatters:
            return path, 'terminating'
        # Update mu for next scatter
        mu = isotropic_mu()


# Run a set of photon simulations
def photon_simulations(mean_free_path, slab_width, max_scatters, number_photons=100):
    paths = []
    outcomes = []
    for i in range(number_photons):
        p, outcome = simulate_photon_path(mean_free_path, slab_width, max_scatters)
        paths.append(np.array(p))
        outcomes.append(outcome)

    print("Photon Simulations are finally finished!")
    print(f"Transmitted Photons: {outcomes.count('transmitted')}, "
          f"Reflected Photons: {outcomes.count('reflected')}")

    return paths, outcomes, number_photons


def photon_animation(paths, slab_width, number_photons):

    # Calculate photon paths and outcomes
    # Set up animation of photon paths
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    ax.set_xlim(-0.5e5, 0.5e5)
    ax.set_ylim(-0.1*slab_width, 1.1*slab_width)  # Basing on slab width
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title("Photon Trajectory Simulations through a 1 km slab")

    # Draw slab boundaries
    ax.axhline(0, color='k', lw=2)
    ax.axhline(slab_width, color='k', lw=2)

    # Assign colors for each photon with a colormap
    colors = plt.cm.magma(np.linspace(0, 1, number_photons))
    lines = [ax.plot([], [], color=c, lw=1)[0] for c in colors]
    dots = [ax.plot([], [], 'o', color=c, markersize=5)[0] for c in colors]

    # Determine max number of points to set animation frames
    maxlen = max(len(p) for p in paths)

    def init():
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
        return lines + dots

    def photo_func(frame):
        for i, p in enumerate(paths):
            if frame < len(p):
                xs = p[:frame+1, 0]
                zs = p[:frame+1, 1]
                lines[i].set_data(xs, zs)
                dots[i].set_data(p[frame, 0], p[frame, 1])
        return lines + dots

    animation = FuncAnimation(fig, photo_func, frames=maxlen, init_func=init, interval=80, blit=True)
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation of Photon Scattering in a Slab")
    parser.add_argument('-plot_sim', action='store_true',
                        help="Include this flag if you want to make a plot of the simulation results")
    parser.add_argument('-save_sim', action='store_true',
                        help="Include this flag if you want to save the animation of the simulation results")
    args = parser.parse_args()
    simulation_setup(plot_sim=args.plot_sim, save_sim=args.save_sim)
