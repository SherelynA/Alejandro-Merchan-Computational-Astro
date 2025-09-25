import matplotlib.pyplot as plt
import argparse
import astropy.constants as const


def function(r, G, M, m, R, w):
    return ((G * M )/ r**2) - ((G * m )/ ((R - r)**2)) - (w**2 * r)

def dfunction(r, G, M, m, R, w):
    return (-2 * G * M /(r**3)) - (2 * G * m / (R - r)**3) + (w**2)

def lagrange_point():
    parser = argparse.ArgumentParser(description="Calculate the distance to the Lagrange point L1 between the Earth and the Moon using the Newton method")
    parser.add_argument('-plot', action='store_true',help="Include this flag if you want to plot the positions of the Earth, Moon, and Lagrange point L1")
    args = parser.parse_args()
    #Setting Constants
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    M = const.M_earth.value  # Mass of the Earth in kg
    m = 7.348e22 # Mass of the Moon in kg
    R = 3.844e8  # Distance from Earth to Moon in meters
    w = 2.662e-6 # Angular velocity of the Earth-Moon system in s^-1

    r = 3.6e8    # Initial guess for the distance to the Lagrange point
    tol = 1e-6   # Tolerance for convergence

    while True:
        r_new = (r - function(r, G, M, m, R, w) / dfunction(r, G, M, m, R, w))
        if abs(r_new - r) < tol:
            break
        r = r_new # Setting r to the new value for the next iteration

    if args.plot:
        plt.style.use('dark_background')
        plt.axvline(R / 1e6, color='yellow', lw=0.8, label='Moon',zorder=2)
        plt.axvline(0, color='blue', lw=0.8, label='Earth',zorder=2)
        plt.axvline(r / 1e6, color='red', linestyle='--', label='L1 Point',zorder=2)
        plt.hlines(y=0.5, xmin=0, xmax=r/1e6, colors='pink', linestyles='dashed', label='Distance to L1 Point ~ ' + str(round(r / 1e6, 2)) + ' Mm',zorder=1)
        plt.scatter(r / 1e6, 0.5, color='red',zorder=2)  # Marking the Lagrange point
        plt.scatter(0, 0.5, color='blue', marker='o', s=1000,zorder=2)  # Marking the Earth
        plt.scatter(R / 1e6, 0.5, color='yellow', marker='o', s=200,zorder=2)  # Marking the Moon
        plt.xlabel('Distance from Earth (Megameters)')
        plt.ylabel('Arbitrary Spatial Units')
        plt.title('Finding the Lagrange Point L1')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    return print("Distance to the Lagrange point L1 from the Earth is approximately: " + str(
        round(r)) + " meters" + " or " + str(round(r / 1e6, 2)) + " Megameters")


if __name__ == "__main__":
    lagrange_point()