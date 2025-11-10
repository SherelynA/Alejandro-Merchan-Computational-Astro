import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# This code was made in conjunction with Pycharm Co-Pilot, particularly for the visualization which uses AI (Mellum) to help generate code snippets.

def orbit_setup():
    """
        Description:
        ------------
        Function to set up initial conditions for the orbit of a ball bearing around a rod.
    """
    a = 0       # start time
    b = 10      # end time
    N = 1000     # number of steps
    h = (b - a) / N

    x0, y0, vx0, vy0 = 1.0, 0.0, 0.0, 1.0
    ball_state = np.array([x0, y0, vx0, vy0])
    time_points = np.arange(a, b, h)

    print("Initial ball state conditions for position and velocity:", ball_state)
    return ball_state, h, time_points


def f(state, t, M = 10.0, L = 2.0):
    """
        Description:
        ------------
        Function to compute derivatives with the provided equations of motion
    """
    vx, vy = state[2], state[3]
    r = np.sqrt(state[0]**2 + state[1]**2)
    ax = -M * state[0] / (r**2 * np.sqrt(r**2 + (L/2)**2))
    ay = -M * state[1] / (r**2 * np.sqrt(r**2 + (L/2)**2))
    return np.array([vx, vy, ax, ay])


def rk4_orbit(r, h, time_points):
    """
        Description:
        ------------
        Function to perform RK4 integration for the orbit of the ball bearing.
    """
    x_points = []
    y_points = []
    for t in time_points:
        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_points.append(r[0])
        y_points.append(r[1])
    print("Completed RK4 integration for orbit calculation.")
    return x_points, y_points


def animation_setup(line, dot, x, y):
    """
        Description:
        ------------
        Function to set up the animation functions for the orbit visualization.
    """

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def update(i):
        line.set_data(x[:i], y[:i])
        dot.set_data(x[i], y[i])
        return line, dot

    return init, update


def plot_orbit():
    """
        Description:
        ------------
        Function to plot and animate the orbit of the ball bearing around the rod.
    """
    print("Setting up orbit simulation for the orbit of a ball bearing around a rod.")
    ball_state, h, time_points = orbit_setup()

    print("Starting RK4 integration for orbit calculation. Yay")
    x, y = rk4_orbit(ball_state, h, time_points)
    print("Now displaying an animation for the orbit of a ball bearing around a rod.")
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Orbit of Ball Bearing around Rod')
    ax.grid(True)

    # Initialize line and dot for animation
    (line,) = ax.plot([], [], lw=1.5, color='purple')
    (dot,) = ax.plot([], [], 'ro', markersize=5)

    init, update = animation_setup(line, dot, x, y)

    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, interval=10, blit=True)
    plt.show()
    print("Thank you for using this orbit simulation program! - Sherelyn")


if __name__ == "__main__":
    plot_orbit()
