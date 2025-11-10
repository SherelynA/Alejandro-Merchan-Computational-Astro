import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Condition of Orbit Scenario


def orbit_setup():
    a = 0       # start time
    b = 10      # end time
    N = 1000     # number of steps
    h = (b - a) / N
    #Initial Conditions for position and
    x0, y0, vx0, vy0 = 1.0, 0.0, 0.0, 1.0
    ball_state = np.array([x0, y0, vx0, vy0])
    time_points = np.arange(a, b, h)
    return ball_state, h, time_points


# Function to compute derivatives for x and y
def f(state, t, M = 10.0, L = 2.0):
    vx, vy = state[2], state[3]
    r = np.sqrt(state[0]**2 + state[1]**2)
    ax = -M * state[0] / (r**2 * np.sqrt(r**2 + (L/2)**2))
    ay = -M * state[1] / (r**2 * np.sqrt(r**2 + (L/2)**2))
    return np.array([vx, vy, ax, ay])


# Function for RK4 Integration of Orbit
def rk4_orbit(r, h, time_points):
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
    return x_points, y_points


# Animation Functions
def animation_setup(line, dot, x, y):
    # Initialization function for animation
    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot
    # Update function for animation
    def update(i):
        line.set_data(x[:i], y[:i])
        dot.set_data(x[i], y[i])
        return line, dot

    return init, update

def plot_orbit():
    ball_state, h, time_points = orbit_setup()
    x, y = rk4_orbit(ball_state, h, time_points)
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Orbit of Ball Bearing around Rod')
    ax.grid(True)
    ax.set_aspect('equal')

    # Initialize line and dot for animation
    (line,) = ax.plot([], [], lw=1.5, color='purple')
    (dot,) = ax.plot([], [], 'ro', markersize=5)

    init, update = animation_setup(line, dot, x, y)

    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":
    plot_orbit()
