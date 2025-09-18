import numpy as np
import argparse
import matplotlib.pyplot as plt

def function(t):
    return np.exp(-t**2)

def integration():
    parser = argparse.ArgumentParser(description="Integrate the function e^(-t)^2 from 0 to x using chosen integration method")
    parser.add_argument('X', type=int, help="upper limit of integration")
    parser.add_argument('H', type=float, help="step size to use in the integration")
    parser.add_argument('N', type=int, help="number slices to use in the integration")
    parser.add_argument('-method', choices=["Simpson", "Trapezoidal"], help="Input integration method to evaluate the integral (default is Trapezoidal)", default="Trapezoidal")
    parser.add_argument('-plot', action='store_true', help="Include this flag if you want to plot the function being integrated")
    args = parser.parse_args()

    # Setting parameters from argparse
    upper_lim = args.X
    N = args.N
    h = args.H

    if args.method == "Trapezoidal":

        a = 0
        b = upper_lim

        s = 0.5*(function(a) + 0.5*function(b))

        for k in range(1,N):
            s+=  function(a + k*h)

        evaluated_trap = h*s

        if args.plot:
            t = np.linspace(0, upper_lim, N)
            plt.plot(t, function(t), label=r'$e^{-t^2}$',color='pink')
            plt.fill_between(t, function(t), where=(t >= 0) & (t <= upper_lim), color='purple', alpha=0.5)
            plt.title(r'Function $e^{-t^2}$ from 0 to ' + str(upper_lim))
            plt.xlabel('t')
            plt.ylabel(r'$e^{-t^2}$')
            plt.legend()
            plt.grid()
            plt.show()

        return print("The Integral evaluated from 0 to " + str(upper_lim) + " using the Trapezoidal rule is: " + str(evaluated_trap))

    elif args.method == "Simpson":

        if N % 2 == 1:
            N += 1
            print("N must be even for Simpson's rule, so N has been increased by 1 to be: " + str(N))

        a = 0
        b = upper_lim

        N1 = (N // 2)
        N2 = (N // 2) - 1

        s = (function(a) + function(b))

        for i in np.arange(1, N1, 1): # odd indices
            s += 4 * function(a + ((2 * i - 1) * h))
        for i in np.arange(1, N2, 1): # even indices
            s += 2 * function(a + (2 * i * h))

        evaluated_simp = (1 / 3) * (h*s)

        if args.plot:
            t = np.linspace(0, upper_lim, N)
            plt.plot(t, function(t), label=r'$e^{-t^2}$',color='pink')
            plt.fill_between(t, function(t), where=(t >= 0) & (t <= upper_lim), color='purple', alpha=0.5)
            plt.title(r'Function $e^{-t^2}$ from 0 to ' + str(upper_lim))
            plt.xlabel('t')
            plt.ylabel(r'$e^{-t^2}$')
            plt.legend()
            plt.grid()
            plt.show()

        return print("The Integral evaluated from 0 to " + str(upper_lim) + " using Simpson's rule is: " + str(evaluated_simp))

    else:
        return print("Please select a valid integration method and if all else fails solve it by hand!")


if __name__ == "__main__":
    integration()
