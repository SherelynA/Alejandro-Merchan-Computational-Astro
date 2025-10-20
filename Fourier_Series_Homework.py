# Necessary Imports
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from cmath import exp, pi
from astropy.io import fits


def eclipsing_binary(file_path, dft=False, idft=False, plot_IDFT=False, plot_OBS=False):
    """
    Description:
    ------------
    Function Analyzing Flux Observations for Eclipsing Binaries

    Parameters:
    -----------
    - file_path: string
        File path of the data file
    - dft: bool, default: False
        Calculating a Discrete Fourier Transformation from user data. Returns coefficients.
    - idft: bool, default: False
        Calculating an Inverse Discrete Fourier Transformation from coefficients. Returns reconstructed y values.
    - plot_IDFT: bool, default: False
        Plot the Inverse Discrete Fourier Transformation
    - plot_OBS: bool, default: False
        Plot the Observed Data from the provided path
    Example:
    ------------

    """

    times, fluxs, flux_errors = eb_data(file_path)

    # Calling the Discrete Fourier Transformation Function

    if ((dft == True) and (idft == True)):
        c = discrete_fourier_trans(fluxs)
        x_rec = inverse_discrete_fourier_trans(c, len(fluxs))

        if ((plot_OBS == True) and (plot_IDFT == True)):
            print("Creating a plot of the Observed Data and Reconstructed Data from the IDFT")
            plt.figure(figsize=(8, 4))
            plt.plot(times, fluxs, label="Observed Data", color='k')
            plt.plot(times, x_rec, label="Reconstructed Data from (IDFT)", lw=0.9, color='magenta')
            plt.legend()
            plt.xlabel("Barycentric Julian Date")
            plt.ylabel("Flux")
            plt.show()

        elif ((plot_OBS == True) and (plot_IDFT == False)):
            print("Creating a plot of the Observed Data")
            plt.figure(figsize=(8, 4))
            plt.plot(times, fluxs, label="Observed Data", color='k')
            plt.legend()
            plt.xlabel("Barycentric Julian Date")
            plt.ylabel("Flux")
            plt.show()

        elif ((plot_OBS == False) and (plot_IDFT == True)):
            print("Creating a plot of the Reconstructed Data from the IDFT")
            plt.figure(figsize=(8, 4))
            plt.plot(times, x_rec, label="Reconstructed Data from (IDFT)", lw=0.9, color='magenta')
            plt.legend()
            plt.show()
        else:
            print("No plots were asked for by Prof. Ari!")

    elif ((dft == True) and (idft== False)):
        c = discrete_fourier_trans(fluxs)
        if ((plot_OBS == True) and (plot_IDFT == False)):
            print("Creating a plot of the Observed Data")
            plt.figure(figsize=(8, 4))
            plt.plot(times, fluxs, label="Observed Data", color='k')
            plt.legend()
            plt.xlabel("Barycentric Julian Date")
            plt.ylabel("Flux")
            plt.show()
        elif ((plot_OBS == False) and (plot_IDFT == True)):
            print("A plot of the Inverse Fourier Transformation was asked but the analysis itself was not called. Please change idft=True")
            raise ValueError
        else:
            print("No plots were asked for by Prof. Ari!")
    elif ((dft == False) and (idft== True)):
        print("The coefficients from the Fourier Transformation are needed for the Inverse Fourier Transformation, but the DFT was not called. Please change DFT to True")
        raise ValueError
    else:
        print("No analysis was chosen by the user, please choose an analysis of the provided observed data.")


def eb_data(file_path):
    """

    Function for reading in TESS FITS file for Eclipsing Binary

    """

    # binary_path = 'tic0458859364.fits'  # Path to the FITS file
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
    else:
        print(f"Error: File not found at {file_path}")
        raise ValueError

    hdul = fits.open(file_path)  # Opening the FITS file
    times = hdul[1].data['times']  # Creating a variable for the times column
    fluxs = hdul[1].data['fluxes']  # Creating a variable for the fluxes column
    flux_errs = hdul[1].data['ferrs']  # Creating a variable for the flux errors column
    mask_data = (times > 2320) & (times < 2330)  # Mask to filter the data between 2320 and 2330 in Barycentric Julian Date
    mtimes = times[mask_data]  # Applying the mask to the times
    mfluxs = fluxs[mask_data]  # Applying the mask to the fluxes
    mflux_errs = flux_errs[mask_data]  # Applying the mask to the flux errors
    return mtimes, mfluxs, mflux_errs


def discrete_fourier_trans(y):
    """

    Function for Discrete Fourier Transformation

    """
    N = len(y)
    c = zeros(N//2+1, dtype=complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n] * exp(-2j * pi * k * n / N)

    return c


def inverse_discrete_fourier_trans(c,N):
    """

    Function for Inverse Discrete Fourier Transformation

    """
    x_rec = zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N // 2 + 1):
            x_rec[n] += c[k] * exp(2j * pi * k * n / N)
            if k not in [0, N // 2]:
                x_rec[n] += np.conj(c[k]) * exp(-2j * pi * k * n / N)

    x_rec /= N
    return x_rec.real


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyzing  Flux Observations for Eclipsing Binaries")
    parser.add_argument("file_path", type=str, help="The path to the file to be processed.")
    parser.add_argument('-dft', action='store_true',
                        help="Include this flag if you want to calculate the DFT")
    parser.add_argument('-idft', action='store_true',
                        help="Include this flag if you want to make a plot of the IDFT")
    parser.add_argument('-plot_IDFT', action='store_true',
                        help="Include this flag if you want to make a plot of the IDFT")
    parser.add_argument('-plot_OBS', action='store_true',
                        help="Include this flag if you want to make a plot of the Observed Data")
    args = parser.parse_args()

    eclipsing_binary(args.file_path, dft=args.dft, idft=args.idft, plot_IDFT=args.plot_IDFT, plot_OBS=args.plot_OBS)
