import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from astropy.table import Table
from cmath import exp, pi
from astropy.io import fits

def main():
    times, fluxs, flux_errors = star_data()
    c = dft(fluxs)
    x_rec = idft(c,fluxs)
    plt.figure(figsize=(8, 4))
    plt.plot(times, fluxs, label="Original Data", alpha=0.6)
    plt.plot(times, x_rec, label="Reconstructed (IDFT)", lw=1.5)
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()

def star_data():
    binary_path = 'tic0458859364.fits'  # Path to the FITS file
    hdul = fits.open(binary_path)  # Opening the FITS file
    times = hdul[1].data['times']  # Creating a variable for the times column
    fluxs = hdul[1].data['fluxes']  # Creating a variable for the fluxes column
    flux_errs = hdul[1].data['ferrs']  # Creating a variable for the flux errors column
    mask_data = (times > 2200) & (times < 2400)  # Mask to filter the data between 2200 and 2800 days
    mtimes = times[mask_data]  # Applying the mask to the times
    mfluxs = fluxs[mask_data]  # Applying the mask to the fluxes
    mflux_errs = flux_errs[mask_data]  # Applying the mask to the flux errors
    return mtimes, mfluxs, mflux_errs


def dft(y):
    # Function for Discrete Fourier Transform
    N = len(y)
    c = zeros(N//2+1, dtype=complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n] * exp(-2j * pi * k * n / N)

    return c


def idft(c,y):
    # Function for Inverse Discrete Fourier Transform
    N = len(y)
    x_rec = zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N // 2 + 1):
            x_rec[n] += c[k] * exp(2j * pi * k * n / N)
            if k not in [0, N // 2]:
                x_rec[n] += np.conj(c[k]) * exp(-2j * pi * k * n / N)

    x_rec /= N
    return x_rec.real

if __name__ == "__main__":
    main()
