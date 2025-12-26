import os
import numpy as np
import pandas as pd

# The Pycharm Co-pilot was used to write the descriptions for the parameters and return values for the functions in this file.

def determine_broad(input_directory):
    """
    Determines the broadening parameters for spectral lines from the specified input directory.
    """
    files = set(os.listdir(input_directory))

    if {'H2.broad', 'He.broad'}.issubset(files):
        broadening = 'H2-He'
        return broadening
    elif 'air.broad' in files:
        broadening = 'air'
        return broadening
    else:
        raise FileNotFoundError(
            'No broadening files found. '
            'Expected H2.broad + He.broad or air.broad.'
        )


def read_H2_He(input_directory):
    """
    Reads the H2 and He broadening files and extracts necessary parameters.
    """

    def _read_broad_file(filename):
        df = pd.read_csv(
            os.path.join(input_directory, filename),
            sep=' ',
            header=None,
            skiprows=1
        )
        J = df[0].to_numpy()
        gamma_0 = df[1].to_numpy()
        n_L = df[2].to_numpy()
        return J, gamma_0, n_L

    # Read files
    J_H2, gamma_0_H2, n_L_H2 = _read_broad_file('H2.broad')
    J_He, gamma_0_He, n_L_He = _read_broad_file('He.broad')

    J_max_H2 = int(J_H2.max())
    J_max_He = int(J_He.max())
    J_max = max(J_max_H2, J_max_He)

    # Determine J grid
    J_broad_all = J_H2 if J_max_H2 >= J_max_He else J_He

    # Extend arrays if needed (pad with last value)
    if J_max_H2 < J_max:
        pad = J_max - J_max_H2
        gamma_0_H2 = np.pad(gamma_0_H2, (0, pad), mode='edge')
        n_L_H2 = np.pad(n_L_H2, (0, pad), mode='edge')

    if J_max_He < J_max:
        pad = J_max - J_max_He
        gamma_0_He = np.pad(gamma_0_He, (0, pad), mode='edge')
        n_L_He = np.pad(n_L_He, (0, pad), mode='edge')

    return J_max, J_broad_all, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He


def read_air(input_directory):
    """
    Reads the air broadening file and extracts necessary parameters.
    """
    df = pd.read_csv(
        os.path.join(input_directory, 'air.broad'),
        sep=' ',
        header=None,
        skiprows=1
    )

    J_broad_all = df[0].to_numpy()
    gamma_0_air = df[1].to_numpy()
    n_L_air = df[2].to_numpy()
    J_max = int(J_broad_all.max())

    return J_max, J_broad_all, gamma_0_air, n_L_air


def compute_H2_He(
    gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2,
    gamma_0_He, n_L_He, X_He
):
    """
    Computes the H2 + He broadened Lorentzian HWHM.
    """
    T_ratio = T_ref / T
    P_ratio = P / P_ref

    gamma = (
        gamma_0_H2 * (T_ratio ** n_L_H2) * P_ratio * X_H2 +
        gamma_0_He * (T_ratio ** n_L_He) * P_ratio * X_He
    )

    return gamma


def compute_air(gamma_0_air, T_ref, T, n_L_air, P, P_ref):
    """
    Computes the air-broadened Lorentzian HWHM.
    """
    T_ratio = T_ref / T
    P_ratio = P / P_ref

    gamma = gamma_0_air * (T_ratio ** n_L_air) * P_ratio
    return gamma

