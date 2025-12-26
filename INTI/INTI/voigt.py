import numpy as np
from scipy.special import wofz as Faddeeva

# Physical constants
from INTI.constants import kb, c

# The Pycharm Co-pilot was used to write the descriptions for the parameters and return values for the functions in this file.

def Voigt_HWHM(gamma_L, alpha_D):
    """
    This function  computes the Voigt half-width at half-maximum (HWHM).

    Parameters:
    gamma_L (array-like or scalar) :
        Lorentzian HWHM
    alpha_D (array-like or scalar) :
        Doppler HWHM
    Returns:
    gamma_V (array-like or scalar) :
        Voigt HWHM

    """
    gamma_L = np.asarray(gamma_L, dtype=float)
    alpha_D = np.asarray(alpha_D, dtype=float)

    gamma_L2 = gamma_L * gamma_L

    return 0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L2 + alpha_D * alpha_D)


def Voigt_and_derivatives(nu, gamma, alpha, norm):
    """
    This function computes the Voigt profile and its derivatives with respect to alpha and nu.

    Parameters:
    nu (array-like) :
        Frequency grid
    gamma (float) :
        Lorentzian HWHM
    alpha (float) :
        Doppler HWHM
    norm (float) :
        Normalization factor
    Returns:
    Voigt (array-like) :
        Voigt profile values
    dV_da (array-like) :
        Derivative of Voigt profile with respect to alpha
    dV_dv (array-like) :
        Derivative of Voigt profile with respect to nu
    Returns:
    Voigt, dV_da, dV_dv
    """

    # Constants
    ln2 = np.log(2.0)
    sqrt_ln2 = np.sqrt(ln2)
    sqrt_pi = np.sqrt(np.pi)

    inv_alpha = 1.0 / alpha
    inv_alpha2 = inv_alpha * inv_alpha

    # Dimensionless variables
    x = sqrt_ln2 * nu * inv_alpha
    y = sqrt_ln2 * gamma * inv_alpha

    z = x + 1j * y
    W = Faddeeva(z)

    K = W.real
    L = W.imag

    # Voigt profile
    prefactor = np.sqrt(ln2 / np.pi) * inv_alpha
    Voigt = prefactor * K / norm

    # Derivatives
    b1 = y / sqrt_pi
    b2 = (x * x - y * y) - 0.5
    b3 = -2.0 * x * y

    const_da = 2.0 * inv_alpha2 * np.sqrt(ln2 / np.pi)
    const_dnu = 2.0 * inv_alpha2 * np.sqrt(ln2 * ln2 / np.pi)

    dV_da = const_da * (b1 + b2 * K + b3 * L) / norm
    dV_dv = const_dnu * (y * L - x * K) / norm

    return Voigt, dV_da, dV_dv


def Generate_Voigt_grid_molecules( Voigt_arr, dV_da_arr, dV_dnu_arr, gamma_arr, alpha_arr, cutoffs, N_Voigt):
    """
    Generate Voigt profile grid and its derivatives for molecules.

    Parameters:
    Voigt_arr (array-like) :
        Voigt profile array to be filled
    dV_da_arr (array-like) :
        Derivative of Voigt profile with respect to alpha array to be filled
    dV_dnu_arr (array-like) :
        Derivative of Voigt profile with respect to nu array to be filled
    gamma_arr (array-like) :
        Lorentzian HWHM array
    alpha_arr (array-like) :
        Doppler HWHM array
    cutoffs (array-like) :
        Cutoff frequencies array
    N_Voigt (array-like) :
        Number of Voigt points array

    """

    norm = 0.998  # empirical normalization

    for i, gamma in enumerate(gamma_arr):
        for j, alpha in enumerate(alpha_arr):

            Nij = N_Voigt[i, j]
            nu_max = cutoffs[i, j]

            nu = np.linspace(0.0, nu_max, Nij)

            V, dV_da, dV_dnu = Voigt_and_derivatives(
                nu, gamma, alpha, norm
            )

            Voigt_arr[i, j, :Nij] = V
            dV_da_arr[i, j, :Nij] = dV_da
            dV_dnu_arr[i, j, :Nij] = dV_dnu


def precompute_molecules(nu_compute, dnu_out, m, T, Voigt_sub_spacing, Voigt_cutoff, N_alpha_samples, gamma_L, cut_max):
    """
        Precompute Voigt profile grid and its derivatives for molecules.

        Parameters:
        nu_compute (array-like):
            Frequency grid for computation
        dnu_out (float):
            Output frequency spacing
        m   (float):
            Molecular mass
        T   (float):
            Temperature
        Voigt_sub_spacing (float):
            Voigt sub-grid spacing factor
        Voigt_cutoff (float):
            Voigt cutoff factor
        N_alpha_samples (int):
            Number of alpha samples
        gamma_L (array-like):
            Lorentzian HWHM array
        cut_max (float):
            Maximum cutoff value

        Returns
        -------
        nu_sampled (array-like):
            Sampled frequency grid
        alpha_sampled (array-like):
            Sampled Doppler HWHM grid
        cutoffs (array-like):
            Cutoff frequencies array
        N_Voigt (array-like):
            Number of Voigt points array
        Voigt_arr (array-like):
            Voigt profile array
        dV_da_arr (array-like):
            Derivative of Voigt profile with respect to alpha array
        dV_dnu_arr (array-like):
            Derivative of Voigt profile with respect to nu array
        dnu_Voigt (array-like):
            Voigt frequency spacing array
        """

    # Frequency sampling
    nu_sampled = np.logspace(
        np.log10(nu_compute[0]),
        np.log10(nu_compute[-1]),
        N_alpha_samples
    )

    # Doppler widths
    alpha_sampled = (
        np.sqrt(2.0 * kb * T * np.log(2.0) / m)
        * (nu_sampled / c)
    )

    # Voigt HWHM (broadcasting over J and alpha)
    gamma_V = Voigt_HWHM(
        gamma_L[:, None],
        alpha_sampled[None, :]
    )

    # Cutoffs and spacing
    cutoffs = np.minimum(Voigt_cutoff * gamma_V, cut_max)
    dnu_Voigt = np.minimum(gamma_V * Voigt_sub_spacing, dnu_out)

    # Number of grid points
    N_Voigt = np.rint(cutoffs / dnu_Voigt).astype(np.int64) + 1
    dnu_Voigt = cutoffs / (N_Voigt - 1)

    max_N = np.max(N_Voigt)

    # Allocate arrays
    shape = (len(gamma_L), len(alpha_sampled), max_N)
    Voigt_arr = np.zeros(shape)
    dV_da_arr = np.zeros(shape)
    dV_dnu_arr = np.zeros(shape)

    # Populate grids
    Generate_Voigt_grid_molecules(
        Voigt_arr, dV_da_arr, dV_dnu_arr,
        gamma_L, alpha_sampled, cutoffs, N_Voigt
    )

    return (nu_sampled, alpha_sampled, cutoffs, N_Voigt, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt)

