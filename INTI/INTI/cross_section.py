# Necessary Imports
import numpy as np
import time
import h5py
from tqdm import tqdm
from INTI.constants import kb, c, T_ref,c2

# The Pycharm Co-pilot was used to write the descriptions for the parameters and return values for the functions in this file.

def determine_index(value,grid_start,grid_end,grid_size):
    """
    Determine the index of a value in a grid.

    Parameters:
    value (float):
        The value to locate.
    grid_start (float):
        The start of the grid.
    grid_end (float):
        The end of the grid.
    grid_size (int):
        The number of points in the grid.

    Returns:
    index: The index of the value in the grid.
    """
    if (value < grid_start):
        return 0
    elif (value > grid_end):
        return grid_size - 1
    else:
        i = (grid_size - 1) * ((value - grid_start) / (grid_end - grid_start))
        if ((i%1.0) <= 0.5):

            return int(i)

        else:
            return int(i)+1


def prior_index(value,grid_start,grid_end,grid_size):
    """
    Determine the prior index of a value in a grid.

    Parameters:
    value (float):
        The value to locate.
    grid_start (float):
        The start of the grid.
    grid_end (float):
        The end of the grid.
    grid_size (integer):
        The number of points in the grid.

    Returns:
    index: The prior index of the value in the grid.
    """
    if (value < grid_start):
        return 0
    elif (value > grid_end):
        return grid_size - 1
    else:
        i = (grid_size - 1) * (value - grid_start) / (grid_end - grid_start)
        index = int(i)
        return index


def general_prior_index(value, grid):
    """
    Determine the prior index of a value in a general grid.

    Parameters:
    value (float):
        The value to locate.
    grid (np.ndarray):
        The grid array.

    Returns:
    int: The prior index of the value in the grid.
    """

    if (value > grid[-1]):
        return (len(grid) - 1)
    if (value < grid[0]):
        value = grid[0]
    if (value > grid[-2]):
        value = grid[-2]

    index = 0

    for i in range(len(grid)):
        if (grid[i] > value):
            index = i - 1
            break

    return index


def calculate_transition_freq(E , states, upper_state, lower_state):
    """
    Calculate the transition frequency between two states.

    Parameters:
    E (np.ndarray):
        Energy levels of the states.
    states (list):
        List of states.
    upper_state (list):
        List of upper states for transitions.
    lower_state (list):
        List of lower states for transitions.

    Returns:
    nu_trans (np.ndarray): Transition frequencies.

    """
    nu_trans = np.zeros(len(states))

    # Check condition of data for the states
    if states[-1] == len(states):
        complete_data = True
    else:
        complete_data = False

    for i in range(len(states)):
        if complete_data:
            E_upper = E[upper_state[i]-1]
            E_lower = E[lower_state[i]-1]
        else:
            E_upper = E[general_prior_index(upper_state[i], states)]
            E_lower = E[general_prior_index(lower_state[i], states)]

        nu_trans[i] = E_upper - E_lower

    return nu_trans


def calculate_line_intensity(S_ref, Q_T, Q_ref, T_ref, T, E_low, nu_0):
    """
    Calculate the line intensity at a given temperature.

    Parameters:
    S_ref (np.ndarray): Reference line intensities.
    Q_T (float): Partition function at temperature T.
    Q_ref (float): Reference partition function.
    T_ref (float): Reference temperature.
    T (float): Temperature at which to calculate line intensity.
    E_low (np.ndarray): Lower state energies.
    nu_0 (np.ndarray): Transition frequencies.

    Returns:
    np.ndarray: Line intensities at temperature T.
    """
    # S = S_ref * ((Q_ref / Q_T) * np.exp((-c2 * E_low / T)) / np.exp(-c2 * E_low / T_ref)) * \
    #       ((1 - np.exp(-c2 * nu_0 / T)) / (1 - np.exp(-c2 * nu_0 / T_ref)))
    S = S_ref * ((Q_ref / Q_T) * np.exp(-1.0*c2*E_low*((1.0/T) - (1.0/T_ref))) *
                 ((1.0 - np.exp(-1.0*c2*nu_0/T))/(1.0 - np.exp(-1.0*c2*nu_0/T_ref))))

    return S


def calculate_cross_section(sigma, nu_grid, nu_0,cutoffs, S, J_low, J_broad_all,
                            alpha, log_alpha, alpha_sampled, log_alpha_sampled,
                            N_voigt, Voigt_array, dV_da_array, dV_dnu_array, dnu_Voigt, dnu_out):
    """
    Calculate the cross-section for given parameters.
    Parameters:
    sigma (np.ndarray):
        Cross section array to be filled.
    nu_grid (np.ndarray):
        Frequency grid.
    nu_0 (np.ndarray):
        Transition frequencies.
    cutoffs (np.ndarray):
        Cutoff frequencies array.
    S (np.ndarray):
        Line strengths.
    J_low (np.ndarray):
        Lower state rotational quantum numbers.
    J_broad_all (np.ndarray):
        Broadening quantum numbers for all lines.
    alpha (np.ndarray):
        Doppler broadening parameters.
    log_alpha (np.ndarray):
        Logarithm of Doppler broadening parameters.
    alpha_sampled (np.ndarray):
        Sampled Doppler broadening parameters.
    log_alpha_sampled (np.ndarray):
        Logarithm of sampled Doppler broadening parameters.
    N_voigt (int):
        Number of points in Voigt profile.
    Voigt_array (np.ndarray):
        Precomputed Voigt profile array.
    dV_da_array (np.ndarray):
        Derivative of Voigt profile w.r.t. a.
    dV_dnu_array (np.ndarray):
        Derivative of Voigt profile w.r.t. nu.
    dnu_Voigt (np.ndarray):
        Frequency step in Voigt profile.
    dnu_out (float):
        Output frequency spacing.
    Returns:
    (None) The function modifies the sigma array in place.
    """
    # Store variables that are constant across all lines to save lookup time
    N_grid = len(nu_grid)
    nu_grid_min = nu_grid[0]
    nu_grid_max = nu_grid[-1]
    log_alpha_sampled_min = log_alpha_sampled[0]
    log_alpha_sampled_max = log_alpha_sampled[-1]
    N_log_alpha_sampled = len(log_alpha_sampled)

    # This method of calculatin transitions comes from the Cthulhu code in Agrawal & MacDonald (2024).
    for i in tqdm(range(len(nu_0)), desc="Calculating transitions", unit="lines"):

        # Store commonly used quantities as variables to save lookup time
        J_i = J_low[i]
        S_i = S[i]
        nu_0_i = nu_0[i]

        # Find index in sampled alpha array closest to actual alpha (approximate thermal broadening)
        idx_alpha = determine_index(log_alpha[i], log_alpha_sampled_min, log_alpha_sampled_max, N_log_alpha_sampled)

        # Find index in lower J broadening file array corresponding the lower J of this transition

        idx_J_i = general_prior_index(J_i, J_broad_all)

        # Store wing cutoff for this transition
        cutoff = cutoffs[idx_J_i, idx_alpha]

        # Load template Voigt function and derivatives for this gamma (J_i) and closest value of alpha
        Voigt_0 = Voigt_array[idx_J_i, idx_alpha, :]
        dV_da_0 = dV_da_array[idx_J_i, idx_alpha, :]
        dV_dnu_0 = dV_dnu_array[idx_J_i, idx_alpha, :]

        # Load number of template Voigt function wavenumber points and grid spacing
        dnu_Voigt_line = dnu_Voigt[idx_J_i, idx_alpha]

        # Find difference between true alpha and closest pre-computed value
        d_alpha = (alpha[i] - alpha_sampled[idx_alpha])

        # Store grid spacing ratio between the output and pre-computed line grid
        R_nu = dnu_out / dnu_Voigt_line

        left_cut_loc = ((nu_0_i - cutoff) - nu_grid_min) / dnu_out
        right_cut_loc = ((nu_0_i + cutoff) - nu_grid_min) / dnu_out

        # Compute exact location of line core (float in output grid units)
        core_loc = (nu_0_i - nu_grid_min) / dnu_out

        # If line core through both left and right cutoffs don't intersect grid points
        if ((left_cut_loc > int(core_loc)) and (right_cut_loc < (int(core_loc) + 1))):
            # No contribution to cross-section at grid values
            pass

        # If a grid point lies within left cutoff but not within the right cutoff
        elif ((left_cut_loc < int(core_loc)) and (right_cut_loc < (int(core_loc) + 1))):

            # Here only one grid point on the left wing contributes to the cross section

            # Find leftmost grid point within left wing cutoff
            if (left_cut_loc > 0.0):
                idx_left = int(left_cut_loc) + 1  # Round down then add 1 to find desired grid point
            else:  # Exception for lines near lower edge boundary
                idx_left = int(left_cut_loc - 1) + 1  # Round down then add 1 to find desired grid point

            # Cover edge case: the lowest idx_right is the left edge of the grid
            if (idx_left <= 0):
                idx_left = 0

            # Compute exact location of first grid point within left wing cutoff (float in template grid units)
            k_ref_exact = (core_loc - idx_left) * R_nu  # R_nu maps from output grid to template grid spacing

            # Round to find nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)

            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[idx_left] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                      (dV_dnu_0[k_ref] * d_Delta_nu))

        # If a grid point lies within right cutoff but not within the left cutoff
        elif ((left_cut_loc > int(core_loc)) and (right_cut_loc > (int(core_loc) + 1))):

            # Here only one grid point on the right wing contributes to the cross section

            # Find rightmost grid point within right wing cutoff
            idx_right = int(right_cut_loc)  # Round down to find desired grid point

            # Cover edge case: the highest idx_right is the right edge of the grid
            if (idx_right >= N_grid):
                idx_right = N_grid - 1

            # Compute exact location of first grid point within right wing cutoff (float in template grid units)
            k_ref_exact = (idx_right - core_loc) * R_nu

            # Round to find the nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)

            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[idx_right] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                       (dV_dnu_0[k_ref] * d_Delta_nu))

        # General case where grid points sample both the left and right wings
        else:

            # Find leftmost grid point within left wing cutoff
            if (left_cut_loc > 0.0):
                idx_left = int(left_cut_loc) + 1  # Round down then add 1 to find desired grid point
            else:  # Exception for lines near lower edge boundary
                idx_left = int(left_cut_loc - 1) + 1  # Round down then add 1 to find desired grid point

            # Cover edge case: the lowest idx_right is the left edge of the grid
            if (idx_left <= 0):
                idx_left = 0

            # Find first grid point within the right wing cutoff
            idx_right_start = int(core_loc) + 1  # Exact core (rounded down to grid) plus 1

            # Find rightmost grid point within right wing cutoff
            idx_right = int(right_cut_loc)  # Round down to find desired grid point

            # Cover edge case: the highest idx_right is the right edge of the grid
            if (idx_right >= N_grid):
                idx_right = N_grid - 1

            # cross-section calculation at left wing cutoff

            k_ref_exact = (core_loc - idx_left) * R_nu  # R_nu maps from output grid to template grid spacing

            # Round to find the nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)

            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[idx_left] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                      (dV_dnu_0[k_ref] * d_Delta_nu))

            # Proceed along the left wing towards core cutoff

            # Add cross-section contribution from the left wing
            for k in range(idx_left + 1, idx_right_start):
                # Increment k_ref_exact by the relative spacing between the k and k_ref grids
                k_ref_exact -= R_nu  # Stepping closer to the line core

                # Round to find the nearest point on template grid
                k_ref = int(k_ref_exact + 0.5)

                # Compute wavenumber difference between true wavenumber and closest template point
                d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

                # 1st order Taylor expansion in alpha and Delta_nu
                sigma[k] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                   (dV_dnu_0[k_ref] * d_Delta_nu))

            # Cross-section calculation at right wing cutoff

            # Reflect once crossed into the right wing
            k_ref_exact = abs(k_ref_exact - R_nu)

            # Round to find nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)

            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[idx_right_start] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) + (dV_dnu_0[k_ref] * d_Delta_nu))

            # Proceed along the right wing away from core cutoff

            # Add cross-section contribution from the right wing
            for k in range(idx_right_start + 1, idx_right + 1):  # +1 to include end index

                # Increment k_ref_exact by the relative spacing between the k and k_ref grids
                k_ref_exact += R_nu  # Stepping away from the line core

                # Round to find the nearest point on template grid
                k_ref = int(k_ref_exact + 0.5)

                # Compute wavenumber difference between true wavenumber and closest template point
                d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line

                # 1st order Taylor expansion in alpha and Delta_nu
                sigma[k] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                   (dV_dnu_0[k_ref] * d_Delta_nu))


def HITRAN_cross_section(linelist_files, input_dir, nu_grid, sigma, alpha_sampled, m, T, Q_T, Q_ref, J_max,
                         J_broad_all, N_voigt, cutoffs, Voigt_array, dV_da_array,
                         dV_dnu_array, dnu_Voigt, S_cut,verbose=False):
    """
    Calculate the cross-sections for a given linelist.

    Parameters:
    linelist_files (list):
        List of linelist files.
    input_dir (str):
        Directory containing the linelist files.
    nu_grid (np.ndarray):
        Frequency grid.
    sigma (np.ndarray):
        Cross section array to be filled.
    alpha_sampled (np.ndarray):
        Sampled broadening parameters.
    m (float):
        Mass of the molecule.
    T (float):
        Temperature.
    Q_T (float):
        Partition function at temperature T.
    Q_ref (float):
        Reference partition function.
    J_max (int):
        Maximum rotational quantum number.
    J_broad_all (np.ndarray):
        Broadening quantum numbers for all lines.
    N_voigt (int):
        Number of points in Voigt profile.
    cutoff_factor (float):
        Cutoff factor for line wings.
    Voigt_array (np.ndarray):
        Precomputed Voigt profile array.
    dV_da_array (np.ndarray):
        Derivative of Voigt profile w.r.t. a.
    dV_dnu_array (np.ndarray):
        Derivative of Voigt profile w.r.t. nu.
    dnu_Voigt (float):
        Frequency step in Voigt profile.
    S_cut (float):
        Line intensity cutoff.
    verbose (bool):
        Verbosity flag.

    Returns:
    np.ndarray: Calculated cross-section array.
    """

    # Set counters for processed transitions
    nu_0_total = 0

    for file in linelist_files:
        if verbose:
            print(f"Processing file: {file}")

    nu_min = nu_grid[0]
    nu_max = nu_grid[-1]
    dnu_out = nu_grid[1] - nu_grid[0]
    log_alpha_sampled = np.log10(alpha_sampled)

    time_begin = time.perf_counter()

    # Go through line_list files

    for n in range(len(linelist_files)):

        trans_file = h5py.File(input_dir + linelist_files[n], 'r')

        t_running = time.perf_counter()

        # Read in values from the provided linelist file

        nu_0_in = np.array(trans_file['Transition Wavenumber'])
        S_ref_in = np.power(10.0, np.array(trans_file['Log Line Intensity']))
        E_low_in = np.array(trans_file['Lower State E'])
        J_low_in = np.array(trans_file['Lower State J'])

        # Get rid of transitions not in the grid
        print("Filtering transitions within frequency grid.")
        nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        S_ref = S_ref_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        E_low = E_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        J_low = J_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]

        # Update total transitions counter
        nu_0_total += len(nu_0)

        # Ensure that when transitions are not in increasing order, they are arranged
        sort_index = np.argsort(nu_0)
        nu_0 = nu_0[sort_index]
        S_ref = S_ref[sort_index]
        E_low = E_low[sort_index]
        J_low = J_low[sort_index]

        # For J'' above the tabulated max, treat the broadening as that of J_max
        J_low[np.where(J_low > J_max)] = J_max

        # Compute the doppler broadening parameter for each line
        print("Calculating Doppler broadening parameters.")
        alpha = np.sqrt(2.0 * kb * T * np.log(2.0) / m ) * (nu_0 / c)
        log_alpha = np.log10(alpha)

        # Calculate line intensities at temperature T
        print("Calculating line intensities at temperature")
        S_T = calculate_line_intensity(S_ref, Q_T, Q_ref, T_ref, T, E_low, nu_0)

        # Make sure to consider for only lines above the intensity cutoff
        nu_0 = nu_0[np.where(S_T >= S_cut)]
        J_low = J_low[np.where(S_T >= S_cut)]
        S_T = S_T[np.where(S_T >= S_cut)]

        # Delete arrays to free up memory
        del nu_0_in, S_ref_in, S_ref, E_low_in, E_low, J_low_in

        # Continue if any transitions in the file satisfy the grid boundaries
        if len(nu_0) > 0:
            # Add contributions from the lines to the cross-section array (sigma)
            calculate_cross_section(sigma, nu_grid, nu_0, cutoffs, S_T, J_low, J_broad_all,
                                    alpha, log_alpha, alpha_sampled, log_alpha_sampled,
                                    N_voigt, Voigt_array, dV_da_array, dV_dnu_array,
                                    dnu_Voigt, dnu_out)

        t_end = time.perf_counter()
        total_running = t_end - t_running

        if verbose:
            print(f"Time taken for {linelist_files[n]}: {total_running:.2f} seconds")

        trans_file.close()

    t_end_calc = time.perf_counter()
    total_calc = t_end_calc - time_begin

    print("We have completed the calculation")
    print("Completed" +" "+str(nu_0_total)+" " + "transitions in " + str(total_calc)+" "+"seconds")



