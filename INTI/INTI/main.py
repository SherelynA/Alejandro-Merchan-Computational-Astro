# Necessary imports for the INTI framework
import os
import numpy as np
import pandas as pd
import re
import time
from scipy.interpolate import UnivariateSpline as Interp
import contextlib
with contextlib.redirect_stdout(None): # suppress HITRAN automatic print statement
    from hapi.hapi import molecularMass, moleculeName, isotopologueName


from INTI.constants import c, kb, u, P_ref, T_ref
from INTI.helper import write_output


import INTI.hitran as HITRAN
import INTI.helper as HELP
import INTI.voigt as VOIGT
import INTI.cross_section as calculate
import INTI.broadening as broadening

# The Pycharm Co-pilot was used to write the descriptions for the parameters and return values for the functions in this file.

def mass_from_hitran(species,isotopologue_id, linelist='hitran'):
    """
    Retrieve the molecular mass from the HITRAN database for a given molecule and isotopologue.

    Parameters:
    species (string):
        The chemical formula of the molecule (e.g., 'H2O', 'CO2').
    molecule_id (integer):
        The provided HITRAN molecule ID.
    isotopologue_id (integer):
        The provided HITRAN isotopologue ID. The most abundant isotopologue is 1.
    ___
    Description:
    This function retrieves the molecular mass of a specified molecule and its isotopologue
    from the HITRAN database. It also validates the provided molecule ID and isotopologue ID.
    ___
    EXAMPLE USAGE:
    mass = mass_from_hitran('H2O', 1, 1)
    Returns:
    float: Molecular mass in atomic mass units (amu).
    """
    try:

        if linelist == 'hitran':
            mol_ID = 1

            while moleculeName(mol_ID) != species:
                mol_ID += 1

            molecule_ID = int(mol_ID)
            isotope_ID = int(isotopologue_id)

        # Validating user provided molecule ID and species name match with HITRAN, to ensure correct mass is retrieved.
        if (molecule_ID < 1) or (molecule_ID > 61):
            # User provided invalid molecule ID for the HITRAN database
            raise ValueError("Molecule ID must be an integer between 1 and 61.")
        else:
            # User provided valid molecule ID
            if moleculeName(molecule_ID).upper() != species.upper():
                # User provided invalid species name for the given molecule ID
                raise ValueError(f"Molecule ID {molecule_ID} does not correspond to species {species}. Check HITRAN database for correct molecule ID or Species name.")
            else:
                # User provided valid species name and valid molecule ID.
                if (isotope_ID < 1) or (isotope_ID > 12) or (isotope_ID == 0):
                    if isotope_ID != 0:
                        # User provided invalid isotope ID
                        raise ValueError("Isotopologue ID must be between 1 and 12, or 0 (an allowable isotope for carbon dioxide (CO2)).")
                else:
                    # User provided valid molecule and isotope IDs
                    try:
                        isotope_name = isotopologueName(molecule_ID, isotope_ID)
                        # Clean isotope name from HITRAN formatting

                        return molecularMass(molecule_ID, isotope_ID)

                    except Exception as e:
                        # User provided valid molecule ID, but invalid isotope ID for that molecule
                        raise ValueError(f"Isotopologue ID {isotope_ID} is not valid for molecule ID {molecule_ID}. Check HITRAN database for correct isotopologue ID.") from e
    except Exception as e:
        # User provided molecule ID and or isotope ID, but they are not integers
        raise ValueError(f"Provided molecule ID is not an integer. HITRAN only accepts molecule IDs (1-61)") from e


def read_pf(input_dir):
    """
    Read in an external partition function file from the specified directory.

    Parameters:
        input_dir (string): Directory path where the partition function file is located.
    ___
    Description:
    This function reads a partition function file from the specified local directory.
    ___
    EXAMPLE USAGE:
    T_pf_raw, Q_raw = read_pf('data/molecule/input/')

    Returns:
    T_pf_raw: numpy array of temperature partition function data.
    Q_raw: numpy array of partition function values.
    """
    print("Reading in partition function file...")

    # Try and find partition funciton file that ends in '.pf'
    pf_file_name = [filename for filename in os.listdir(input_dir) if filename.endswith('.pf')]
    if not pf_file_name:
        raise FileNotFoundError("No partition function file with '.pf' extension found in the specified directory.")
    pf_file = pd.read_csv(input_dir + pf_file_name[0], sep= ' ', header=None, skiprows=1)

    # Extract temperature and partition function values
    T_pf_raw = pf_file[0].to_numpy(dtype=np.float64)
    Q_raw = pf_file[1].to_numpy(dtype=np.float64)

    return T_pf_raw, Q_raw


def interp_pf(T_pf_raw, Q_raw,T,T_ref):
    """
    Interpolate partition function values at a given temperature.

    Parameters:
    T_pf_raw (numpy array):
        array of temperature partition function data.
    Q_raw (numpy array):
        array of partition function values.
    T (float):
        temperature at which to interpolate the partition function.
    T_ref  (float):
        reference temperature for normalization (default is T_ref).

    Returns:
    Q_interp: float, interpolated partition function value at temperature T.
    ___
    Description:
    This function interpolates the partition function values at a specified temperature
    using spline interpolation.

    EXAMPLE USAGE:
    Q_T, Q_T_ref = interp_pf(T_pf_raw, Q_raw, 1500, T_ref=296)

    Returns:
    Q_T: float, partition function value at temperature T.
    Q_T_ref: float, partition function value at reference temperature T_ref.
    """
    # Create cubic spline interpolation of the partition function data
    spline = Interp(T_pf_raw, Q_raw, k=3)

    # Create a new temperature grid up to 10,000K
    T_pf_interp = np.linspace(1, 10000, 9999)

    # Evaluate the spline at the desired temperature
    Q_interp = spline(T_pf_interp)

    # Find index of the temperature closest to T chosen by user
    Temp_idx = np.argmin(np.abs(T_pf_interp - T))
    Temp_ref_idx = np.argmin(np.abs(T_pf_interp - T_ref))

    # Find partition function value at user specified temperature T
    Q_T = Q_interp[Temp_idx]
    Q_T_ref = Q_interp[Temp_ref_idx]

    return Q_T, Q_T_ref


def make_nu_grid(nu_min, nu_max, dnu_out):
    """
    Create a frequency grid based on specified minimum and maximum frequencies and resolution.

    Parameters:
    nu_min (float):
        minimum frequency (in cm^-1).
    nu_max (float):
        float, maximum frequency (in cm^-1).
    resolution (float):
        desired resolution (in cm^-1).
    ___
    Description:
    This function generates a frequency grid between the specified minimum and maximum
    frequencies with a given resolution.

    EXAMPLE USAGE:
    nu_grid = make_nu_grid(200, 2500, 0.01)

    Returns:
        nu_grid: numpy array of frequency grid points.

    """

    # Calculate the number of points needed for the specified resolution
    nu_min = min(1, nu_min)
    nu_max = nu_max + 1000

    num_points = int((nu_max - nu_min) / dnu_out + 1)

    # Generate the frequency grid
    nu_grid = np.linspace(nu_min, nu_max, num_points)

    return nu_grid


def call_inti(species='', isotopologue=1, ionization_state=1):
    """
    Main function is to call the downloaded files and retrieve data for the desired species.

    Parameters:
    species (string):
        The chemical formula of the hydrocarbon (e.g., 'CH4', 'C2H2', 'C2H6', 'C2H4', 'C4H2').
    isotopologue (integer, string):
        The HITRAN isotopologue ID (default is 1 for the most common isotopologue).
        The user can also input 'default' to automatically detect the most abundant isotopologue.
    line_list (string):
        The line list to in this case will be default because we are using HITRAN/HITEMP.
    ionization_state (integer):
        The ionization state of the species (default is 1 since we are dealing with neutral hydrocarbons).
    Returns:
       None.
    """
    hydrocarbons_chem = ['CH4', 'C2H2', 'C2H6', 'C2H4', 'C4H2']
    hydrocarbon_names = ['METHANE', 'ACETYLENE', 'ETHANE', 'ETHYLENE', 'DIACETYLENE']

    if species != '':
        user_prompt = False
    else:
        user_prompt = True

    if user_prompt:

        while True:
            species = input("Enter the chemical formula of the molecule (e.g., 'CH4', 'C2H2', 'C2H6', 'C2H4', 'C4H2')").strip()
            isotopologue = input("Enter the isotopologue ID (enter 'default' for the most abundant isotopologue):")
            print(isotopologue)

            # Validate User Input against supported hydrocarbons chemical formulas
            if species.upper() in hydrocarbons_chem:
                break
            # Validate User Input against supported hydrocarbons names if chemical formula is not provided
            elif species.upper() in hydrocarbon_names:
                index = hydrocarbon_names.index(species.title())
                species = hydrocarbons_chem[index]
                break
            else:
                # User has not provided a valid hydrocarbon chemical formula or name.
                print("Invalid input. Please enter a valid chemical formula or name for a hydrocarbon (e.g., 'CH4' or 'METHANE').")

        if isinstance(isotopologue, str):
            if isotopologue.lower() == 'default':
                iso = 1
            else:
                try:
                    iso = int(isotopologue)
                except ValueError:
                    raise ValueError("Isotopologue ID must be an integer like 1 for the most common isotope or 'default' for automatic detection.")

        spe = HITRAN.check(species, iso)
        ion = ionization_state

        print("\nValidating user provided inputs...")

        HITRAN.call_HITRAN(spe, iso)

        print(f" Species: {spe}, Isotopologue ID: {iso}, Ionization State: {ion}")

    if not user_prompt:
        # Validate User Input against supported hydrocarbons chemical formulas
        if species.upper() not in hydrocarbons_chem and species.upper() not in hydrocarbon_names:
            raise ValueError("Invalid species input. Please enter a valid chemical formula or name for a hydrocarbon (e.g., 'CH4' or 'METHANE').")

        iso = isotopologue
        ion = ionization_state
        species_name = species

        if isinstance(isotopologue, str):
            if isotopologue.lower() == 'default':
                iso = 1
            else:
                try:
                    iso = int(isotopologue)
                except ValueError:
                    raise ValueError("Isotopologue ID must be an integer like 1 for the most common isotope or 'default' for automatic detection.")

        print("\nValidating user provided inputs...")

        spe = HITRAN.check(species_name, iso)

        HITRAN.call_HITRAN(spe, iso)
        print(f" Species: {spe}, Isotopologue ID: {iso}, Ionization State: {ion}")
        print("\nLine list successfully retrieved.")


def calculate_inti(input_dir, species, temperature, pressure=None, log_pressure=None, database='hitran',
                   isotope=1, ionization_state=1, line_list='default', set_mass=None, nu_min=200, nu_max=25000,
                   dnu_out=0.01,broad_type='default', broadening_file='', X_H2=0.85, X_He=0.15, Voigt_cutoff=500,
                   Voigt_subspacing = (1.0/6.0), N_alpha_sample=500, S_cut = 1.0e-100, cut_max=100.0, N_cores=1, verbose=True):
    """
    Calculate the absorption cross-section using the INTI framework.
    Parameters:
    input_dir (string):
        Directory path where the input files are located.
    species (string):
        The chemical formula of the molecule (e.g., 'H2O', 'CO2').
    temperature (float):
        Temperature at which to calculate the cross-section (in Kelvin).
    pressure (float):
        Pressure at which to calculate the cross-section (in atm). Either pressure or log_pressure must be provided.
    log_pressure (float):
        Logarithm (base 10) of pressure at which to calculate the cross-section. Either pressure or log_pressure must be provided.
    isotope (integer):
        The HITRAN isotopologue ID (default is 1 for the most common isotopologue).
    ionization_state (integer):
        The ionization state of the species (default is 1). This won't affect neutral molecules.
    line_list (string):
        The line list to use (default is 'default' which uses HITRAN/HITEMP).
    set_mass (double, optional):
        If provided, use this mass (in amu) instead of retrieving from HITRAN.
    nu_min (double,optional):
        Minimum wavenumber for cross-section calculation (in cm^-1).
    nu_max (double,optional):
        Maximum wavenumber for cross-section calculation (in cm^-1).
    dnu_out (double, optional):
        Desired resolution for the frequency grid (in cm^-1).
    broad_type (string,optional):
        The type of broadening to use ('default', 'H2-He', 'air').
    broadening_file (string, optional):
        Path to a custom broadening file (if not using default).
    X_H2 (double, optional):
        Mole fraction of H2 for H2-He broadening (default is 0.85).
    X_He (double, optional):
        Mole fraction of He for H2-He broadening (default is 0.15).
    Voigt_cutoff (double, optional):
        Cutoff multiplier for Voigt profile (default is 500).
    Voigt_subspacing (double, optional):
        Sub-spacing factor for Voigt profile (default is 1/6).
    N_alpha_sample (integer, optional):
        Number of alpha samples for Voigt profile precomputation (default is 500).
    S_cut (double, optional):
        Minimum line strength cutoff for including lines in the calculation (default is 1.0e-100).
    cut_max (double, optional):
        Maximum cutoff value for line profiles (default is 30.0 cm^-1).
    N_cores (integer, optional):
        Number of CPU cores to use for parallel processing (default is 1).
    verbose (boolean, optional):
        If True, print detailed progress information (default is True).
    ___
    Description:
    This function serves as the main interface to calculate absorption cross-sections.
    It integrates various components of the INTI framework to perform the calculations
    based on user-defined parameters.

    EXAMPLE USAGE:
    cross_section = calculate_inti(input_dir='data/', species='H2O', temperature=1500,
                                   pressure=1.0, isotope=1, line_list='HITEMP',
                                   nu_min=200, nu_max=2500, dnu_out=0.01)
    Returns:
    nu_out (double):
        wavenumber
    sigma_out (double):
        cross-section (cm^2/molecule)
    """

    print('INTI is shining on you! Beginning calculations for cross-sections')

    # Keep track of time taken for calculation
    start_time = time.perf_counter()

    # Validate User Inputs
    if pressure is None and log_pressure is None:
        raise ValueError("Either a value for pressure or log_pressure must be provided in units of bar.")


    if log_pressure is not None:
        log_pressure = HELP.require_scalar(log_pressure, "log_pressure")
        pressure = 10.0 ** log_pressure
    else:
        pressure = HELP.require_scalar(pressure, "pressure")
        log_pressure = np.log10(pressure)

    temperature = HELP.require_scalar(temperature, "temperature")

    input_directory = HELP.find_input_directory(input_dir, species, isotope, line_list)

    line_list, isotopologue = HELP.parse_directory(input_directory)


    linelist_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.h5')]

    print("Found line list files, and loading HITRAN format")

    # Call and read in the partition function
    T_pf_raw, Q_raw = read_pf(input_directory)

    # Figure out the mass of the species
    if (set_mass is None):
        mass = mass_from_hitran(species, isotope) * u
    else:
        mass = set_mass * u

    is_molecule = True

    if is_molecule and broad_type == 'default':

        broad_type = broadening.determine_broad(input_directory)

        if broad_type == 'H2-He':
            J_max, J_broad_all, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = broadening.read_H2_He(input_directory)

        elif broad_type == 'air':
            J_max, J_broad_all, gamma_0_air, n_L_air = broadening.read_air(input_directory)

    elif is_molecule and broad_type != 'default':

        if (broad_type == 'H2-He' and 'H2.broad' in os.listdir(input_directory)
                and 'He.broad' in os.listdir(input_directory)):
            J_max, J_broad_all, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = broadening.read_H2_He(input_directory)

        elif broad_type == 'air' and 'air.broad' in os.listdir(input_directory):
            J_max, J_broad_all, gamma_0_air, n_L_air = broadening.read_air(input_directory)

        else:
            raise ValueError("Broadening type specified does not match available broadening files in the input directory.")

    # Prepare pressure and temperature
    P = pressure
    T = temperature

    # Interpolate partition function at temperature T
    Q_T, Q_T_ref = interp_pf(T_pf_raw, Q_raw, T, T_ref)

    if is_molecule:

        # Compute Lorentzian broadening parameters if needed
        if broad_type == 'H2-He':
            gamma = broadening.compute_H2_He(gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2, gamma_0_He, n_L_He, X_He)
        elif broad_type == 'air':
            gamma = broadening.compute_air(gamma_0_air, T_ref, T, n_L_air, P, P_ref)

        # Create frequency grid if needed
        nu_compute = make_nu_grid(nu_min, nu_max, dnu_out)

        #make cross-section output arrays

        sigma_compute = np.zeros(len(nu_compute))

        print("Pre-computing Voigt profiles...")

        t1 = time.perf_counter()

        (nu_sampled, alpha_sampled,
         cutoffs, N_Voigt, Voigt_arr,
         dV_da_arr, dV_dnu_arr,
         dnu_Voigt) = VOIGT.precompute_molecules(nu_compute, dnu_out, mass, T,
                                                 Voigt_subspacing, Voigt_cutoff,
                                                 N_alpha_sample, gamma, cut_max)

        t2 = time.perf_counter()
        time_precompute = t2 - t1

        print('Voigt profiles computed in ' + str(time_precompute) + ' s')

        if isotopologue == 1:
            label = species
        else:
            label = species +' ('+isotopologue+')'

        print('Generating cross section for ' + label + ' at P = ' + str(P) + ' bar, T = ' + str(T) + ' K')

    if database == 'hitran':
        calculate.HITRAN_cross_section(linelist_files, input_directory,
                                       nu_compute, sigma_compute, alpha_sampled, mass,
                                       T, Q_T, Q_T_ref, J_max, J_broad_all,
                                       N_Voigt, cutoffs, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt,
                                       S_cut, verbose)

    nu_out = nu_compute[(nu_compute >= nu_min) & (nu_compute <= nu_max)]
    sigma_out = sigma_compute[(nu_compute >= nu_min) & (nu_compute <= nu_max)]

    output_filename = re.sub('/input/', '/output/', input_directory)

    if not os.path.exists(output_filename):
        os.makedirs(output_filename)

    write_output(output_filename, species, T, np.log10(P), broad_type, broadening_file, nu_out, sigma_out)

    time_final = time.perf_counter()
    total_time = time_final - start_time

    print('\nTotal time for INTI calculation: ' + str(total_time) + ' s')