import os
import re
import shutil
import time
import h5py
import pandas as pd
import numpy as np
import contextlib
with contextlib.redirect_stdout(None):  # suppress HITRAN automatic print statement
    from hapi.hapi import db_begin, fetch, abundance, moleculeName, isotopologueName
from numbers import Number
import INTI.hitran as HITRAN

# The Pycharm Co-pilot was used to write the descriptions for the parameters and return values for the functions in this file.


def require_scalar(value, name):
    if isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(
            f"{name} must be a single scalar value, not {type(value).__name__}"
        )
    if not isinstance(value, Number):
        raise TypeError(
            f"{name} must be a numeric scalar"
        )
    return float(value)


def write_output(output_directory, species, T, log_P,
                 broad_type, broadening_file, nu_out, sigma_out):
    """
    Parameters
    ----------
    output_directory : String
        Local directory where the output data is to be stored.
    species : String
        Name of molecule or atomic.
    T : int
        Temperature (K) the cross-section was computed at.
    log_P : int
        DESCRIPTION.
    broad_type : String
        The type of broadening used in computing the cross-section.
    nu_out : TYPE
        DESCRIPTION.
    sigma_out : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Add ionisation state for atoms

    if broadening_file == '':
        f = open(
            (output_directory + species + '_T' + str(T) + 'K_log_P' + str(log_P) + '_' + broad_type + '_sigma.txt'),
            'w')
    else:
        f = open((output_directory + species + '_T' + str(T) + 'K_log_P' + str(log_P) + '_' + broad_type +
                  '_' + os.path.splitext(broadening_file)[0] + '_sigma.txt'), 'w')

    for i in range(len(nu_out)):
        f.write('%.8f %.8e \n' % (nu_out[i], sigma_out[i]))

    f.close()


def process_hdf_chunk(chunk, upper_ds, lower_ds, logA_ds, total_written):
    ''' Process chunk of data from convert_to_hdf

    Parameters
    ----------
    chunk : _type_
        _description_
    upper_ds : _type_
        _description_
    lower_ds : _type_
        _description_
    logA_ds : _type_
        _description_
    total_written : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    data = np.array(chunk, dtype=np.float64)

    upper_state = data[:, 0].astype(np.uint32)
    lower_state = data[:, 1].astype(np.uint32)
    log_Einstein_A = np.log10(data[:, 2] + 1e-250).astype(np.float32)

    n_new = len(upper_state)
    new_size = total_written + n_new

    # Extend datasets
    upper_ds.resize((new_size,))
    lower_ds.resize((new_size,))
    logA_ds.resize((new_size,))

    # Write chunk
    upper_ds[total_written:new_size] = upper_state
    lower_ds[total_written:new_size] = lower_state
    logA_ds[total_written:new_size] = log_Einstein_A

    return new_size


def convert_to_hdf(file='', mol_ID=0, iso_ID=0, database=''):
    '''
    Convert a given file to HDF5 format.

    Parameters
    ----------
    file : String, optional
        File name. The default is ''.
    mol_ID : int, optional
        HITRAN/HITEMP molecule ID. The default is 0.
    iso_ID : int, optional
        HITRAN/HITEMP isotopologue ID. The default is 0.
    alkali : bool, optional
        Whether or not the species is an alkali metal. The default is False.
    database : String, optional
        Database that the line list came from. The default is ''.
    chunk_size : int, optional
        The size of the chunks to be read and written into the hdf5 file. The default is 5 million.
    compression_type : String, optional
        The compression algorithm to be used during the writing to hdf5. The default is 'lzf', due to its speed.

    Returns
    -------
    None.

    '''

    start_time = time.time()

    if (database in ['HITRAN']):  # Read file downloaded from HITRAN/HITEMP, keep relevant data,
        # and store data in a new HDF5 file

        # Different HITRAN2020 formats for different molecules leads us to read in .par files with different field widths
        # See https://hitran.org/media/refs/HITRAN_QN_formats.pdf

        # Group 1
        if mol_ID in {1, 3, 9, 10, 12, 20, 21, 25, 29, 31, 32, 33, 35, 37, 38, 49}:
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 5, 1, 6, 12, 1, 7, 7]
            J_col = 13
            if mol_ID in {10, 33}:  # Handle NO2 and HO2 J_cols separately, since HITRAN provides N instead of J
                Sym_col = 17

        # Group 2
        elif mol_ID in {2, 4, 5, 14, 15, 16, 17, 19, 22, 23, 26, 36, 44, 45, 46, 48, 53, 43}:
            if mol_ID == 43:  # C4H2 format exception
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 2, 2, 2, 1, 1, 3, 1, 1, 6, 12, 1, 7,
                                 7]
                J_col = 19
            else:
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 5, 1, 3, 1, 5, 6, 12, 1, 7, 7]
                J_col = 15

        # Group 3
        elif (mol_ID == 6 and iso_ID in {1, 2}) or mol_ID in {30, 42, 52}:
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 3, 2, 3, 5, 6, 12, 1, 7, 7]
            J_col = 14

        # Group 4
        elif mol_ID in {11, 24, 27, 28, 39, 40, 41, 51, 54, 55} or (mol_ID == 6 and iso_ID in {3, 4}):
            if mol_ID == 11 and iso_ID == 1:  # 14-NH3 format exception
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 3, 2, 1, 3, 3, 1, 6, 12, 1, 7, 7]
                J_col = 13
            elif mol_ID == 27:  # C2H6 format exception
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 2, 3, 4, 6, 12, 1, 7, 7]
                J_col = 13
            else:
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 2, 2, 1, 4, 6, 12, 1, 7, 7]
                J_col = 13

        # Group 5
        elif mol_ID == 47:
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 2, 3, 1, 6, 12, 1, 7, 7]
            J_col = 14

        # Group 6
        elif mol_ID in {7, 50}:
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 1, 1, 3, 1, 3, 5, 1, 6, 12, 1, 7, 7]
            J_col = 17

        # Group 7
        elif mol_ID in {8, 18, 13}:
            if mol_ID in {8, 18}:  # NO and ClO formats
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 2, 5, 1, 5, 6, 12, 1, 7, 7]
                J_col = 15
            elif mol_ID in {13}:  # OH format
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 1, 2, 5, 2, 5, 6, 12, 1, 7, 7]
                J_col = 15

        else:
            raise Exception("Error: the molecule symmetry for HITRAN ID " + str(
                mol_ID) + " is not currently implemented in Cthulhu")

        trans_file = pd.read_fwf(file, widths=field_lengths, header=None)

        # Get only the necessary columns from the .par file (trans_file can be thought of as par_file, the var name was kept for convenience)
        nu_0 = np.array(trans_file[2])
        log_S_ref = np.log10(np.array(trans_file[3]) / abundance(mol_ID, iso_ID))
        gamma_L_0_air = np.array(trans_file[5]) / 1.01325  # Convert from cm^-1 / atm -> cm^-1 / bar
        E_lower = np.array(trans_file[7])
        n_L_air = np.array(trans_file[8])
        J_lower = np.array(trans_file[J_col]).astype(np.float64)

        if mol_ID in {10,
                      33}:  # Handle creation of NO2 and HO2 J_lower columns, as the given value is N on HITRAN not J
            Sym = np.array(trans_file[Sym_col])
            for i in range(len(J_lower)):
                if Sym[i] == '+':
                    J_lower[i] += 0.5
                else:
                    J_lower[i] -= 0.5

        hdf_file_path = os.path.splitext(file)[0] + '.h5'

        # Write the data to our HDF5 file
        with h5py.File(hdf_file_path, 'w') as hdf:
            hdf.create_dataset('Transition Wavenumber', data=nu_0, dtype='f4')  # store as 32-bit unsigned float
            hdf.create_dataset('Log Line Intensity', data=log_S_ref, dtype='f4')
            hdf.create_dataset('Lower State E', data=E_lower, dtype='f4')
            hdf.create_dataset('Lower State J', data=J_lower, dtype='f4')
            hdf.create_dataset('Air Broadened Width', data=gamma_L_0_air, dtype='f4')
            hdf.create_dataset('Temperature Dependence of Air Broadening', data=n_L_air, dtype='f4')

        os.remove(file)

    print("This file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")


def download_HITRAN_line_list(mol_ID, iso_ID, folder, nu_min=1, nu_max=100000):
    """
    Download line list using the fetch() function already in HITRAN.

    Parameters
    ----------
    mol_ID : int
        HITRAN molecular ID.
    iso_ID : int
        HITRAN isotopologue ID.
    folder : String
        Local directory where the line list is to be stored.
    nu_min : int, optional
        Minimum wavenumber for which the line list is downloaded. The default is 1.
    nu_max : int, optional
        Maximum wavenumber for which the line list is downloaded. The default is 100,000.

    Returns
    -------
    None.

    """

    db_begin(folder)
    fetch(moleculeName(mol_ID), mol_ID, iso_ID, nu_min, nu_max)

    print("\nLine list downloaded. Converting file to HDF to save storage space...")

    for file in os.listdir(folder):
        if file.endswith('.data'):
            convert_to_hdf(file=(folder + file), mol_ID=mol_ID,
                           iso_ID=iso_ID, database='HITRAN')


def create_directories(molecule='', isotopologue='', line_list='', database='HITRAN',
                       mol_ID=0, iso_ID=0, ionization_state=1, VALD_data_dir=''):
    '''
    Create new folders on local machine to store the relevant data

    Parameters
    ----------
    molecule : String, optional
        Molecule name. The default is ''.
    isotopologue : String, optional
        Isotopologue name. The default is ''.
    line_list : String, optional
        Species line list. For HITRAN, HITEMP, and VALD, the line list is the same as the database. The default is ''.
    database : String, optional
        Database the line list was derived from. The default is ''.
    mol_ID : int, optional
        Molecular ID number as specified on HITRAN / HITEMP. The default is 0.
    iso_ID : int, optional
        Isotopologue ID number as specified on HITRAN / HITEMP. The default is 0.
    ionization_state : int, optional
        Ionization state of atomic species. The default is 1.
    VALD_data_dir : String, optional
        Local directory VALD line list is stored in. The default is ''.

    Returns
    -------
    line_list_folder : String
        Local directory containing the line list.

    '''

    input_folder = './input'

    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)


    if (database in ['HITRAN']):

        molecule_folder = input_folder + '/' + moleculeName(mol_ID) + '  ~  '

        iso = isotopologueName(mol_ID, iso_ID)  # Will need to format the isotopologue name to match ExoMol formatting

        iso_name = HITRAN.clean_isotope_name(iso)

        molecule_folder += iso_name

        line_list_folder = molecule_folder + '/' + database + '/'

        if os.path.exists(molecule_folder) == False:
            os.mkdir(molecule_folder)

        # If we don't remove an existing HITRAN folder, we encounter a Lonely Header exception from hapi.py
        if (database == 'HITRAN'):
            if os.path.exists(line_list_folder):
                shutil.rmtree(line_list_folder)
            os.mkdir(line_list_folder)

        else:
            if os.path.exists(line_list_folder) == False:
                os.mkdir(line_list_folder)

    return line_list_folder


def find_input_directory(file_path, molecule, isotope, linelist):
    """
    Constructs the input directory path based on the provided parameters.
    Parameters:
    file_path (str): Base file path.
    molecule (str): Name of the molecule.
    isotope (str): Isotope of the molecule.
    linelist (str): Name of the linelist.
    """

    molecule_dictionary = HITRAN.create_id_dict()
    molecule_id = molecule_dictionary.get(molecule)

    if isotope == 1:
        isotope = isotopologueName(molecule_id, 1)
    else:
        isotope = isotopologueName(molecule_id, isotope)
    isotope = HITRAN.clean_isotope_name(isotope)

    if linelist =='default':
        linelist = 'HITRAN'

    tag = isotope

    input_directory = (file_path + molecule + '  ~  ' + tag + '/' +
                       linelist + '/')

    if os.path.exists(input_directory):
        return input_directory
    else:
        raise FileNotFoundError(f"Input directory {input_directory} does not exist.")


def parse_directory(directory):
    """
    Determine which linelist and isotopologue this directory contains data for (assumes data was downloaded using our script)

    Parameters
    ----------
    directory : String
        Local directory containing the line list file[s], broadening data, and partition function.
    database : String
        Database line list is derived from.

    Returns
    -------
    linelist : String
        Line list for which the cross-section is to be calculated.
    isotopologue : String
        Molecular isotopologue for which the cross-section is to be calculated.

    """

    directory_name = os.path.abspath(directory)
    linelist = os.path.basename(directory_name)

    directory_name = os.path.dirname(directory_name)

    molecule = os.path.basename(directory_name)
    isotopologue = re.sub('.+[  ~]', '', molecule)  # Keep isotope part of the folder name

    return linelist, isotopologue
