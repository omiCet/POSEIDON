''' 
Functions to interpolate chemical composition grids.

'''

import os
import h5py
import numpy as np
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
import re

from .utility import shared_memory_array
from .supported_chemicals import supported_species, fastchem_supported_species, vulcan_supported_species, inactive_species

vulcan_grid_list = ["VULCAN_test", "VULCAN_Grid1.1"] #allowed list of vulcan grids; files in inputs with name GRID_database.hdf5. to be editable by user


def load_chemistry_grid(chemical_species, grid = 'fastchem', 
                        comm = MPI.COMM_WORLD, rank = 0):
    '''
    Load a chemical abundance grid.

    Args:
        chemical_species (list or np.array of str):
            List of chemical species to load mixing ratios from grid.
        grid (str):
            Name of the pre-computed chemical abundance grid. The file should be
            located in the POSEIDON input directory (specified in your .bashrc
            file) with a name format like 'GRID_database.hdf5' 
            (e.g. 'fastchem_database.hdf5'). By default, POSEIDON ships with
            an equilibrium chemistry grid computed from the fastchem code:
            https://github.com/exoclime/FastChem
            (Options: fastchem).
        comm (MPI communicator):
            Communicator used to allocate shared memory on multiple cores.
        rank (MPI rank):
            Rank used to allocate shared memory on multiple cores.

    Returns:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
    
    '''

    if (rank == 0):
        print("Reading in database for equilibrium chemistry model...")

    # Check that the selected chemistry grid is supported
    if (grid not in ['fastchem']):
        raise Exception("Error: unsupported chemistry grid")

    # Find the directory where the user downloaded the input grid
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")

    # Load list of chemical species supported by both the fastchem grid and POSEIDON
    supported_chem_eq_species = np.intersect1d(supported_species, 
                                                fastchem_supported_species)
        
    # If chemical_species = ['all'] then default to all species
    if ('all' in chemical_species):
        chemical_species = supported_chem_eq_species

    # Check all user-specified species are compatible with the fastchem grid
    else:
        if (np.any(~np.isin(chemical_species, supported_chem_eq_species)) == True):
            raise Exception("A chemical species you selected is not supported " +
                            "for equilibrium chemistry models.\n")
            
    # Open chemistry grid HDF5 file
    database = h5py.File(input_file_path + '/chemistry_grids/' + grid + '_database.hdf5', 'r')

    # Load the dimensions of the grid
    T_grid = np.array(database['Info/T grid'])
    P_grid = np.array(database['Info/P grid'])
    Met_grid = np.array(database['Info/M/H grid'])
    C_to_O_grid = np.array(database['Info/C/O grid'])

    # Find sizes of each dimension
    T_num, P_num, \
    Met_num, C_O_num = len(T_grid), len(P_grid), len(Met_grid), len(C_to_O_grid)

    # Store number of chemical species
    N_species = len(chemical_species)

    # Create array to store the log mixing ratios from the grid 
    log_X_grid, _ = shared_memory_array(rank, comm, (N_species, Met_num, C_O_num, T_num, P_num))
    
    # Only first core needs to load the mixing ratios into shared memory
    if (rank == 0):

        # Add each chemical species to mixing ratio array
        for q, species in enumerate(chemical_species):

            # Load grid for species q, then reshape into a 4D numpy array
            array = np.array(database[species+'/log(X)'])
            array = array.reshape(Met_num, C_O_num, T_num, P_num)

            # Package grid for species q into combined array
            log_X_grid[q,:,:,:,:] = array

    # Close HDF5 file
    database.close()
        
    # Force secondary processors to wait for the primary to finish
    comm.Barrier()

    # Package atmosphere properties
    chemistry_grid = {'grid': grid, 'log_X_grid': log_X_grid, 'T_grid': T_grid, 
                      'P_grid': P_grid, 'Met_grid': Met_grid, 'C_to_O_grid': C_to_O_grid,
                     }

    return chemistry_grid


def interpolate_log_X_grid(chemistry_grid, log_P, T, C_to_O, log_Met, 
                           chemical_species, return_dict = True):
    '''
    Interpolate a pre-computed grid of chemical abundances (e.g. an equilibrium
    chemistry grid) onto a model P-T profile, metallicity, and C/O ratio.

    Args:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
        log_P (float or np.array of float): 
            Pressure profile provided by the user (in log scale and in bar).
            A single value will be expanded into an array np.full(length, P), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            10^{-7} to 10^{2} bar are supported.
        T (float or np.array of float):
            Temperature profile provided by the user (K).
            A single value will be expanded into an array np.full(length, T), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            300 to 4000 K are supported.
        C_to_O (float or np.array of float):
            Carbon to Oxygen (C/O) ratio provided by the user.
            A single value will be expanded into an array np.full(length, C_O), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            0.2 to 2 are supported.
        log_Met (float or np.array of float):
            Planetary metallicity (in log scale. 0 represents 1x solar).
            A single value will be expanded into an array np.full(length, Met), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            -1 to 4 are supported.
        chemical_species (str or np.array of str):
            List of chemical species to interpolate mixing ratios for.
        return_dict (bool):
            If False, return an array of shape (len(species), len(P_array)).

    Returns:
        log_X_interp_dict (dict) ---> if return_dict = True:
            A dictionary of log mixing ratios with keys being the same names as 
            specified in chemical_species.

        log_X_interp_array (np.array of float) ---> if return_dict=False:
            An array containing the log mixing ratios for the species specified
            in chemical_species.
    
    '''

    # Unpack chemistry grid properties
    grid = chemistry_grid['grid']
    log_X_grid = chemistry_grid['log_X_grid']
    T_grid = chemistry_grid['T_grid']
    P_grid = chemistry_grid['P_grid']
    Met_grid = chemistry_grid['Met_grid']
    C_to_O_grid = chemistry_grid['C_to_O_grid']

    # Store lengths of input P, T, C/O and metallicity arrays
    len_P, len_T, \
    len_C_to_O, len_Met = np.array(log_P).size, np.array(T).size, \
                          np.array(C_to_O).size, np.array(log_Met).size
    max_len = max(len_P, len_T, len_C_to_O, len_Met)

    np.seterr(divide = 'ignore')

    # Check that the chemical species we want to interpolate are supported
    if (grid == 'fastchem'):
        supported_species = fastchem_supported_species
    else:
        raise Exception("Error: unsupported chemistry grid")
    if isinstance(chemical_species, str):
        if chemical_species not in supported_species: 
            raise Exception(chemical_species + " is not supported by the equilibrium grid.")
    else:
        for species in chemical_species:
            if species not in supported_species: 
                raise Exception(species + " is not supported by the equilibrium grid.")

    # Check that the desired pressures, temperatures, C/O and metallicity fall within the grid
    def not_valid(params, grid, is_log):
        if is_log:
            return (10**np.max(params) < grid[0]) or (10**np.min(params) > grid[-1])
        else:
            return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(log_P, P_grid, True):
        raise Exception("Requested pressure is out of the grid bounds.")
    if not_valid(T, T_grid, False):
        raise Exception("Requested temperature is out of the grid bounds.")
    if not_valid(C_to_O, C_to_O_grid, False):
        raise Exception("Requested C/O is out of the grid bounds.")
    if not_valid(log_Met, Met_grid, True):
        raise Exception("Requested M/H is out of the grid bounds.")
    
    # For POSEIDON's standard 3D temperature field
    if (len(T.shape) == 3):

        # Check validity of input array shapes
        T_shape = np.array(T).shape
        assert len_C_to_O == 1                # C_O should be a single value
        assert len_Met == 1                   # log_Met should be a single value
        assert len(log_P.shape) == 1          # log_P should be a 1D array
        assert log_P.shape[0] == T_shape[0]   # Size of log_P should match first dimension of T
        
        reps = np.array(T_shape[1:])
        reps = np.insert(reps, 0, 1)
        log_P = log_P.reshape(-1, 1, 1)
        log_P = np.tile(log_P, reps) # 1+T_shape[1:] is supposed to be (1, a, b) where T_shape[1:] = (a,b) is the second and third dimension of T. log_P should have the same dimension as T: (len(P), a, b)
        C_to_O = np.full(T_shape, C_to_O)
        log_Met = np.full(T_shape, log_Met)

    # For either a single (P, T, Met, C_to_O) or arrays
    else:
        if not (len_P in (1, max_len) and len_T in (1, max_len) and len_C_to_O in (1, max_len) and len_Met in (1, max_len)):
            raise Exception("Input shape not accepted. The lengths must either be the same or 1.")

        if len_P == 1:
            log_P = np.full(max_len, log_P)
        if len_T == 1:
            T = np.full(max_len, T)
        if len_C_to_O == 1:
            C_to_O = np.full(max_len, C_to_O)
        if len_Met == 1:
            log_Met = np.full(max_len, log_Met)

    # Interpolate mixing ratios from grid onto P-T profile, metallicity, and C/O of the atmosphere
    def interpolate(species):

        # Find index of the species
        q = np.where(chemical_species == species)[0][0]

        # Create interpolator object
        grid_interp = RegularGridInterpolator((np.log10(Met_grid), C_to_O_grid, T_grid, 
                                              np.log10(P_grid)), log_X_grid[q,:,:,:,:])
        
        return grid_interp(np.vstack((np.expand_dims(log_Met, 0), np.expand_dims(C_to_O, 0), 
                                      np.expand_dims(T, 0), np.expand_dims(log_P, 0))).T).T
    
    # Returning an array (default) 
    if not return_dict:
        if isinstance(chemical_species, str):
            return interpolate(chemical_species)
        log_X_list = []
        for _, species in enumerate(chemical_species):
            log_X_list.append(interpolate(species))
        log_X_interp_array = np.array(log_X_list)
        return log_X_interp_array
    
    # Returning a dictionary
    else:
        log_X_interp_dict = {}
        if isinstance(chemical_species, str):
            log_X_interp_dict[chemical_species] = interpolate(chemical_species)
            return log_X_interp_dict
        for _, species in enumerate(chemical_species):
            log_X_interp_dict[species] = interpolate(species)
        return log_X_interp_dict


def load_vulcan_chemistry_grid(chemical_species, grid = '', 
                        comm = MPI.COMM_WORLD, rank = 0):
    '''
    Load a chemical mixing profile grid computed with VULCAN.

    Args:
        chemical_species (list or np.array of str):
            List of chemical species to load mixing ratios from grid.
        grid (str):
            Name of the pre-computed chemical abundance grid. The file should be
            located in the POSEIDON input directory (specified in your .bashrc
            file) with a name format like 'GRID_database.hdf5' 
            (e.g. 'vulcan_database.hdf5'). 
        comm (MPI communicator):
            Communicator used to allocate shared memory on multiple cores.
        rank (MPI rank):
            Rank used to allocate shared memory on multiple cores.

    Returns:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database, as well
            as associated information (pressures and temperature-pressure 
            profile).
            
    '''

    # Add ability to add other grids to the allowed_list at some point
    if not grid in vulcan_grid_list:
        raise Exception("Error: This VULCAN grid is currently not supported.")
    
    else:
        if (rank == 0):
            print("Reading in database for VULCAN model...")

        # Find the directory where the user downloaded the input grid
        input_file_path = os.environ.get("POSEIDON_input_data")

        if input_file_path == None:
            raise Exception("POSEIDON cannot locate the input folder.\n" +
                            "Please set the 'POSEIDON_input_data' variable in " +
                            "your .bashrc or .bash_profile to point to the " +
                            "POSEIDON input folder.")

        # Load list of chemical species supported by both VULCAN and POSEIDON
        intersection_supported_species = np.intersect1d(np.append(supported_species,inactive_species), 
                                                    vulcan_supported_species)
            
        # If chemical_species = ['all'] then default to all species
        if ('all' in chemical_species):
            chemical_species = intersection_supported_species

        # Check all user-specified species are compatible with the VULCAN grid
        else:
            if (np.any(~np.isin(chemical_species, intersection_supported_species)) == True):
                raise Exception("A chemical species you selected is not supported " +
                                "in VULCAN or POSEIDON.\n")
                
        # Open chemistry grid HDF5 file
        database = h5py.File(input_file_path + '/chemistry_grids/' + grid + '_database.hdf5', 'r')

        # Determine the axes of the grid and load in the corresponding dimensions
        property_names = np.array(re.findall(r"[_\w]+", database['Misc_info/property_names'].asstr()[0]))
        grid_lists = np.array([database[f'Dimensions/{prop}_list'][...] for prop in property_names], dtype=np.ndarray)

        # Find sizes of each dimension
        dim_sizes = np.array([len(list) for list in grid_lists])

        # Load other info
        pressures = np.array(database['Misc_info/pressures'])
        P_num = len(pressures)
        temp_profiles = np.array(database['Misc_info/temp_profiles'])
        conv_flags = np.array(database['Misc_info/conv_flags'])
        conv_flags = conv_flags.reshape(dim_sizes)

        # Check that all dimensions are strictly increasing
        for list in grid_lists:
            if not np.all(list[:-1] <= list[1:]):
                raise Exception("Error: values along each dimension of the grid must be provided in strictly increasing order")
        # Check that pressure is strictly decreasing
        if not np.all(pressures[:-1] >= pressures[1:]):
                raise Exception("Error: pressures must be provided in strictly decreasing order")

        # Store number of chemical species
        N_species = len(chemical_species)

        # Create array to store the log mixing ratios from the grid 
        log_X_grid, _ = shared_memory_array(rank, comm, (N_species, *dim_sizes, P_num))
        
        # Only first core needs to load the mixing ratios into shared memory
        if (rank == 0):

            # Add each chemical species to mixing ratio array
            for q, species in enumerate(chemical_species):

                # Load grid for species q, then reshape into a 4D numpy array
                array = np.array(database[species+'/log(X)']) #database[species+'/log(X)'] is a 2D array; axis 0 = runs, axis 1 = pressure
                array = array.reshape(*dim_sizes, P_num)

                # Package grid for species q into combined array
                log_X_grid[q,...] = array

        #replace all negative infinities (i.e. 0 mixing ratio) with 1E-100
        def remove_infinities(log_X_grid):
            log_X_grid[log_X_grid == -np.inf] = -100
            return log_X_grid
        log_X_grid = remove_infinities(log_X_grid)

        # Close HDF5 file
        database.close()
            
        # Force secondary processors to wait for the primary to finish
        comm.Barrier()

        # Package atmosphere properties
        chemistry_grid = {'grid': grid, 'log_X_grid': log_X_grid, 'pressures': pressures, 'temp_profiles': temp_profiles, 'conv_flags':
                        conv_flags, 'species': chemical_species, 'property_names': property_names, 'grid_lists': grid_lists}

        return chemistry_grid


def interpolate_vulcan_log_X_grid(chemistry_grid, param_names, param_values, log_P, chemical_species, return_dict = True, use_conv_flag = True):
    '''
    Interpolate a pre-computed grid of VULCAN chemical mixing profiles onto a particular point in the parameter space. 
    Then interpolates the mixing profiles onto the provided pressures.

    Args:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
        param_names (list of str):
            The names of the free parameters in the grid.
        param_values (list of float, or list of np.array of float):
            The values of the free parameters at which evaluation is desired. If a list element is
            only a single float of value x, then it will be expanded into an array np.full(length, x)
            where length = max(len(parameter_1), len(parameter_2), ...).
        log_P (np.array of float):
            log10 of the pressures in bar. 
        chemical_species (str or np.array of str):
            List of chemical species to interpolate mixing ratios for.
        return_dict (bool):
            If False, return an array of shape (len(species), len(P_array)).
        use_conv_flag (bool):
            If True, interpolates over a flag which indicates whether the runs in the VULCAN grid converged/
            did not converge.

    Returns:
        log_X_interp_dict (dict) ---> if return_dict = True:
            A dictionary of log mixing ratios with keys being the same names as 
            specified in chemical_species.

        log_X_interp_array (np.array of float) ---> if return_dict=False:
            An array containing the log mixing ratios for the species specified
            in chemical_species.

        conv_flag (bool):
            Interpolated VULCAN convergence flag. If use_conv_flag is False, returns None instead.
    
    '''

    # Unpack chemistry grid properties
    grid = chemistry_grid['grid']
    log_X_grid = chemistry_grid['log_X_grid']
    log_pressures_list = np.log10(chemistry_grid['pressures'])
    conv_flags = chemistry_grid['conv_flags']
    property_names = chemistry_grid['property_names']
    grid_lists = chemistry_grid['grid_lists']

    # Number of values passed to the function along each axis of the grid
    input_lens = [np.size(elem) for elem in param_values]
    max_len = max(input_lens)

    np.seterr(divide = 'ignore')

    # Check that the chemical species we want to interpolate are supported
    if (grid in vulcan_grid_list):
        supported_species = vulcan_supported_species
    else:
        raise Exception("Error: unsupported VULCAN grid")
    if isinstance(chemical_species, str):
        if chemical_species not in supported_species: 
            raise Exception(chemical_species + " is not supported by VULCAN grids.")
    else:
        for species in chemical_species:
            if species not in supported_species: 
                raise Exception(species + " is not supported by VULCAN grids.")

    # Check that the log pressures of the grid are strictly decreasing
    if not np.all(log_pressures_list[:-1] >= log_pressures_list[1:]):
        raise Exception("Log pressures for the profiles in the input chemistry grid must be strictly decreasing.")

    # Determine which axes only have one value
    axes_lens = np.array([len(list) for list in grid_lists])
    one_val_only = np.array(axes_lens < 2, bool)

    # Verify that the axes with more than one value are given in param_names, and fix if not in the same order
    more_than_one_val = [not entry for entry in one_val_only]
    grid_varied_property_names = property_names[more_than_one_val] #names of the axes corresponding to the variable parameters
    grid_varied_lists = grid_lists[more_than_one_val] #just the axes corresponding to the variable parameters
    if set(grid_varied_property_names) == set(param_names):
        index_list = [np.nonzero(grid_varied_property_names == elem)[0][0] for elem in param_names]

        # Check whether the order is the same; if not then reorder
        if not np.all(index_list[1:] >= index_list[:-1]):
            # Change the order of param_names and param_values to match the order in property_names
            reordered_param_names = [param_names[i] for i in index_list]
            reordered_param_values = [param_values[i] for i in index_list]
            param_names = reordered_param_names
            param_values = reordered_param_values

    else:
        raise Exception(f"Error: the input parameters do not match the free parameters of the loaded-in grid ({grid_varied_property_names})")
    

    # Check that the desired values along each free parameter axis fall within the grid
    def not_valid(params, grid):
            return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    for i in range(len(param_values)):
        if not_valid(param_values[i], grid_varied_lists[i]):
            raise Exception(f"Requested {param_names[i]} is out of the grid bounds ({np.min(grid_varied_lists[i])}, {np.max(grid_varied_lists[i])}).")
    if not_valid(log_P, np.flip(log_pressures_list)): #Must flip log pressures during check since ordered from greatest to least
        raise Exception(f"Requested log pressures(s) are out of the grid bounds ({np.min(log_pressures_list)}, {np.max(log_pressures_list)}).")


    # For either a single point in the grid or arrays
    else:
        # Check shapes of the desired values along each axis
        for val in input_lens:
            if not val in (1, max_len):
                raise Exception("Input shape not accepted. The lengths must either be the same or 1.")

    # Interpolate the mixing ratio profiles from grid onto the provided points
    def interpolate(species):

        # Find index of the species
        q = np.where(chemical_species == species)[0][0]

        # Decide which properties to include in the interpolation (only those with more than one value in the grid)
        interpolate_over = np.array(grid_lists, dtype=np.ndarray)[more_than_one_val]

        # Create interpolator object
        grid_interp = RegularGridInterpolator(interpolate_over, 
                                              np.squeeze(log_X_grid)[q,...])
        
        #This interpolates the grid at each point defined by the input parameters
        expanded_dims_eval_points = np.empty(shape=(np.size(param_values),), dtype=np.ndarray)
        for i, elem in enumerate(param_values): 
            if np.size(elem) == 1: 
                # If only one value provided, np.full turns it into an array
                expanded_dims_eval_points[i] = np.expand_dims(np.full(max_len, elem), 0)
            else: 
                expanded_dims_eval_points[i] = np.expand_dims(elem, 0)
        interpolated_profiles = grid_interp(np.vstack(expanded_dims_eval_points).T).T

        # 1D interpolation of the interpolated mixing ratio profiles onto a list of pressures
        return np.interp(np.expand_dims(log_P, 0), np.flip(log_pressures_list), np.flip(interpolated_profiles[:,0]))[0]
    
    
    # Interpolate the convergence flag
    def interpolate_conv():
        # Decide which properties to include in the interpolation (only those with more than one value in the grid)
        interpolate_over = np.array(grid_lists, dtype=np.ndarray)[more_than_one_val]

        #Create interpolator object
        grid_interp = RegularGridInterpolator(interpolate_over, np.squeeze(conv_flags))

        #This interpolates the grid at each point defined by the input parameters
        expanded_dims_eval_points = np.empty(shape=(np.size(param_values),), dtype=np.ndarray)
        for i, elem in enumerate(param_values): 
            if np.size(elem) == 1: 
                # If only one value provided, np.full turns it into an array
                expanded_dims_eval_points[i] = np.expand_dims(np.full(max_len, elem), 0)
            else: 
                expanded_dims_eval_points[i] = np.expand_dims(elem, 0)
        interpolated_conv_flag = grid_interp(np.vstack(expanded_dims_eval_points).T).T

        return interpolated_conv_flag

    if use_conv_flag:
        conv_flag = interpolate_conv()
    else:
        conv_flag = None

    # Returning an array (default) 
    if not return_dict:
        if isinstance(chemical_species, str):
            return interpolate(chemical_species), conv_flag
        log_X_list = []
        for _, species in enumerate(chemical_species):
            log_X_list.append(interpolate(species))
        log_X_interp_array = np.array(log_X_list)
        return log_X_interp_array, conv_flag
    
    # Returning a dictionary
    else:
        log_X_interp_dict = {}
        if isinstance(chemical_species, str):
            log_X_interp_dict[chemical_species] = interpolate(chemical_species)
            return log_X_interp_dict, conv_flag
        for _, species in enumerate(chemical_species):
            log_X_interp_dict[species] = interpolate(species)

        return log_X_interp_dict, conv_flag