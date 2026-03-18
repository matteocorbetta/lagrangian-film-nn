
from typing import List
import h5py
import numpy as np
import os

# paths
# =====
# --- Helper function to get the project root and construct the full data path ---
def get_project_data_path(sub_path=''):
    """
    Constructs an absolute path to a location within the project's /data directory.
    Assumes this script is run from within the project structure (e.g., src/data/).
    """
    # Get the directory of the current script (e.g., 'path/to/project_root/src/data')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up to the project root (e.g., 'path/to/project_root')
    # Assuming script is in src/data/, need to go up two levels
    project_root = os.path.join(script_dir, '..', '..')
    
    # Construct the path to the desired data subdirectory
    full_data_path = os.path.join(project_root, 'data', sub_path)
    
    return full_data_path

# Save / Load Files
# ===================
def save_list_of_arrays_to_h5(list_of_arrays: List[np.array], system: str = 'doublependulum', filename: str = 'list_of_trajectories.h5') -> None:
    """Saves a list of 2D NumPy or JAX arrays into a single compressed HDF5 file.

    Each array in the list is stored as a separate dataset within an HDF5 group
    named 'trajectories'. The individual datasets are named sequentially
    (e.g., 'trajectory_000', 'trajectory_001', etc.). The function ensures that
    the target directory structure exists before writing the file.

    Args:
        list_of_arrays (List[Union[np.ndarray, jnp.ndarray]]): A list of 2D arrays
            (either NumPy arrays or JAX arrays) to be saved.
        system (str, optional): The name of the subdirectory within the project's
            'data/' folder where the HDF5 file should be stored. Defaults to 'doublependulum'.
        filename (str, optional): The name of the HDF5 file to create.
            Defaults to 'trajectories.h5'.

    Side Effects:
        - Creates the target directory (e.g., 'project_root/data/doublependulum/') if it does not already exist.
        - Creates an HDF5 file at the specified path.
        - Prints messages indicating directory creation and successful file saving.
    """
    full_path = os.path.join(get_project_data_path(system), filename)

    output_dir = os.path.dirname(full_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    with h5py.File(full_path, 'w') as f:
        # Create a group to better organize your arrays, if desired
        # If you have different types of lists, you could create multiple groups
        group = f.create_group('trajectories') 
        
        for i, arr in enumerate(list_of_arrays):
            # Convert JAX array to NumPy array for saving if it's not already
            np_arr = np.asarray(arr)
            # Create a dataset for each array within the group
            # Use gzip compression, which is generally a good balance.
            # You can also use 'lzf' for faster (de)compression, sometimes at lower ratio.
            group.create_dataset(f'trajectory_{i:03d}', data=np_arr, compression="gzip", compression_opts=9)
    print(f"List of arrays saved to {full_path}")


def load_list_of_arrays_from_h5(system: str = 'doublependulum', filename: str = 'trajectories.h5') -> List[np.array]:
    """Loads a list of 2D NumPy arrays from a specified HDF5 file.

    The function constructs the file path relative to the project's root
    data directory. It expects the arrays to be stored within an HDF5 group
    named 'trajectories', with individual datasets named systematically (e.g., 'trajectory_000').

    Args:
        system (str, optional): The name of the subdirectory within the project's
            'data/' folder where the HDF5 file is located. Defaults to 'doublependulum'.
        filename (str, optional): The name of the HDF5 file to load. Defaults to 'trajectories.h5'.

    Returns:
        List[np.array]: A list of 2D NumPy arrays loaded from the HDF5 file.
            Returns an empty list if the specified file does not exist.
    """
    full_filepath = os.path.join(get_project_data_path(system), filename)

    loaded_trajs = []
    loaded_params = []
    if not os.path.exists(full_filepath):
        print(f"Error: File not found at {full_filepath}")
        return []

    with h5py.File(full_filepath, 'r') as f:
        group = f['trajectories']
        for key in sorted(group.keys()):
            if key.startswith('trajectory'):
                loaded_trajs.append(group[key][()])
            elif key.startswith('param'):
                loaded_params.append(group[key][()])

    print(f"Loaded {len(loaded_trajs)} trajectories from {full_filepath}")
    return loaded_trajs, loaded_params

if __name__ == '__main__':

    path = get_project_data_path('doublependulum')

    print('end')
