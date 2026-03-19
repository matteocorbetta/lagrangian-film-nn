import numpy as np
from jax.experimental.ode import odeint
from doublependulum import DoublePendulum, angular_state_initial_conditions, mass_length_samples
from utils import load_list_of_arrays_from_h5


def initialize_system(sample: int, sys_properties: dict, system: str = 'doublependulum'):
    
    if system == 'doublependulum':
        masses, lengths = sys_properties['mass'][sample], sys_properties['length'][sample]
        dp = DoublePendulum(m1=float(masses[0]), m2=float(masses[1]), l1=float(lengths[0]), l2=float(lengths[1]))
        integration_fun = dp.analytical_state_transition
        return integration_fun
    else:
        return None


if __name__ == '__main__':

    print('Generating datasets')
    import os
    import h5py

    # Define simulation variables:
    # ==========================
    system    = 'doublependulum'
    save_filename = 'dp_trajectories_samples.h5'

    data_type = np.float64
    n_samples = 50
    t_span = [0, 20]    # time span
    n_sim_steps = 200

    
    num_steps = int( n_sim_steps * (t_span[1] - t_span[0]) )    # number of time steps
    times = np.linspace(t_span[0], t_span[1], num_steps).astype(data_type)
    times_v = times.reshape((-1,1))
    if system == 'doublependulum':
        # Write rejection-sampling to generate only low-energy trajectories 
        initial_conditions = []
        parameters = mass_length_samples(n_samples)
        j = 0
        while len(initial_conditions) < n_samples:
            ic = angular_state_initial_conditions(1, data_type=data_type)
            # dp = DoublePendulum()   # Initialize double pendulum
            # if dp.is_low_energy(q=ic[0][:2], q_dot=ic[0][2:], m1=parameters['mass'][j, 0], m2=parameters['mass'][j, 1], l1=parameters['length'][j, 0], l2=parameters['length'][j, 1]):
            initial_conditions.append(ic)
            j += 1

        initial_conditions = np.stack(initial_conditions).squeeze(1)
    else:
        initial_conditions = []
        parameters = []

    # --- Prepare HDF5 file for streaming writes ---
    # Construct the full path for the file, including its directory
    # Assuming get_project_data_path is available from utils.py
    from utils import get_project_data_path 
    full_filepath = os.path.join(get_project_data_path(system), save_filename)
    output_dir = os.path.dirname(full_filepath)

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    # ----------------------------------------------
    
    # define tolerances for integration error
    rtol, atol = 1e-10, 1e-10

    # Open the HDF5 file in write mode ('w') once before the loop
    # This creates the file and allows us to add datasets iteratively
    with h5py.File(full_filepath, 'w') as f:
        # Create a group to store all trajectories
        h5_group = f.create_group('trajectories')

        for sample in range(n_samples):
            print(f'Generaing sample: {sample+1}')
            x0 = initial_conditions[sample]
            integration_fun = initialize_system(sample, sys_properties=parameters, system='doublependulum')
            x_t = odeint(integration_fun, x0, t=times, rtol=rtol, atol=atol)
            
            # Add time vector to store dt used for simulation
            np_x_t = np.concatenate((times_v, np.asarray(x_t)), axis=1)

            # Save the current trajectory as a new dataset in the HDF5 group
            h5_group.create_dataset(f'trajectory_{sample:03d}', data=np_x_t, compression="gzip", compression_opts=9)
            h5_group.create_dataset(f'params_{sample:03d}', data=np.concatenate((parameters['mass'][sample], parameters['length'][sample])), compression='gzip', compression_opts=9)
    print()
    print(f"\nAll trajectories saved to {full_filepath}")  
    
    # reload them for verification
    Xt, params = load_list_of_arrays_from_h5(system=system, filename=save_filename)
    print(f"Loaded {len(Xt)} trajectories for verification.")
    
    