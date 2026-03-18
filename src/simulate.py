from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
import equinox as eqx


def rk4_step(full_state, dt, model, norm_stats):
    
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    dXdt_mean, dXdt_std = norm_stats['dXdt_mean'], norm_stats['dXdt_std']

    def f(state):
        state_norm = (state - X_mean[:4]) / X_std[:4]
        p_norm = full_state[4:]
        qtt_norm = model(state_norm[:2], state_norm[2:], p_norm)
        qtt_phy = qtt_norm * dXdt_std + dXdt_mean
        return jnp.concatenate([state[2:], qtt_phy])
    
    state = full_state[:4]
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return jnp.concatenate([new_state, full_state[4:]])

# --- Rollout Function (JITted with lax.scan) ---
def make_rollout(n_steps, norm_stats):
    @eqx.filter_jit
    def rollout(model, state0, dt):
        """
        Simulates the system's trajectory using the provided model and RK4 integration.

        Args:
            model (callable): A function that computes the second derivatives (qtt)
                            given (q, qt, p).
            state0 (jnp.ndarray): The initial full state vector
                                [q1, q2, w1, w2, m1, m2, l1, l2] in unnormalized units.
            dt (float): The time step size for integration.
            n_steps (int): The number of integration steps to perform.

        Returns:
            jnp.ndarray: A 2D array containing the full state at each time step,
                        including the initial state. Shape is (n_steps + 1, state_dim).
        """
        
        # Generate the time points for each step where the RK4 step starts
        # This sequence will be passed to lax.scan as `xs`
        # times_for_steps = jnp.arange(0, (n_steps-1)*dt, dt)

        def scan_fn(carry_state, _):
            """
            Function to be iterated by lax.scan.
            
            Args:
                carry_state (jnp.ndarray): The state from the previous iteration.
                current_t_for_step (float): The current time 't' for this integration step.
            
            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: 
                    - next_carry_state: The updated state to pass to the next iteration.
                    - output_to_collect: The current state, which will be collected.
            """
            next_state = rk4_step(carry_state, dt, model, norm_stats)
            return next_state, next_state

        # Execute the simulation loop using lax.scan
        # lax.scan returns (final_carry, accumulated_outputs)
        # The initial carry is `state0`.
        # `times_for_steps` provides the `current_t_for_step` for each iteration.
        _, states_collected = lax.scan(scan_fn, state0, None, length=n_steps - 1)

        # Prepend the initial state to the collected states to get the full trajectory
        all_states = jnp.concatenate([state0[jnp.newaxis, :], states_collected])

        return all_states
    return rollout

def save_rollout_data(
    save_dir: Path,
    filename_prefix: str,
    times: jax.Array,
    gt_states: jax.Array,
    sim_states: jax.Array,
    params_phys: jax.Array,
    case_label: str = "" # e.g., "train_traj_0", "test_traj_0", "ood_case_0"
):
    """Saves ground truth and simulated trajectory data to a compressed .npz file."""
    full_filename = save_dir / f"{filename_prefix}_{case_label}.npz"

    # Convert JAX arrays to NumPy for saving, if they aren't already
    np.savez_compressed(
        full_filename,
        times=np.asarray(times),
        ground_truth=np.asarray(gt_states),
        simulated=np.asarray(sim_states),
        physical_parameters=np.asarray(params_phys)
    )
    print(f"✅ Rollout data saved to: {full_filename}")

    

if __name__ == '__main__':

    print('Simulation utils')
