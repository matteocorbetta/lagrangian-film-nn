from typing import List, Dict, Tuple
from pathlib import Path

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from data.doublependulum import DoublePendulum

def save_model(model: eqx.Module, fname: Path):
    """Saves an Equinox model's leaves (trainable parameters) to a file.

    Args:
        model (eqx.Module): The Equinox model to save.
        fname (Path): The base path/filename for the model. A '.eqx' extension will be added.
    """
    eqx.tree_serialise_leaves(str(fname)+'.eqx', model)
    print(f'Model saved: {fname}')

def load_model(model: eqx.Module, fname: str = "model") -> eqx.Module:
    """Loads an Equinox model's leaves (trainable parameters) from a file.

    Args:
        model (eqx.Module): An uninitialized Equinox model with the correct architecture
                            to load the parameters into.
        fname (Path): The base path/filename of the model to load. Assumes a '.eqx' extension.

    Returns:
        eqx.Module: The model with loaded parameters.
    """
    model = eqx.tree_deserialise_leaves(str(fname)+".eqx", model)
    print(f"Model loaded from: {fname}")
    return model

def compute_V_stats(datasets: List[jax.Array], params: List[jax.Array], idx_train: np.ndarray) -> Tuple[jnp.Array, jnp.Array]:
    """Computes mean and standard deviation of potential energy for the training set.

    Args:
        datasets (List[jnp.Array]): List of all trajectory datasets.
        params (List[jnp.Array]): List of all system parameters corresponding to datasets.
        idx_train (np.ndarray): Indices of trajectories belonging to the training set.

    Returns:
        Tuple[jnp.Array, jnp.Array]: Mean and standard deviation of potential energy.
    """
    V_all = []
    for i in idx_train:
        traj = datasets[i]
        p = params[i]
        #  Instantiate DoublePendulum for each parameter set to compute potential energy
        dp = DoublePendulum(m1=p[0], m2=p[1], l1=p[2], l2=p[3])
        # traj[:, 1:3] contains q1, q2
        V_all.append(dp.potential_energy(traj[:, 1:3]))

    V_all = jnp.stack(V_all).reshape((-1,))
    return jnp.mean(V_all), jnp.std(V_all)


def build_temporal_batch(x: jnp.Array, 
                         y: jnp.Array, 
                         batch_size: int, 
                         temporal_chunk_len: int, 
                         step_key) -> List[jnp.Array, jnp.Array]:
    """Builds a batch of data by sampling random temporal chunks from random trajectories.

    Args:
        x (jax.Array): Input data, shape (num_trajectories, time_steps, features).
        y (jax.Array): Target data, shape (num_trajectories, time_steps, output_dim).
        batch_size (int): Number of trajectories to sample for this batch.
        temporal_chunk_len (int): Length of the time chunk to extract from each sampled trajectory.
        step_key (jax.Array): JAX PRNGKey for random sampling.

    Returns:
        Tuple[jax.Array, jax.Array]: A batch of inputs (x_batch) and targets (y_batch),
                                     concatenated along the first axis.
                                     Shapes: (batch_size * temporal_chunk_len, features)
                                     and (batch_size * temporal_chunk_len, output_dim).
    """
    # Split key for different random operations
    sample_idxs = jax.random.randint(step_key, (batch_size,), 0, len(x))
    
    # Sample 'batch_size' starting time indices for chunks
    time_idxs   = jax.random.randint(step_key, (batch_size,), 0, x.shape[1]-temporal_chunk_len)
    
    # Extract chunks. Using a list comprehension and then concatenate is JAX-compatible.
    x_chunks = [x[sample_idxs[i]][time_idxs[i] : time_idxs[i] + temporal_chunk_len] for i in range(len(sample_idxs))]
    y_chunks = [y[sample_idxs[i]][time_idxs[i] : time_idxs[i] + temporal_chunk_len] for i in range(len(sample_idxs))]

    x_batch = jnp.concatenate(x_chunks, axis=0)  # [K*T, 8]
    y_batch = jnp.concatenate(y_chunks, axis=0)  # [K*T, 2]    
    return x_batch, y_batch


def normalize_data(Xtrain: jnp.Array,
                   Xval: jnp.Array,
                   Xtest: jnp.Array, 
                   dXdt_train: jnp.Array, 
                   dXdt_val: jnp.Array,
                   dXdt_test: jnp.Array, 
                   len_params: int, 
                   normalize: bool = True) -> List[jnp.Array, jnp.Array, jnp.Array, jnp.Array, Dict]:
    """Normalizes the input (X) and target (dXdt) datasets based on training set statistics.

    Args:
        Xtrain (jax.Array): Input training data.
        Xval (jax.Array): Input validation data.
        Xtest (jax.Array): Input test data.
        dXdt_train (jax.Array): Target training data (accelerations).
        dXdt_val (jax.Array): Target validation data (accelerations).
        dXdt_test (jax.Array): Target test data (accelerations).
        len_params (int): The number of unique parameter sets (trajectories)
                                          in the dataset. Used to handle cases where
                                          parameters might be constant across all train samples (std=0).
        normalize (bool, optional): If True, perform normalization. Otherwise, return data as is.
                                    Defaults to True.

    Returns:
        Tuple[jax.Array, ...]: Normalized (or original) X_train, X_val, X_test,
                               dXdt_train, dXdt_val, dXdt_test, and a dictionary of norm_stats.
                               The tuple order is: Xtrain_norm, Xval_norm, Xtest_norm,
                               dXdt_train_norm, dXdt_val_norm, dXdt_test_norm, norm_stats.
    """
    # Compute norm stats
    # ===================
    X_mean    = jnp.mean(Xtrain, axis=(0,1))    # average over multiple trajectories and full length of trajectory
    X_std     = jnp.std(Xtrain, axis=(0,1))     # average over multiple trajectories and full length of trajectory
    dXdt_mean = jnp.mean(dXdt_train, axis=(0,1))
    dXdt_std  = jnp.std(dXdt_train, axis=(0,1))

    # Zero out mean/std for angles so they pass through unchanged
    # --------------------------------------------------------
    X_mean = X_mean.at[0].set(0.0).at[1].set(0.0)
    X_std  = X_std.at[0].set(1.0).at[1].set(1.0)

    # Parameter normalization 
    # ========================
    # if we're testing a single trajectory, do not normalize parameters as std=0.
    # this is fine as the absolute value of parameters if faily small (around 1.)
    if len_params == 1:
        for i in [4, 5, 6, 7]:
            X_mean = X_mean.at[i].set(0.0)
            X_std = X_std.at[i].set(1.0)
    
    # Normalization stats
    # =================
    norm_stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'dXdt_mean': dXdt_mean, 'dXdt_std': dXdt_std
    }

    if normalize:
        # Input normalization
        Xtrain_norm = (Xtrain - X_mean) / X_std
        Xval_norm = (Xval - X_mean) / X_std
        Xtest_norm = (Xtest - X_mean) / X_std

        # Targets: accelerations — zero-mean normalize
        dXdt_train_norm = (dXdt_train - dXdt_mean) / dXdt_std
        dXdt_val_norm = (dXdt_val - dXdt_mean) / dXdt_std
        dXdt_test_norm = (dXdt_test - dXdt_mean) / dXdt_std
        
        return Xtrain_norm, Xval_norm, Xtest_norm, dXdt_train_norm, dXdt_val_norm, dXdt_test_norm, norm_stats
    else:
        return Xtrain, Xval, Xtest, dXdt_train, dXdt_val, dXdt_test, norm_stats
    

def build_input_output(datasets: List[jax.Array], params: List[jax.Array], dt: float) -> Tuple[jax.Array, jax.Array]:
    """Preprocesses raw trajectory data into model inputs (X) and targets (dXdt).

    The input X is augmented with system parameters.
    The target dXdt is computed via numerical differentiation of velocities.

    Args:
        datasets (List[jnp.Array]): List of raw trajectory data, each of shape (time_steps, 5)
                                    where columns are [time, q1, q2, w1, w2].
        params (List[jnp.Array]): List of system parameters, each of shape (4,)
                                  [m1, m2, l1, l2].
        dt (float): Time step size, used for numerical differentiation.

    Returns:
        Tuple[jax.Array, jax.Array]: A tuple containing:
            - X (jax.Array): Concatenated input states, shape (num_trajectories, time_steps, features).
                             Features order: [q1, q2, w1, w2, m1, m2, l1, l2].
            - dXdt (jax.Array): Computed accelerations, shape (num_trajectories, time_steps, 2).
    """
    X, dXdt = [], []
    for i, traj in enumerate(datasets):
        # Get the state vector and its derivative
        # ----------------------------------------
        x = traj[:, 1:]
        xdot = np.gradient(np.array(x[:, 2:]), dt, axis=0, edge_order=2)
        
        # get the parameters
        # ----------------
        p = params[i]
        # Tile parameters to match the time_steps dimension of x
        p_tiled = np.tile(p, (x.shape[0], 1))
        
        # augmented state vector  [q1, q2, w1, w2, m1, m2, l1, l2]
        # --------------------        
        x = jnp.concatenate([x, p_tiled], axis=1)

        # store variables for stacking layer
        # ----------------
        X.append(x)
        dXdt.append(xdot)
    
    # Stack into 3D array
    X = jnp.stack(X)
    dXdt = jnp.stack(dXdt)
    return X, dXdt



def train_test_split(X: jax.Array, n_train: float = 0.7, n_val: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        X (jax.Array): The full dataset (e.g., input trajectories).
        n_train (float, optional): Proportion of data to use for the training set. Defaults to 0.7.
        n_val (float, optional): Proportion of data to use for the validation set. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Indices for train, validation, and test sets.
    """

    total_samples = X.shape[0]
    rng_ = np.random.default_rng(seed)

    # Calculate sizes
    size_train = int(round(total_samples * n_train))
    size_val   = int(round(total_samples * n_val))
    size_test  = total_samples - size_train - size_val

    # Ensure sizes are non-negative
    size_train = max(0, size_train)
    size_val   = max(0, size_val)
    size_test  = max(0, size_test)
    if size_train + size_val + size_test == 0:
        raise ValueError("Calculated sizes lead to zero total samples. Check n_train, n_val, and X.shape[0].")

    # Get all indices
    all_indices = np.arange(total_samples)

    # Shuffle all indices for random split
    rng_.shuffle(all_indices)

    # Split into train, val, test based on shuffled indices
    idx_train = all_indices[:size_train]
    idx_val   = all_indices[size_train : size_train + size_val]
    idx_test  = all_indices[size_train + size_val : size_train + size_val + size_test]

    return idx_train, idx_val, idx_test
    



# ============================================================
# DIAGNOSTICS
# ============================================================
 
def run_diagnostics(model: eqx.Module, 
                    Xtrain: jnp.Array, 
                    Xtrain_norm: jnp.Array, 
                    dXdt_train: jnp.Array, 
                    dXdt_train_norm: jnp.Array, 
                    norm_stats: Dict, 
                    params: List[jnp.Array]):
    """One-step acceleration check and Lagrangian structure diagnostics."""
 
    state0_physical = Xtrain[0, 0, :4]
    state0_norm     = (state0_physical - norm_stats['X_mean'][:4]) / norm_stats['X_std'][:4]
    p_norm          = Xtrain_norm[0, 0, 4:]
 
    # model prediction
    q_tt_norm = model(state0_norm[:2], state0_norm[2:], p_norm)
    q_tt_phys = q_tt_norm * norm_stats['dXdt_std'] + norm_stats['dXdt_mean']
    print("q_tt_norm:", q_tt_norm)
    print("q_tt_phys:", q_tt_phys)
 
    # ground truth
    p      = Xtrain[0, 0, 4:]
    dp     = DoublePendulum(m1=p[0], m2=p[1], l1=p[2], l2=p[3])
    deriv  = dp.analytical_state_transition(Xtrain[0, 0, :4], 0.0)
    gt_norm = (deriv[2:] - norm_stats['dXdt_mean']) / norm_stats['dXdt_std']
    print("gt q_tt physical:", deriv[2:])
    print("gt q_tt normalized:", gt_norm)
    print("stored dXdt[0,0]:", dXdt_train[0, 0])
 
    # mass matrix
    lagrangian_fn = lambda _q, _qt: model.compute_lagrangian(_q, _qt, p_norm)
    l_qt_fn  = jax.grad(lagrangian_fn, argnums=1)
    l_qt_qt  = jax.jacobian(l_qt_fn, argnums=1)(state0_norm[:2], state0_norm[2:])
    l_q      = jax.grad(lagrangian_fn, argnums=0)(state0_norm[:2], state0_norm[2:])
    l_qt_q   = jax.jacobian(l_qt_fn, argnums=0)(state0_norm[:2], state0_norm[2:])
    rhs      = l_q - l_qt_q @ state0_norm[2:]
 
    print("l_qt_qt:\n", l_qt_qt)
    print("cond(l_qt_qt):", jnp.linalg.cond(l_qt_qt))
    print("eigenvalues:", jnp.linalg.eigvalsh(l_qt_qt))
    print("l_q:", l_q)
    print("rhs:", rhs)
 
    # physical M for reference
    M11_phys = (p[0] + p[1]) * p[2]**2
    M22_phys = p[1] * p[3]**2
    print(f"physical M11: {M11_phys:.3f}, M22: {M22_phys:.3f}")
    print(f"model    M11: {l_qt_qt[0,0]:.3f}, M22: {l_qt_qt[1,1]:.3f}")
    