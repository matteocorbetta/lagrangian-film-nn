from pathlib import Path
import argparse
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
from lagrangian_film_nn.data import load_list_of_arrays_from_h5
from lagrangian_film_nn.data.doublependulum import DoublePendulum, ParameterSupport

from lagrangian_film_nn.simulate import make_rollout, save_rollout_data
from lagrangian_film_nn.train_utils import normalize_data, build_input_output, train_test_split, run_diagnostics, load_model
from lagrangian_film_nn.utils import RESULTS_DIR


# GLOBAL VARS
# ===============
# Create specific subdirectories for different types of rollouts
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
OOD_RESULTS_DIR = RESULTS_DIR / "ood_tests"

# Ensure these directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)
OOD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SEED = 5678 # to match training and have the same test set data

# FUNCTIONS
# ===========
def resolve_dataset(model_path: Path, dataset: Optional[str] = None) -> str:
    """Returns the dataset filename the model was trained on.

    Args:
        model_path (Path): Checkpoint path without extension; '<model_path>.npz' is read.
        dataset (str, optional): Explicit filename. Overrides the one stored with the model;
                                 required for models trained before it was stored.

    Returns:
        str: HDF5 filename in data/doublependulum/.
    """
    if dataset is not None:
        return dataset
    with np.load(str(model_path) + '.npz', allow_pickle=True) as run_info:
        if 'dataset' not in run_info.files:
            raise KeyError(f"{model_path}.npz does not store the training dataset. Pass --dataset explicitly.")
        return str(run_info['dataset'])


def test_ood(model, norm_stats, x0, dt, n_steps, params_ood, support: ParameterSupport,
             fname_prefix: str, save_dir: Path, case_idx: int = 0):
    """
    Test model on out-of-distribution parameters.
    params_ood: dict with keys m1, m2, l1, l2
    support: training-parameter support, used to report how far the case lies outside it.
    save_dir: directory where the rollout data is saved.
    """
    m1, m2, l1, l2 = params_ood['m1'], params_ood['m2'], params_ood['l1'], params_ood['l2']
    distance = support.distance([m1, m2, l1, l2])[0]
    
    # generate ground truth trajectory
    dp = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2)
    
    # use a low-energy initial condition
    # x0 = jnp.array([0.3, -0.3, 0.1, -0.1])
    # assert dp.is_low_energy(x0[:2], x0[2:], m1, m2, l1, l2), "Initial condition is not low energy"
    
    times = jnp.arange(n_steps) * dt
    gt_states = odeint(dp.analytical_state_transition, x0, t=times, rtol=1e-10, atol=1e-10)

    # normalize params p for model
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    p_phys = jnp.array([m1, m2, l1, l2])
    p_norm = (p_phys - X_mean[4:]) / X_std[4:]
    qa_init = jnp.concatenate([x0, p_norm])

    # rollout
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    states_ = rollout_fn(model, qa_init, dt=dt)

    # Save OOD results data
    save_rollout_data(
        save_dir=save_dir,
        filename_prefix=fname_prefix,
        times=times,
        gt_states=gt_states,
        sim_states=states_,
        params_phys=params_ood,
        case_label=f"case_{case_idx}"
    )

    # plot
    state_names = [r'$q_1$', r'$q_2$', r'$\dot{q}_1$', r'$\dot{q}_2$']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.reshape(-1)
    distance_str = ", ".join(f"{n}: {d:.2f}" for n, d in zip(support.COORD_NAMES, distance))
    fig.suptitle(f'OOD test — m1={m1}, m2={m2}, l1={l1}, l2={l2}\n'
                 f'training support: {support.describe()}\n'
                 f'distance outside support [box widths]: {distance_str}')
    for i, ax in enumerate(axs):
        ax.plot(times, gt_states[:, i], label='ground truth')
        ax.plot(times, states_[:, i], '--', label='sim')
        ax.set_title(state_names[i])
        if i == 0:
            ax.legend()
    plt.tight_layout()
    return fig


def plot_energy(model, qa_init, dt, n_steps, norm_stats):
    """
    Roll out the model and plot the drift of its normalized Hamiltonian over
    time.
    
    qa_init: [q1, q2, q1_t, q2_t, p_norm...] where the state is physical and the
    parameters are normalized for the model.
    
    """
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    states_ = rollout_fn(model, qa_init, dt=dt)  # [n_steps, 8], physical state

    X_mean = norm_stats['X_mean']
    X_std  = norm_stats['X_std']
    p_norm = qa_init[4:]

    def compute_H(state_phys):
        state_norm = (state_phys - X_mean[:4]) / X_std[:4]
        q_norm  = state_norm[:2]
        q_t_norm = state_norm[2:]
        q_trig  = jnp.array([jnp.sin(q_norm[0]), jnp.cos(q_norm[0]),
                              jnp.sin(q_norm[1]), jnp.cos(q_norm[1])])
        film_p = model.film_net(p_norm).reshape(model.n_hidden, 2)
        chol = model.compute_cholesky_entries(q_trig, film_p)
        L = jnp.array([[jax.nn.softplus(chol[0]), 0.0],
                        [chol[1],                  jax.nn.softplus(chol[2])]])
        M = L.T @ L + jnp.eye(2) * 1e-6
        T = 0.5 * q_t_norm @ M @ q_t_norm
        V = model.compute_potential(q_norm, p_norm)
        return T + V

    # Model normalized Hamiltonian along rollout
    H_model = jax.vmap(compute_H)(states_[:, :4])
    H_drift_relative = (H_model - H_model[0]) / jnp.abs(H_model[0])
    t = jnp.arange(n_steps) * dt

    # fig, axes = plt.subplots(1, 1, figsize=(14, 4))
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)
    ax.plot(t, H_drift_relative*100, label='model H drift')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('relative drift of normalized energy [%]')
    ax.set_title('Normalized Hamiltonian relative drift')
    ax.legend()

    plt.tight_layout()
    return fig


def gen_multiple_plots(X, X_norm, norm_stats, save_dir: Path, n_steps: int = 3000, num_plots: int = 10, fname_prefix: str = 'rollout'):
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    for traj_num in range(num_plots):
        q_init = X[traj_num, 0, :4]
        p_norm = X_norm[traj_num, 0, 4:]
        qa_init = jnp.concatenate([q_init, p_norm])
        states_ = rollout_fn(model, qa_init, dt=dt)

        # Save rollout data
        p_phys = X[traj_num, 0, 4:]
        save_rollout_data(
            save_dir=save_dir,
            filename_prefix=fname_prefix,
            times=time_v[:n_steps], # Only save times relevant to n_steps
            gt_states=X[traj_num, :n_steps, :4], # Ground truth q, q_t for n_steps
            sim_states=states_[:, :4], # Simulated q, q_t for n_steps
            params_phys=p_phys, # Physical parameters of the system
            case_label=f"traj_{traj_num}"
        )
        # -------------------------

        # Plotting (assuming plot_rollout is defined to take specific data)
        fig = plot_rollout(X[traj_num], states_, time_v, n_steps)
        fig.savefig(save_dir / f"{fname_prefix}_{traj_num}.png", dpi=300) # Save plots in results/rollouts/<run>/
        plt.close(fig) # Close figure to free memory


def plot_rollout(X, states_, time_v, n_steps):
    state_names = [r'$q_1$', r'$q_2$', r'$q_{1,t}$', r'$q_{2,t}$']
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs = axs.reshape((-1,))
    fig.suptitle(f'Pendulum properties:\nmasses=({round(X[0, 4],2)},{round(X[0, 5],2)}), rods=({round(X[0, 6],2)}, {round(X[0, 7],2)})')
    for i, ax in enumerate(axs):
        ax.plot(time_v[:n_steps], X[:n_steps, i], label='ground truth')
        ax.plot(time_v[:n_steps], states_[:, i], '--', label='sim')
        ax.set_title(state_names[i], fontsize=10)
        if i == 0:
            ax.legend(fontsize=10)
    return fig


if __name__ == '__main__':

    # COMMAND LINE ARGUMENTS
    # =======================
    parser = argparse.ArgumentParser(description="Rollouts, energy drift and OOD tests for a trained Lagrangian NN.")
    parser.add_argument('--model', type=Path, required=True,
                        help="Checkpoint path without extension, e.g. results/models/model_new_K2_s123_T512_<timestamp>.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="HDF5 file in data/doublependulum/. Defaults to the one stored with the model.")
    args = parser.parse_args()

    # LOAD MODEL
    # ===================
    # Architecture is rebuilt from the '.json' saved next to the checkpoint.
    model, hyperparams = load_model(fname=args.model)
    model_name = args.model.name
    print(f"Potential method: {hyperparams['potential_method']}")

    # Per-run output folders, so results of different models do not overwrite each other
    run_rollouts_dir = ROLLOUTS_DIR / model_name
    run_ood_dir      = OOD_RESULTS_DIR / model_name
    run_rollouts_dir.mkdir(parents=True, exist_ok=True)
    run_ood_dir.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # =======
    # Dataset is a list of arrays of shape [T, 5] = (time, q1, q2, q1_t, q2_t)
    dataset_name = resolve_dataset(args.model, args.dataset)
    print(f"Dataset: {dataset_name}")
    datasets, p = load_list_of_arrays_from_h5(system='doublependulum', filename=dataset_name)
    dt     = datasets[0][1, 0] - datasets[0][0, 0]
    time_v = datasets[0][:, 0]
    
    # Build dataset for training: from x = [q, dqdt], build dx/dt numerically
    # ========================================================================
    X, dXdt = build_input_output(datasets=datasets, params=p, dt=dt)
    
    # Build train/test sets
    # ==========================
    idx_train, idx_val, idx_test = train_test_split(X, n_train=0.8, n_val=0.1, seed=SEED)
    # extract
    Xtrain, Xval, Xtest = X[idx_train], X[idx_val], X[idx_test]
    dXdt_train, dXdt_val, dXdt_test = dXdt[idx_train], dXdt[idx_val], dXdt[idx_test]

    # Training-parameter support: parameters are constant along a trajectory, so take t=0
    support = ParameterSupport.from_params(np.asarray(Xtrain[:, 0, 4:]))
    print(f"Training support: {support.describe()}")

    # Normalize data
    # -----------------
    # X columns: [q1, q2, q1_t, q2_t, m1, m2, l1, l2]
    # q1, q2 are raw angles in radians — trig transform handles them, don't normalize
    # q1_t, q2_t, m1, m2, l1, l2 — normalize by mean and std
    Xtrain_norm, Xval_norm, Xtest_norm, \
        dXdt_train_norm, dXdt_val_norm, dXdt_test_norm, \
            norm_stats = normalize_data(Xtrain, Xval, Xtest, dXdt_train, dXdt_val, dXdt_test, len(p), normalize=True)

    # Simulate rollout and save data
    # ===============================
    n_steps = 3000

    # Save and plot train set rollouts
    fig_train = gen_multiple_plots(X=Xtrain, X_norm=Xtrain_norm, norm_stats=norm_stats, save_dir=run_rollouts_dir,
                                   n_steps=n_steps, num_plots=3, fname_prefix=model_name+'_train')

    # Save and plot test set rollouts
    fig_test = gen_multiple_plots(X=Xtest, X_norm=Xtest_norm, norm_stats=norm_stats, save_dir=run_rollouts_dir,
                                  n_steps=n_steps, num_plots=10, fname_prefix=model_name+'_test')
    
    # Compute energy
    # ================
    for traj_num in range(10):
        qa_init = jnp.concatenate([Xtest[traj_num, 0, :4], Xtest_norm[traj_num, 0, 4:]])
        p_phys = Xtest[traj_num, 0, 4:]
        fig = plot_energy(model, qa_init, dt, n_steps=n_steps, norm_stats=norm_stats)
        # Save energy plot in results/rollouts/
        fig.savefig(run_rollouts_dir / f'{model_name}_energy_test_{traj_num}.png', dpi=300)
        plt.close(fig)
    
    # OOD tests — progressively further from training distribution
    # =============================================================
    ood_cases = [
        {'m1': 2.5, 'm2': 1.5, 'l1': 1.2, 'l2': 1.3},   # heavy masses, in-range lengths
        {'m1': 1.0, 'm2': 0.8, 'l1': 2.5, 'l2': 2.3},   # in-range masses, long rods
        {'m1': 3.0, 'm2': 2.0, 'l1': 2.8, 'l2': 2.5},   # both out of range
        {'m1': 0.3, 'm2': 0.2, 'l1': 0.5, 'l2': 0.6},   # small masses and lengths
    ]
    x0s = [
        jnp.array([0.4, -0.4, 0.1, -0.1]),
        jnp.array([-0.8, 0.8, 0.3, -0.3]),
        jnp.array([-0.6, 0.6, -0.9, 0.9]),
        jnp.array([1.5, 1.0, -1.2, 0.35]),
    ]
    for i, p_ood in enumerate(ood_cases):
        p_vec = [p_ood['m1'], p_ood['m2'], p_ood['l1'], p_ood['l2']]
        if not support.is_ood(p_vec)[0]:
            print(f"WARNING: OOD case {i} {p_ood} lies inside the training support.")
        fig = test_ood(model, norm_stats, x0s[i], dt, n_steps=n_steps, params_ood=p_ood, support=support,
                       fname_prefix=model_name, save_dir=run_ood_dir, case_idx=i)
        fig.savefig(run_ood_dir / f'{model_name}_ood_{i}.png', dpi=300)
        
    print('End of script')
    plt.show()