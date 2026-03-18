from pathlib import Path
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
from data import load_list_of_arrays_from_h5
from data.doublependulum import DoublePendulum

from lnn import LagrangianNN
from simulate import make_rollout, save_rollout_data
from train_utils import normalize_data, build_input_output, train_test_split, run_diagnostics, load_model


# GLOBAL VARS
# ===============
MODEL_DIR = Path(__file__).resolve().parent / "models"   # ← src/models/
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" # <- results
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
def test_ood(model, norm_stats, x0, dt, n_steps, params_ood, fname_prefix: str, case_idx: int = 0):
    """
    Test model on out-of-distribution parameters.
    params_ood: dict with keys m1, m2, l1, l2
    """
    m1, m2, l1, l2 = params_ood['m1'], params_ood['m2'], params_ood['l1'], params_ood['l2']
    
    # generate ground truth trajectory
    dp = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2)
    
    # use a low-energy initial condition
    # x0 = jnp.array([0.3, -0.3, 0.1, -0.1])
    # assert dp.is_low_energy(x0[:2], x0[2:], m1, m2, l1, l2), "Initial condition is not low energy"
    
    times = jnp.arange(n_steps) * dt
    gt_states = odeint(dp.analytical_state_transition, x0, t=times, rtol=1e-10, atol=1e-10)

    # normalize params for model
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    p_phys = jnp.array([m1, m2, l1, l2])
    p_norm = (p_phys - X_mean[4:]) / X_std[4:]
    state0_full = jnp.concatenate([x0, p_norm])

    # rollout
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    states_ = rollout_fn(model, state0_full, dt=dt)

    # Save OOD results data
    save_rollout_data(
        save_dir=OOD_RESULTS_DIR,
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
    fig.suptitle(f'OOD test — m1={m1}, m2={m2}, l1={l1}, l2={l2}\n(training range: m∈[0.55,1.82], l∈[0.9,2.0])')
    for i, ax in enumerate(axs):
        ax.plot(times, gt_states[:, i], label='ground truth')
        ax.plot(times, states_[:, i], '--', label='sim')
        ax.set_title(state_names[i])
        if i == 0:
            ax.legend()
    plt.tight_layout()
    return fig


def plot_energy(model, state0_full, dt, n_steps, norm_stats):
    """
    Rollout the model and plot H(t) = T + V along the trajectory.
    
    state0_full: [q1, q2, w1, w2, p_norm...] — physical state + normalized params
    
    """
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    states_ = rollout_fn(model, state0_full, dt=dt)  # [n_steps, 8], physical state

    X_mean = norm_stats['X_mean']
    X_std  = norm_stats['X_std']
    p_norm = state0_full[4:]

    def compute_H(state_phys):
        state_norm = (state_phys - X_mean[:4]) / X_std[:4]
        q_norm  = state_norm[:2]
        qt_norm = state_norm[2:]
        trig_q  = jnp.array([jnp.sin(q_norm[0]), jnp.cos(q_norm[0]),
                              jnp.sin(q_norm[1]), jnp.cos(q_norm[1])])
        film_params = model.film_net(p_norm).reshape(model.n_hidden, 2)
        chol = model.compute_cholesky_entries(trig_q, film_params)
        L = jnp.array([[jax.nn.softplus(chol[0]), 0.0],
                        [chol[1],                  jax.nn.softplus(chol[2])]])
        M = L.T @ L + jnp.eye(2) * 1e-6
        T = 0.5 * qt_norm @ M @ qt_norm
        V = model.compute_potential(trig_q, p_norm)
        return T + V

    # model H along rollout
    H_model = jax.vmap(compute_H)(states_[:, :4])
    H_drift_relative = (H_model - H_model[0]) / jnp.abs(H_model[0])
    t = jnp.arange(n_steps) * dt

    # fig, axes = plt.subplots(1, 1, figsize=(14, 4))
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)
    ax.plot(t, H_drift_relative*100, label='model H drift')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('H(t) - H(0) / H(0) [%]')
    ax.set_title('Model Hamiltonian relative drift')
    ax.legend()

    plt.tight_layout()
    return fig


def gen_multiple_plots(X, X_norm, norm_stats, n_steps: int = 3000, num_plots: int = 10, fname_prefix: str = 'rollout'):
    rollout_fn = make_rollout(n_steps=n_steps, norm_stats=norm_stats)
    for traj_num in range(num_plots):
        state0 = X[traj_num, 0, :4]
        p_norm = X_norm[traj_num, 0, 4:]
        state0_full = jnp.concatenate([state0, p_norm])
        states_ = rollout_fn(model, state0_full, dt=dt)

        # Save rollout data
        p_phys = X[traj_num, 0, 4:]
        save_rollout_data(
            save_dir=ROLLOUTS_DIR,
            filename_prefix=fname_prefix,
            times=time_v[:n_steps], # Only save times relevant to n_steps
            gt_states=X[traj_num, :n_steps, :4], # Ground truth q, qt for n_steps
            sim_states=states_[:, :4], # Simulated q, qt for n_steps
            params_phys=p_phys, # Physical parameters of the system
            case_label=f"traj_{traj_num}"
        )
        # -------------------------

        # Plotting (assuming plot_rollout is defined to take specific data)
        fig = plot_rollout(X[traj_num], states_, time_v, n_steps)
        fig.savefig(ROLLOUTS_DIR / f"{fname_prefix}_{traj_num}.png", dpi=300) # Save plots in results/rollouts/
        plt.close(fig) # Close figure to free memory


def plot_rollout(X, states_, time_v, n_steps):
    state_names = [r'$q_1$', r'$q_2$', r'$qdot_1$', r'$qdot_2$']
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

    # LOAD MODEL
    # ===================
    pos_dim   = 2
    vel_dim   = 2
    param_dim = 4
    state_dim = pos_dim + vel_dim

    key = jax.random.PRNGKey(123)
    model_key, train_key = jax.random.split(key)
    
    model_name = f"model_T512_20260317_133032"
    model = LagrangianNN(pos_dim=pos_dim,vel_dim=pos_dim, param_dim=param_dim,hidden_dim=128, n_hidden=2,key=model_key)
    model = load_model(model, MODEL_DIR / model_name)

    # LOAD DATA
    # =======
    # Dataset is a list of arrays of shape [T, 5] = (time, q1, q2, qdot1, qdot2)
    datasets, params = load_list_of_arrays_from_h5(system='doublependulum', filename='dp_trajectories.h5')
    dt     = datasets[0][1, 0] - datasets[0][0, 0]
    time_v = datasets[0][:, 0]
    
    # Build dataset for training: from x = [q, dqdt], build dx/dt numerically
    # ========================================================================
    X, dXdt = build_input_output(datasets=datasets, params=params, dt=dt)
    
    # Build train/test sets
    # ==========================
    idx_train, idx_val, idx_test = train_test_split(X, n_train=0.8, n_val=0.1, seed=SEED)
    # extract
    Xtrain, Xval, Xtest = X[idx_train], X[idx_val], X[idx_test]
    dXdt_train, dXdt_val, dXdt_test = dXdt[idx_train], dXdt[idx_val], dXdt[idx_test]

    # Normalize data
    # -----------------
    # X columns: [q1, q2, w1, w2, m1, m2, l1, l2]
    # q1, q2 are raw angles in radians — trig transform handles them, don't normalize
    # w1, w2, m1, m2, l1, l2 — normalize by mean and std
    Xtrain_norm, Xval_norm, Xtest_norm, \
        dXdt_train_norm, dXdt_val_norm, dXdt_test_norm, \
            norm_stats = normalize_data(Xtrain, Xval, Xtest, dXdt_train, dXdt_val, dXdt_test, len(params), normalize=True)

    # Simulate rollout and save data
    # ===============================
    n_steps = 3000

    # Save and plot train set rollouts
    fig_train = gen_multiple_plots(X=Xtrain, X_norm=Xtrain_norm, norm_stats=norm_stats, 
                                   n_steps=n_steps, num_plots=3, fname_prefix=model_name+'_train')
    
    # Save and plot test set rollouts
    fig_test = gen_multiple_plots(X=Xtest, X_norm=Xtest_norm, norm_stats=norm_stats, 
                                  n_steps=n_steps, num_plots=10, fname_prefix=model_name+'_test')
    
    # Compute energy
    # ================
    for traj_num in range(10):
        state0_full = jnp.concatenate([Xtest[traj_num, 0, :4], Xtest_norm[traj_num, 0, 4:]])
        params_phys = Xtest[traj_num, 0, 4:]
        fig = plot_energy(model, state0_full, dt, n_steps=n_steps, norm_stats=norm_stats)
        # Save energy plot in results/rollouts/
        fig.savefig(str(ROLLOUTS_DIR) + '/' + str(model_name) + f'_energy_test_{traj_num}.png', dpi=300)
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
    for i, params_ood in enumerate(ood_cases):
        fig = test_ood(model, norm_stats, x0s[i], dt, n_steps=n_steps, params_ood=params_ood, fname_prefix=model_name, case_idx=i)
        fig.savefig(str(OOD_RESULTS_DIR) + '/' + str(model_name) + f'_ood_{i}.png', dpi=300)
        
    print('End of script')
    plt.show()