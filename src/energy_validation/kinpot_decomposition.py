import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np # For np.load, np.asarray
from pathlib import Path
from typing import Dict # For type hints
import sys

# Path(__file__).resolve() gives the absolute path to kinpot_decomposition.py
# .parent gives src/validation/
# .parents[1] gives src/
# So, Path(__file__).resolve().parents[1] is the 'src' directory.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import necessary components from other modules
from lnn import LagrangianNN # Model architecture
from train_utils import load_model, normalize_data # Helper functions
from data.doublependulum import DoublePendulum # For analytical comparison

# GLOBAL VARS (define relative to this script's location)
# =======================================================

MODEL_DIR = SRC_DIR / "models" # Points to src/models/
RESULTS_DIR = SRC_DIR.parent / "results" # Points to project_root/results/
KPD_RESULTS_DIR = RESULTS_DIR / "kinpot_decomposition" # Specific dir for these plots
KPD_RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists


def plot_TV_decomposition(model, norm_stats, params_phys, n_grid=50):
    """
    Validates the learned T-V decomposition by comparing model V(q) and T(q, q_dot)
    against analytical values over a grid of configurations.
    
    params_phys: [m1, m2, l1, l2] in physical units
    """
    from data.doublependulum import DoublePendulum
    
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    
    # normalize params for model
    p_phys = jnp.array(params_phys)
    p_norm = (p_phys - X_mean[4:]) / X_std[4:]
    
    dp = DoublePendulum(m1=p_phys[0], m2=p_phys[1], l1=p_phys[2], l2=p_phys[3])
    
    # Grid over q1, q2 in [-pi, pi]
    q1_grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    q2_grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    Q1, Q2  = jnp.meshgrid(q1_grid, q2_grid)
    
    # --- Potential energy V(q1, q2) ---
    def model_V(q1, q2):
        q = jnp.array([q1, q2])
        trig_q = jnp.array([jnp.sin(q[0]), jnp.cos(q[0]),
                             jnp.sin(q[1]), jnp.cos(q[1])])
        return model.compute_potential(trig_q, p_norm)
    
    def gt_V(q1, q2):
        return dp.potential_energy(jnp.array([q1, q2]))
    
    V_model = jax.vmap(jax.vmap(model_V))(Q1, Q2)
    V_gt    = jax.vmap(jax.vmap(gt_V))(Q1, Q2)
    
    # align scales: V is defined up to a constant, compare shapes
    V_model_aligned = (V_model - jnp.mean(V_model)) / (jnp.std(V_model) + 1e-8)
    V_gt_aligned    = (V_gt    - jnp.mean(V_gt))    / (jnp.std(V_gt)    + 1e-8)
    
    # --- Kinetic energy T(q, q_dot) at fixed q_dot ---
    q_dot_test = jnp.array([1.0, 1.0])  # fixed representative velocity
    # q_dot_test = jnp.array([0.1, 0.1])
    # q_dot_test = jnp.array([3., 3.])
    q_dot_norm = (q_dot_test - X_mean[2:4]) / X_std[2:4]
    
    def model_T(q1, q2):
        q = jnp.array([q1, q2])
        trig_q = jnp.array([jnp.sin(q[0]), jnp.cos(q[0]),
                             jnp.sin(q[1]), jnp.cos(q[1])])
        film_params = model.film_net(p_norm).reshape(model.n_hidden, 2)
        chol = model.compute_cholesky_entries(trig_q, film_params)
        L = jnp.array([[jax.nn.softplus(chol[0]), 0.0],
                        [chol[1],                  jax.nn.softplus(chol[2])]])
        M = L.T @ L + jnp.eye(2) * 1e-6
        return 0.5 * q_dot_norm @ M @ q_dot_norm
    
    def gt_T(q1, q2):
        q = jnp.array([q1, q2])
        return dp.kinetic_energy(q, q_dot_test)
    
    T_model = jax.vmap(jax.vmap(model_T))(Q1, Q2)
    T_gt    = jax.vmap(jax.vmap(gt_T))(Q1, Q2)
    
    T_model_aligned = (T_model - jnp.mean(T_model)) / (jnp.std(T_model) + 1e-8)
    T_gt_aligned    = (T_gt    - jnp.mean(T_gt))    / (jnp.std(T_gt)    + 1e-8)
    
    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'm1={p_phys[0]:.2f}, m2={p_phys[1]:.2f}, l1={p_phys[2]:.2f}, l2={p_phys[3]:.2f}; q_dot={q_dot_test}')
    
    vmin_V = min(V_gt_aligned.min(), V_model_aligned.min())
    vmax_V = max(V_gt_aligned.max(), V_model_aligned.max())
    vmin_T = min(T_gt_aligned.min(), T_model_aligned.min())
    vmax_T = max(T_gt_aligned.max(), T_model_aligned.max())
    
    # V plots
    im = axes[0, 0].contourf(Q1, Q2, V_gt_aligned, levels=30, vmin=vmin_V, vmax=vmax_V)
    axes[0, 0].set_title('V ground truth (normalized)')
    axes[0, 0].set_xlabel('q1'); axes[0, 0].set_ylabel('q2')
    plt.colorbar(im, ax=axes[0, 0])
    
    im = axes[0, 1].contourf(Q1, Q2, V_model_aligned, levels=30, vmin=vmin_V, vmax=vmax_V)
    axes[0, 1].set_title('V model (normalized)')
    axes[0, 1].set_xlabel('q1'); axes[0, 1].set_ylabel('q2')
    plt.colorbar(im, ax=axes[0, 1])
    
    im = axes[0, 2].contourf(Q1, Q2, jnp.abs(V_model_aligned - V_gt_aligned), levels=30)
    axes[0, 2].set_title('|V model - V gt|')
    axes[0, 2].set_xlabel('q1'); axes[0, 2].set_ylabel('q2')
    plt.colorbar(im, ax=axes[0, 2])
    
    # T plots
    im = axes[1, 0].contourf(Q1, Q2, T_gt_aligned, levels=30, vmin=vmin_T, vmax=vmax_T)
    axes[1, 0].set_title('T ground truth (normalized)')
    axes[1, 0].set_xlabel('q1'); axes[1, 0].set_ylabel('q2')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].contourf(Q1, Q2, T_model_aligned, levels=30, vmin=vmin_T, vmax=vmax_T)
    axes[1, 1].set_title('T model (normalized)')
    axes[1, 1].set_xlabel('q1'); axes[1, 1].set_ylabel('q2')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].contourf(Q1, Q2, jnp.abs(T_model_aligned - T_gt_aligned), levels=30)
    axes[1, 2].set_title('|T model - T gt|')
    axes[1, 2].set_xlabel('q1'); axes[1, 2].set_ylabel('q2')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':

    print('Kinetic-Potential Energy Decomposition Validation')

    # MODEL LOADING
    # ===================
    pos_dim = 2
    vel_dim = 2
    param_dim = 4
    
    key = jax.random.PRNGKey(123)
    model_key, _ = jax.random.split(key) # Only need model_key here

    # Dynamically find the latest model (assuming naming convention `best_model_YYYYMMDD_HHMMSS.eqx`)
    model_name = str(MODEL_DIR) + '/' + f"model_T512_20260317_133032"
    # model_name = str(MODEL_DIR) + '/' + f"model_T512_20260317_164201"

    # Instantiate the model architecture
    model = LagrangianNN(
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        param_dim=param_dim,
        hidden_dim=128,
        n_hidden=2,
        key=model_key # Use the same key for initial instantiation
    )
    # Load the trained parameters into the model
    # load_model expects the stem (filename without extension)
    model = load_model(model, model_name) 
    
    # NORM STATS LOADING
    # ===================
    # Assume norm_stats are saved in an .npz file with the same stem as the model
    with np.load(model_name+'.npz', allow_pickle=True) as data:
        norm_stats = data['norm_stats'].item()

    # VALIDATION SCENARIOS
    # ====================
    # Define a set of physical parameters to test (e.g., from training set or OOD)
    test_params_cases = [
        jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64), # Default
        jnp.array([1.5, 0.8, 1.2, 1.0], dtype=jnp.float64), # Example from training range
        jnp.array([2.5, 1.5, 1.2, 1.3], dtype=jnp.float64), # OOD example (heavy masses)
    ]

    for i, params_phys in enumerate(test_params_cases):
        print(f"\nGenerating plots for parameter set {i+1}: {params_phys}")
        fig = plot_TV_decomposition(model, norm_stats, params_phys)
        # Save the plot
        plot_filename = KPD_RESULTS_DIR / f"{model_name}_kpd_case_{i}.png"
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig) # Close figure to free memory

    print('Kinetic-Potential Energy Decomposition Validation Complete.')
    # plt.show() # Uncomment to display plots interactively