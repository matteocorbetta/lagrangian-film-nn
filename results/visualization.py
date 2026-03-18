import sys
from typing import Tuple
from pathlib import Path
import numpy as np

from visualization_utils import animate_comparison, plot_phase_portrait, animate_with_phase, animate_trajectory_on_V

# --- Dynamic path adjustment for imports ---
# Get the absolute path to the 'results' directory (where this script lives)
# Path(__file__).resolve().parent is 'project_root/results/'
RESULTS_DIR = Path(__file__).resolve().parent

# Move up one level to the project root: 'project_root/'
PROJECT_ROOT = RESULTS_DIR.parent

# Now, construct the path to the 'src' directory: 'project_root/src/'
SRC_DIR = PROJECT_ROOT / "src"

# Add 'src/' to Python's system path so it can find modules within 'src/'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ------------------------------------------


from data.doublependulum import DoublePendulum

def load_data_from_file(fname: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Loads ground truth and simulated trajectory data, along with parameters and dt, from an NPZ file.

    This function expects the NPZ file to contain specific keys:
    - 'ground_truth': The ground truth trajectory states.
    - 'simulated': The model's simulated trajectory states.
    - 'physical_parameters': The physical parameters of the system.
    - 'times': The time array from which 'dt' can be calculated.

    It includes a robust loading mechanism for 'physical_parameters' to handle
    cases where it might be saved directly as an array or as a dictionary within an object array.

    Args:
        fname (Path): The path to the `.npz` file containing the simulation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing:
            - gt_states (np.ndarray): Ground truth trajectory states, shape [T, 4].
            - model_states (np.ndarray): Model's simulated trajectory states, shape [T, 4].
            - params (np.ndarray): Physical parameters of the system, shape [4].
            - dt (float): The time step calculated from the 'times' array.
    """
    data = np.load(fname, allow_pickle=True)
    gt_states    = data['ground_truth']               # [T, 4]
    model_states = data['simulated']                  # [T, 4]
    
    try:
        params = data['physical_parameters']        # [4] = [m1, m2, l1, l2]
    except KeyError:
        params = data['physical_parameters'].item()        # [4] = [m1, m2, l1, l2]
        params = np.asarray(list(params.values()))

    dt = float(data['times'][1] - data['times'][0])
    return gt_states, model_states, params, dt


def animate_over_V(fname: Path, model_name: str):
    from lnn.model import LagrangianNN
    import jax
    key = jax.random.PRNGKey(123)
    model_key, _ = jax.random.split(key)
    model = LagrangianNN(pos_dim=2, vel_dim=2, hidden_dim=128, n_hidden=2, param_dim=4, key=model_key)

    with np.load(model_name, allow_pickle=True) as data:
        norm_stats = data['norm_stats'].item()

    _, model_states, params_phys, _ = load_data_from_file(fname)
    dp = DoublePendulum(m1=params_phys[0], m2=params_phys[1], l1=params_phys[2], l2=params_phys[3])

    anim = animate_trajectory_on_V(model_states, model, norm_stats, params_phys, dp, fps=30, speedup=5, n_grid=60)
    anim_fname = str(fname)[:str(fname).rfind('rollouts/')] + 'sample_viz/_traj_on_V.gif'
    anim.save(anim_fname, writer='pillow', fps=30, dpi=120)
    import matplotlib.pyplot as plt
    plt.close()
    print(f'Saved: {fname}')



def save_visualizations(fname: Path, index_: int):

    gt_states, model_states, params, dt = load_data_from_file(fname)

    
    title = f'$p_1 = (m_1, l_1)$ = ({params[0]:.2f}, {params[2]:.2f})\n$p_2 = (m_2, l_2)$ = ({params[1]:.2f}, {params[3]:.2f})'
    dp = DoublePendulum(m1=params[0], m2=params[1], l1=params[2], l2=params[3])
    str_fname = str(fname)
    if 'rollouts/' in str_fname:
        comp_fname = RESULTS_DIR / f'sample_viz/indist_comparison_{index_}.gif'
        phase_sim_fname = RESULTS_DIR / f'sample_viz/indist_pendulum_phase_{index_}.gif'
        phase_fname = RESULTS_DIR / f'sample_viz/indist_phase_{index_}.png'
    elif 'ood' in str_fname:
        comp_fname = RESULTS_DIR / f'sample_viz/ood_comparison_{index_}.gif'
        phase_sim_fname = RESULTS_DIR / f'sample_viz/ood_pendulum_phase_{index_}.gif'
        phase_fname = RESULTS_DIR / f'sample_viz/ood_phase_{index_}.png'
    else:
        comp_fname = RESULTS_DIR / f'sample_viz/comparison_{index_}.gif'
        phase_sim_fname = RESULTS_DIR / f'sample_viz/pendulum_phase_{index_}.gif'
        phase_fname = RESULTS_DIR / f'sample_viz/phase_{index_}.png'

    animate_comparison(
        gt_states, 
        model_states, 
        dp, 
        dt,
        title=title, 
        fname=comp_fname, 
        speedup=5,
        figsize=(9, 4))
    animate_with_phase(model_states, dp, dt, title=title, fname=phase_sim_fname, fps=30, speedup=5, trail_len=80)
    plot_phase_portrait(gt_states, model_states, title=title, fname=phase_fname)




if __name__ == '__main__':

    print("visualization tools")

    #  load saved results
    fname_form = 'rollouts/model_T512_20260317_133032_test_traj_{index}.npz'
    # fname_form = 'ood_tests/model_T512_20260317_133032_case_{index}.npz'

    for index in range(10):
        save_visualizations(RESULTS_DIR / fname_form.format(index=index), index_=index)

    
    