# Lagrangian-FiLM NN

Parameter-conditioned Lagrangian neural networks for double-pendulum dynamics in `JAX/Equinox`.

This repository learns a structured mechanics model from simulated trajectories of a double pendulum with varying masses and rod lengths. The model combines:

- a learned Lagrangian formulation,
- a Feature-wise Linear Modulation (FiLM)-conditioned kinetic branch,
- a learned potential branch,
- automatic differentiation of the Euler-Lagrange equations,
- rollout-based evaluation on held-out and out-of-distribution parameter settings.

The current codebase is experimental and focused on one concrete system: a 2-DoF double pendulum.

**Blind test of ground truth vs. model given same initial conditions**:
![Held-out comparison](results/sample_viz/indist_comparison_0.gif)

## What The Model Learns

The core network takes:

- generalized coordinates and velocities: `[q1, q2, w1, w2]`
- physical parameters: `[m1, m2, l1, l2]`

and predicts:

- generalized accelerations: `[q1_tt, q2_tt]`

Internally, it learns a structured Lagrangian

```text
L(q, qdot, p) = T(q, qdot, p) - V(q, p)
```

where:

- the kinetic term is built from a positive-definite matrix parameterization,
- the kinetic branch is FiLM-conditioned by the physical parameters,
- the potential branch depends on both configuration and parameters,
- accelerations are recovered by differentiating the learned Lagrangian rather than directly regressing dynamics with an unconstrained MLP.

## Repository Highlights

- `src/lnn/model.py`: FiLM-conditioned `LagrangianNN`
- `src/data/doublependulum.py`: analytical double-pendulum dynamics and energy functions
- `src/data/generate_dataset.py`: synthetic trajectory generation
- `src/train.py`: training loop and optimization
- `src/inference.py`: held-out rollouts, energy plots, and OOD tests
- `src/simulate.py`: RK4 rollout utilities
- `results/visualization.py`: GIF and phase-space visualization tools
- `docs/`: MkDocs site content

## Training And Evaluation Workflow

The current workflow is:

1. Generate analytical trajectories for double pendulums with sampled masses and lengths.
2. Build supervised tensors with augmented state `[q1, q2, w1, w2, m1, m2, l1, l2]`.
3. Normalize velocities, parameters, and acceleration targets.
4. Train `LagrangianNN` with Huber loss plus an energy-conservation regularizer.
5. Roll out the learned model with RK4.
6. Compare held-out and OOD trajectories, phase portraits, and learned energy drift.

## Sample Results

### Learned Phase Portraits

The model reproduces the overall phase-space structure reasonably well on in-distribution test trajectories.

![In-distribution phase portrait](results/sample_viz/indist_pendulum_phase_0.gif)

### Out-Of-Distribution Behavior

The repository also includes manual OOD tests over masses and rod lengths outside the training range.

![OOD phase portrait](results/sample_viz/ood_comparison_2.gif)

### Structural Validation

The codebase includes a kinetic/potential decomposition check that compares learned structure against the analytical system over a grid of configurations.

![Kinetic/potential decomposition](results/kinpot_decomposition/model_T512_20260317_133032_kpd_case_0_qd1.png)

## Quick Start

The repository currently implies `uv` as the preferred environment manager via `uv.lock`.

Install dependencies:

```bash
uv sync
```

Generate a dataset:

```bash
uv run python src/data/generate_dataset.py
```

Train a model:

```bash
uv run python src/train.py
```

Run inference, held-out rollouts, and OOD tests:

```bash
uv run python src/inference.py
```

Generate animations and phase plots from saved rollout artifacts:

```bash
uv run python results/visualization.py
```

Build the docs:

```bash
uv run mkdocs serve
```

## Current Limitations

- The implementation is specialized to a 2-DoF double pendulum.
- Several scripts rely on hard-coded model names and example parameter sets.
- The training code is script-first rather than a packaged CLI workflow.
- The learned energy regularizer acts on a normalized, model-induced quantity, not the exact physical Hamiltonian in original units.
- Some repository metadata still needs cleanup, such as the placeholder package name in `pyproject.toml`.


## Inspiration Papers for this work

### DeLaN

- Deep Lagrangian Networks (Lutter et al., ICLR 2019)
  - Paper: <https://arxiv.org/abs/1907.04490>
  - Code: <https://github.com/milutter/deep_lagrangian_networks>

<!-- ### Energy-Based Control Extension

- DeLaN for energy-based control (Lutter et al., IROS 2019)
  - Paper: <https://arxiv.org/abs/1907.04489> -->

### Lagrangian Neural Networks

- Lagrangian Neural Networks (Cranmer et al., 2020)
  - Paper: <https://arxiv.org/abs/2003.04630>

<!-- ### Context-Aware And Floating-Base Variants

- Context-Aware DeLaN / CaDeLaC (Schulze et al., 2025)
  - Paper: <https://arxiv.org/abs/2506.15249>
- Floating-Base DeLaN / FeLaN (Schulze et al., 2025)
  - Paper: <https://arxiv.org/abs/2510.17270> -->
