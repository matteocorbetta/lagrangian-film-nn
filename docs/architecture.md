# Architecture

This page describes the end-to-end repository pipeline, the role of each module, the main data interfaces, and the artifacts produced by training and evaluation.

## End-To-End Flow

The current workflow is:

1. Generate analytical double-pendulum trajectories.
2. Save trajectories and physical parameters to HDF5.
3. Load trajectories and build supervised tensors.
4. Normalize inputs and acceleration targets.
5. Train the FiLM-conditioned Lagrangian neural network.
6. Save model parameters, normalization statistics, and training outputs.
7. Run held-out rollouts, energy diagnostics, and out-of-distribution tests.
8. Generate plots and animations from saved evaluation artifacts.

## Pipeline Overview

### 1. Data generation

`src/data/generate_dataset.py` samples:

- initial angular positions and velocities,
- masses and rod lengths for each trajectory.

It integrates the analytical double-pendulum dynamics with `jax.experimental.ode.odeint` and writes each trajectory plus its parameter vector into an HDF5 file under `data/<system>/`.

### 2. Dataset loading and preprocessing

`src/data/utils.py` loads the saved HDF5 file. `train_utils.build_input_output` then converts raw trajectories into:

- augmented inputs `[q1, q2, w1, w2, m1, m2, l1, l2]`,
- target accelerations estimated numerically from the velocity channels.

### 3. Training preparation

`train_utils.train_test_split` creates train, validation, and test trajectory splits. `train_utils.normalize_data` computes training-set statistics and normalizes:

- velocities and parameters in the input tensor,
- acceleration targets.

Angles are intentionally left unnormalized.

### 4. Model training

`src/train.py` defines:

- the loss function,
- the optimizer and scheduler,
- the early-stopping training loop,
- model saving and evaluation bookkeeping.

Each optimization step samples one temporal chunk from one trajectory through `train_utils.build_temporal_batch`.

### 5. Simulation and evaluation

`src/simulate.py` builds an RK4 rollout function around the learned model. `src/inference.py` uses it to:

- evaluate held-out trajectories,
- plot relative drift in learned normalized energy,
- generate OOD rollouts for manually specified parameter sets,
- save rollout arrays to compressed `.npz` files.

### 6. Visualization and validation

`results/visualization.py` and `results/visualization_utils.py` convert saved rollout results into:

- comparison GIFs,
- pendulum animations,
- phase portraits,
- potential-landscape trajectory animations.

`src/energy_validation/kinpot_decomposition.py` produces contour plots comparing the learned kinetic/potential decomposition against analytical structure.

## Module Map

### `src/data/doublependulum.py`

Analytical double-pendulum system definition:

- energy functions,
- coordinate conversion,
- analytical state transition,
- sampling helpers for initial conditions and physical parameters.

### `src/data/utils.py`

Dataset I/O utilities:

- project-relative data path construction,
- HDF5 save/load helpers for trajectory collections.

### `src/data/generate_dataset.py`

Script for generating the double-pendulum dataset and writing it to disk.

### `src/lnn/model.py`

Defines `LagrangianNN`, the core FiLM-conditioned Lagrangian model.

### `src/losses.py`

Training regularizers and analytical comparison losses:

- energy conservation loss,
- kinetic structure loss,
- potential structure loss.

### `src/train_utils.py`

Training support functions:

- persistence helpers,
- preprocessing,
- normalization,
- splitting,
- temporal batching,
- diagnostics.

### `src/train.py`

Training entrypoint and optimization loop.

### `src/simulate.py`

RK4 rollout utilities and rollout-data persistence.

### `src/inference.py`

Evaluation script for:

- held-out rollouts,
- energy drift plots,
- OOD parameter tests.

### `src/energy_validation/kinpot_decomposition.py`

Validation script for learned kinetic/potential structure.

### `results/visualization.py`

Loads saved rollout `.npz` files and writes plots and GIFs.

### `results/visualization_utils.py`

Low-level animation and plotting helpers used by `results/visualization.py`.

## Data Shapes And Interfaces

### Raw trajectory format

Each raw saved trajectory has shape:

```text
[T, 5] = [time, q1, q2, w1, w2]
```

### Parameter vector

Each trajectory is associated with:

```text
[4] = [m1, m2, l1, l2]
```

### Supervised model input

After preprocessing, each timestep becomes:

```text
[q1, q2, w1, w2, m1, m2, l1, l2]
```

and the resulting tensor has shape:

```text
[num_trajectories, T, 8]
```

### Training targets

Targets are generalized accelerations with shape:

```text
[num_trajectories, T, 2]
```

### Rollout state

Two rollout representations are used:

- physical state only: `[q1, q2, w1, w2]`
- full rollout state with attached normalized parameters: `[q1, q2, w1, w2, p_norm...]`

In practice, the full state used by `make_rollout` has length `8` for this repository.

## Saved Artifacts

### Model outputs

Training and evaluation scripts write:

- `.eqx`: serialized Equinox model leaves
- `.npz`: model metadata such as normalization statistics and rollout outputs
- `.png`: plots such as loss curves, energy drift, or decomposition contours
- `.gif`: animated comparisons and phase-space visualizations

### Result directories

The repository currently uses:

- `src/models/` for saved trained models and associated metadata
- `results/rollouts/` for held-out rollout arrays and plots
- `results/ood_tests/` for OOD rollout arrays and plots
- `results/kinpot_decomposition/` for contour comparisons
- `results/sample_viz/` for generated GIFs and phase portraits

## HDF5 Dataset Schema

The saved dataset is organized under a `trajectories` group. Inside that group, the generation and loading code expect entries of the form:

- `trajectory_*`
- `params_*`

This schema is part of the data contract between `src/data/generate_dataset.py` and `src/data/utils.py`.

## Known Architectural Constraints

Several limitations are visible directly in the implementation:

- Multiple modules are hard-coded for the 2-DoF double-pendulum case.
- The repository is script-first rather than a packaged CLI application.
- Many scripts assume they are run from the repository root.
- Some evaluation scripts contain hard-coded model filenames and example parameter sets.
- `mkdocstrings` API rendering depends on MkDocs being configured to import modules from `src/`.
