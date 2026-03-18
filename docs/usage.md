# Usage

This page describes the operational workflow for the repository: generate data, train a model, evaluate rollouts, visualize outputs, and run structural validation checks.

All commands below assume you are running from the repository root. If you use `uv`, prefer `uv run python ...`; if your environment is already activated, plain `python ...` is equivalent.

## 1. Generate A Dataset

The dataset generation script is `src/data/generate_dataset.py`.

It currently:

- samples double-pendulum masses and rod lengths,
- samples initial states,
- integrates analytical dynamics with `odeint`,
- writes trajectories and parameter vectors to an HDF5 file.

Documented command:

```bash
uv run python src/data/generate_dataset.py
```

Equivalent plain command:

```bash
python src/data/generate_dataset.py
```

### What It Writes

The script writes an HDF5 file under `data/doublependulum/` using the project-relative path helper in `src/data/utils.py`.

The generated dataset contains:

- `trajectory_*` datasets with shape `[T, 5] = [time, q1, q2, w1, w2]`
- `params_*` datasets with shape `[4] = [m1, m2, l1, l2]`

### Current Sampling Note

The script contains a commented-out low-energy rejection check. That means the current generated dataset does not strictly enforce the low-energy filter described in comments; it follows the code as written.

## 2. Train A Model

The main training entrypoint is `src/train.py`.

Documented command:

```bash
uv run python src/train.py
```

Equivalent plain command:

```bash
python src/train.py
```

### What The Training Script Does

`src/train.py`:

- loads the HDF5 dataset,
- converts raw trajectories into supervised tensors,
- splits trajectories into train, validation, and test sets,
- normalizes inputs and acceleration targets,
- samples temporal chunks during optimization,
- trains `LagrangianNN` with early stopping,
- saves model parameters and metadata.

### Major Training Hyperparameters

The module-level defaults are currently:

- `BATCH_SIZE = 512`
- `PATIENCE = 50`
- `LEARNING_RATE = 3e-3`
- `STEPS = 50000`
- `PRINT_EVERY = 100`
- `EVAL_EVERY = 300`
- `SEED = 5678`

### Important Training Detail

`BATCH_SIZE` is currently reused as the temporal chunk length, while each step samples a batch of one trajectory chunk. That is an implementation detail worth knowing when reading loss curves or adjusting scaling behavior.

### Saved Outputs

The training script writes model artifacts under `src/models/`, including serialized Equinox weights and associated metadata such as normalization statistics.

## 3. Run Inference And Evaluations

The main evaluation script is `src/inference.py`.

Documented command:

```bash
uv run python src/inference.py
```

Equivalent plain command:

```bash
python src/inference.py
```

### What It Does

`src/inference.py` currently performs several tasks:

- generates held-out train/test rollouts,
- saves rollout `.npz` files,
- plots relative drift of the learned normalized Hamiltonian,
- evaluates manually specified OOD parameter cases,
- saves OOD result arrays and plots.

### Current Practical Limitation

The script uses a hard-coded model name:

```text
model_T512_20260317_133032
```

Before running evaluation on a different checkpoint, update the embedded model reference and ensure the corresponding `.eqx` and `.npz` files exist in `src/models/`.

## 4. Visualize Saved Results

The visualization entrypoint is `results/visualization.py`.

Documented command:

```bash
uv run python results/visualization.py
```

Equivalent plain command:

```bash
python results/visualization.py
```

### What It Produces

From saved rollout `.npz` files, the script can generate:

- comparison GIFs between ground truth and model trajectories,
- pendulum animations with phase plots,
- static phase portraits,
- trajectory-on-potential animations.

The plotting helpers live in `results/visualization_utils.py`.

## 5. Validate Kinetic/Potential Decomposition

The structural validation script is `src/energy_validation/kinpot_decomposition.py`.

Documented command:

```bash
uv run python src/energy_validation/kinpot_decomposition.py
```

Equivalent plain command:

```bash
python src/energy_validation/kinpot_decomposition.py
```

### What It Checks

The script compares learned and analytical structure over a grid of configurations:

- normalized potential energy contours,
- normalized kinetic energy contours at a fixed representative velocity,
- absolute contour differences between model and analytical quantities.

Like `src/inference.py`, it currently relies on a hard-coded model stem and expects corresponding normalization stats in a sibling `.npz` file.

## Practical Notes

- Run commands from the repository root unless you are deliberately adjusting import paths.
- The codebase is script-first, so many workflows are configured by editing constants inside the scripts.
- `src/inference.py` and `src/energy_validation/kinpot_decomposition.py` contain hard-coded example model names.
- API-level helpers exist, but the primary workflow is still organized around executable scripts.
