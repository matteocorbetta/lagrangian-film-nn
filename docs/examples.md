# Examples

This page gives scenario-based examples for the current double-pendulum workflow. These examples are intended to show what to run, what artifacts to expect, and how to interpret the results.

## Example 1: Train On The Default Double-Pendulum Dataset

### Objective

Train the FiLM-conditioned Lagrangian model on the repository's default synthetic dataset workflow.

### Command

```bash
uv run python src/train.py
```

### Expected Inputs

- an HDF5 dataset under `data/doublependulum/`,
- trajectories shaped as `[T, 5] = [time, q1, q2, w1, w2]`,
- parameter vectors `[m1, m2, l1, l2]`.

### Expected Outputs

- serialized model weights in `src/models/*.eqx`
- metadata and normalization information in `src/models/*.npz`
- training diagnostics such as saved plots if the script is configured to emit them

### Interpretation Notes

The model is trained to predict normalized accelerations while respecting a structured learned Lagrangian. A lower loss does not only reflect regression quality; it also includes the energy-conservation regularizer.

### Limitations

- Hyperparameters are embedded directly in `src/train.py`.
- Temporal chunk sampling is fixed by the current batching logic.
- The implementation is specialized to the double pendulum.

## Example 2: Roll Out A Trained Model On Held-Out Trajectories

### Objective

Compare ground-truth trajectories with model-generated rollouts on held-out examples.

### Command

```bash
uv run python src/inference.py
```

### Expected Outputs

- rollout `.npz` files in `results/rollouts/`
- comparison plots for selected trajectories
- test-set energy plots in the results tree

### Interpretation Notes

Compare:

- angle traces,
- angular velocity traces,
- qualitative drift over long rollouts,
- whether rollout errors remain bounded on held-out trajectories.

### Limitations

`src/inference.py` currently references a hard-coded model stem. Update that value before running against a different checkpoint.

## Example 3: Run Out-Of-Distribution Parameter Tests

### Objective

Evaluate how the learned model behaves on pendulum parameters outside the training distribution.

### Command

```bash
uv run python src/inference.py
```

### Expected Outputs

- OOD rollout `.npz` files in `results/ood_tests/`
- OOD comparison plots for the predefined cases

### Current OOD Cases

The script currently defines manually chosen examples such as:

- heavier masses with in-range lengths,
- in-range masses with longer rods,
- both masses and lengths pushed out of range,
- smaller masses and shorter rods than training.

### Interpretation Notes

In this repository, "OOD" means parameter combinations outside the training distribution for `[m1, m2, l1, l2]`, not new state-space coordinates with the same parameter law.

### Limitations

- OOD scenarios are hard-coded in `src/inference.py`.
- Success here indicates some robustness, not guaranteed physical generalization.

## Example 4: Inspect Learned Energy Behavior

### Objective

Use the rollout energy plots to inspect whether the learned structured dynamics remain approximately conservative over time.

### Command

```bash
uv run python src/inference.py
```

### Expected Outputs

- saved energy-drift plots in the rollout results area

### Interpretation Notes

The plotted quantity is the relative drift of the model's learned normalized Hamiltonian. Low drift is usually a good sign for trajectory consistency, but it does not prove that the model has recovered the exact physical Hamiltonian in original units.

### Limitations

- the quantity is normalized and model-induced,
- low drift can coexist with systematic state prediction error.

## Example 5: Compare Learned And Analytical Kinetic/Potential Structure

### Objective

Inspect whether the learned decomposition into kinetic and potential terms resembles the analytical system.

### Command

```bash
uv run python src/energy_validation/kinpot_decomposition.py
```

### Expected Outputs

- contour plots in `results/kinpot_decomposition/`

### Interpretation Notes

The script aligns model and analytical contours by normalization before plotting, so the goal is shape agreement rather than equality of absolute scale. This is especially important for the potential term, which is defined only up to an additive constant.

### Limitations

- fixed representative velocity is used for kinetic-energy comparisons,
- the script is tied to a hard-coded model name and current 2-DoF assumptions.
