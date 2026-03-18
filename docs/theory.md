# Theory

This page explains the modeling idea implemented in the repository and connects the mathematical structure to the code in `src/lnn/model.py`, `src/losses.py`, and `src/train_utils.py`.

## Problem Setup

The repository models a parameterized double pendulum. Each trajectory is defined by:

- generalized coordinates `q = [q1, q2]`,
- generalized velocities `qdot = [w1, w2]`,
- physical parameters `p = [m1, m2, l1, l2]`.

The learning objective is to predict generalized accelerations from state and parameters while preserving as much mechanical structure as possible.

In the training pipeline, each input timestep is represented as:

```text
[q1, q2, w1, w2, m1, m2, l1, l2]
```

and each target timestep is:

```text
[q1_tt, q2_tt]
```

## Learned Lagrangian Formulation

The model learns a structured Lagrangian

\\[
L(q, \dot{q}, p) = T(q, \dot{q}, p) - V(q, p)
\\]

and uses automatic differentiation to recover accelerations through the Euler-Lagrange equations. In the code, this happens inside `LagrangianNN.__call__`, where gradients and Jacobians of the learned Lagrangian are used to solve for `q_tt`.

The forward pass therefore does not directly regress accelerations with an unconstrained network. Instead, it builds a differentiable mechanics model and computes accelerations from that model.

## Structured Model Design

### Trigonometric embedding of angles

Angles are converted to trigonometric features:

```text
[sin(q1), cos(q1), sin(q2), cos(q2)]
```

This avoids the discontinuity at angular wraparound and makes the representation periodic by construction.

### Kinetic energy branch

The kinetic term is represented through a positive-definite matrix built from Cholesky-style entries predicted by the network. Concretely:

- the kinetic branch receives only trigonometric angle features,
- its hidden activations are modulated by FiLM parameters generated from the physical parameters,
- the output is converted into a lower-triangular matrix `L`,
- the effective mass matrix is formed as `M = L^T L + eps I`.

This guarantees a positive-definite quadratic form for the learned kinetic energy.

### Potential energy branch

The potential branch receives:

- trigonometric angle features,
- normalized physical parameters.

It outputs a scalar learned potential energy.

### FiLM conditioning

The kinetic branch is conditioned on system parameters through a separate FiLM network. For each hidden layer, the FiLM network predicts scale and shift values:

- `gamma`
- `beta`

These are applied to the hidden activations of the kinetic branch. This lets the model adapt its learned inertia structure across different masses and lengths without concatenating parameters directly into the kinetic input.

## Normalization Strategy

The normalization contract is defined in `train_utils.normalize_data`.

### Inputs

- Angles `q1`, `q2` are passed through unchanged.
- Velocities and physical parameters are normalized using training-set statistics.

### Targets

- Target accelerations are normalized to zero mean and unit variance using training-set statistics.

This means the model operates on a mixed representation:

- angles stay in physical angular units,
- velocities and parameters are normalized,
- predicted accelerations are produced in normalized target space and then rescaled for physical rollouts.

## Losses

The main training loss in `src/train.py` combines two components.

### 1. Numerical prediction loss

The primary term is a Huber loss on normalized accelerations:

- predicted `q_tt` from the model,
- normalized ground-truth accelerations from finite-difference preprocessing.

### 2. Energy-conservation regularization

`losses.energy_conservation_loss` penalizes variance in the model's learned Hamiltonian over a temporal chunk. In the current setup, each training batch corresponds to one temporal chunk sampled from one trajectory, so this variance acts as a trajectory-local regularizer.

This is a regularizer on the learned structured normalized energy, not a guarantee of exact physical energy conservation in original units.

### Auxiliary losses present in the codebase

`src/losses.py` also contains:

- `kinetic_loss`
- `potential_loss`

These compare learned structure against analytical quantities, but they are not part of the default training path in `src/train.py`.

## Limits And Caveats

The current implementation has several important constraints:

- The code is effectively hard-coded for a 2-DoF system in multiple places.
- The learned energy during training is a normalized, model-induced quantity, not an exact physical Hamiltonian in original units.
- The current batching logic assumes one trajectory chunk per optimization step.
- The model architecture and rollout code are written for the double-pendulum state layout `[q1, q2, w1, w2]`.

These are implementation facts, not just documentation choices.

## References

The repository cites the following related work.

### DeLaN

- Deep Lagrangian Networks (Lutter et al., ICLR 2019): <https://arxiv.org/abs/1907.04490>
- Reference code: <https://github.com/milutter/deep_lagrangian_networks>

### Energy-based control extension

- DeLaN for energy-based control (Lutter et al., IROS 2019): <https://arxiv.org/abs/1907.04489>

### Lagrangian Neural Networks

- Lagrangian Neural Networks (Cranmer et al., 2020): <https://arxiv.org/abs/2003.04630>

### Context-aware and floating-base follow-ups

- Context-Aware DeLaN / CaDeLaC (Schulze et al., 2025): <https://arxiv.org/abs/2506.15249>
- Floating-Base DeLaN / FeLaN (Schulze et al., 2025): <https://arxiv.org/abs/2510.17270>
