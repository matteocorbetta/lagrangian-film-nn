# Lagrangian-FiLM NN

This repository studies double-pendulum dynamics with a FiLM-conditioned Lagrangian neural network implemented in JAX and Equinox. The model learns a structured Lagrangian from trajectories and physical parameters, then derives accelerations through automatic differentiation of the learned mechanics model.

## What This Project Does

The core model takes generalized coordinates, generalized velocities, and physical parameters as input:

- State: `[q1, q2, w1, w2]`
- Augmented model input: `[q1, q2, w1, w2, m1, m2, l1, l2]`
- Output: generalized accelerations `[q1_tt, q2_tt]`

The current implementation is designed around a double pendulum with variable masses and rod lengths. Training data is generated analytically, the model is trained on normalized accelerations, and evaluation is done with explicit trajectory rollouts and energy diagnostics.

## Current Scope

This codebase currently focuses on one experimental workflow:

- generating synthetic double-pendulum trajectories,
- training a FiLM-conditioned Lagrangian network,
- evaluating held-out and out-of-distribution rollouts,
- visualizing trajectory and energy behavior.

The documentation reflects the repository as it exists today. It documents current behavior and limitations rather than presenting the code as a polished general-purpose package.

## Repository Map

- `src/data/`: analytical double-pendulum dynamics, trajectory generation, HDF5 I/O
- `src/lnn/`: `LagrangianNN` model definition
- `src/train.py`: training entrypoint and optimization loop
- `src/inference.py`: rollout generation, energy plots, OOD evaluation
- `src/simulate.py`: RK4 rollout utilities and rollout-data persistence
- `results/`: postprocessing, animations, and saved evaluation artifacts
- `docs/`: MkDocs site content

## Documentation Guide

- [Installation](installation.md): environment setup, dependencies, and verification
- [Theory](theory.md): the modeling assumptions and structured Lagrangian design
- [Architecture](architecture.md): module map, data flow, shapes, and saved artifacts
- [Usage](usage.md): end-to-end workflow from dataset generation to evaluation
- [Examples](examples.md): scenario-based walkthroughs for common tasks
- [API](api.md): reusable modules and generated reference docs

## Main Entry Points

If you are new to the repository, the fastest path is:

1. Read [Theory](theory.md) to understand what the model is learning.
2. Read [Architecture](architecture.md) for the end-to-end pipeline and data contracts.
3. Use [Usage](usage.md) to run dataset generation, training, inference, and visualization scripts.

Relevant source files:

- `src/lnn/model.py`
- `src/train.py`
- `src/inference.py`
