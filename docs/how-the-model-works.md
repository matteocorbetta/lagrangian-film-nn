# How The Model Works

This page is the practical core of the docs. It collects the architecture choices, the conditioning logic, the loss design, and the current workflow in one place.

## Architecture Overview

The model is split into three pieces:

1. `kinetic_net`
2. `potential_net`
3. `film_net`

Together they define a structured normalized Lagrangian:

$$L = T - V$$

where:

- `T` is a normalized kinetic term
- `V` is a normalized potential term
- accelerations are recovered by differentiating `L`

## Input Transformation

### Trigonometric Angle Features
Angles are converted into their trigonometric representaion with `sin`,`cos`:

$$[q_1, q_2] \rightarrow [\sin(q_1), \cos(q_2), \sin(q_2), \cos(q_2)] $$

before being fed into the model. This avoids the discontinuity at angle wraparound and gives the network periodic features directly.

### Velocities and System Parameters
Angular velocities and system parameters (blob masses and rod lengths) are scaled using mean and standard deviation in a typical Normal-scaling fashion.

## Kinetic Branch

The kinetic branch is the most structured part of the model.

Its job is to define the matrix used in the normalized kinetic energy term. Instead of predicting that matrix directly, the network predicts Cholesky-style entries and builds the final matrix from them. 

$$C_e = f([\sin(q_1), \cos(q_1), \sin(q_2), \cos(q_2)], \boldsymbol{\theta})$$

where: $C_e$ refers to 'Choleksy-entries' and $f(\cdot)$ is the FiLM-parameterized kinertic network. FiLM parameterization is applied by looping over an MLP layer and applying the FiLM parameters $\alpha,\,\beta$ to each hidden layer output:

```python
for i in range(self.n_hidden):
    # Compute layer transformation
    h = net.layers[i](h)
    h = jax.nn.softplus(h)
    
    # FiLM scaling
    gamma = film_params[i, 0]
    beta = film_params[i, 1]

    h = gamma * h + beta
```
A final layer computes the output of $f(\cdot)$, and `softplus` actication is applied to 2 out of three outputs thus ensuring positive values, which enforeces the mass matrix to be positive-definite.
The mass matrix is computed as

$$M(\boldsymbol{q}) = \begin{bmatrix}
 \operatorname{softplus}(C_{e,0}) & 0 \\
    C_{e,1}                       & \operatorname{softplus}(C_{e,2})
\end{bmatrix}$$

Velocities are used to compute the kinetic term in canonical form:

$$T = \frac{1}{2}\, \dot{\boldsymbol{q}}^{\top} \, M(\boldsymbol{q}) \, \dot{\boldsymbol{q}}$$

### Why This Matters

The kinetic term should depend on a positive-definite matrix. An unconstrained network has no reason to produce one. By building the matrix from a triangular factor with positive diagonal entries, the model makes that property intrinsic to the architecture instead of hoping optimization finds it.

### Important Note on FiLM
The FiLM I applied her is per-layer, and not per-neuron as in the original idea. This was simply to keep the model small. Better results may probably be achieved using a per-neuron FiLM as this would allow for more expressivity and better representation of the nonlinear interaction of masses and lengths.

## Potential Branch
The potential branch just takes:

- trigonometric angle features
- normalized physical parameters

and outputs a scalar normalized potential energy.

That split turned out to be a stable compromise between mechanical structure and training simplicity.

## Why FiLM Is Used Only On The Kinetic Branch

This is one of the main design choices in the repo.

The reasoning is:

- the kinetic term is where inertia structure lives
- changing masses and rod lengths changes that inertia structure strongly
- FiLM is a good way to modulate hidden features without redesigning the whole branch

So the kinetic branch is conditioned through FiLM, while the potential branch stays simpler.

## How Accelerations Are Computed

As already said, the model does not output accelerations directly. Instead it:

- computes the normalized Lagrangian
- differentiates it with JAX
- solves the resulting Euler-Lagrange system for `q_tt`

That is the main mechanics trick.

## Loss Function

The default training loss combines:

1. Huber loss on normalized accelerations
2. normalized-energy conservation loss

The first term is the direct supervised objective (error on the accelerations).

The second term penalizes drift in the model's normalized Hamiltonian over a trajectory chunk. One nice thing about this regularizer is that it is generic: it uses the model's own structured energy decomposition rather than analytical accelerations in the loss itself.

## Training Notes

The current setup is intentionally simple:

- trajectories are generated analytically
- acceleration targets are estimated with `np.gradient`
- angles are left unnormalized for the trigonometric transformation
- velocities, parameters, and targets are normalized
- each optimization step samples one contiguous chunk from one trajectory

That last detail matters because the current normalized-energy regularizer is only truly trajectory-local under that batching scheme.

## Usage

The repo is still script-first rather than config-first.

The main workflow is:

1. generate data with `src/data/generate_dataset.py`
2. train with `src/train.py`
3. evaluate with `src/inference.py`
4. visualize saved results from `results/`

That keeps the project easy to inspect, even if it is not yet the cleanest possible reusable interface.

## Source Files To Read

- `src/lnn/model.py`
- `src/losses.py`
- `src/train.py`
- `src/train_utils.py`
- `src/inference.py`
