# Background

## Problem Setup

The repository models a parameterized double pendulum. Each trajectory is defined by:

- generalized coordinates, that is the angular positions of the pendula $\boldsymbol{q} = [q_1, q_2]^{\top}$ $\rightarrow$ `q = [q1, q2]`,
- generalized velocities, that is the angular velocities of the pendula $\dot{\boldsymbol{q}} = [\dot{q}_1, \dot{q}_2]^{\top}$ $\rightarrow$ `qdot = [w1, w2]`,
- physical parameters $\boldsymbol{\theta} = [m_1, m_2, l_1, l_2]^{\top}$ $\rightarrow$ `p = [m1, m2, l1, l2]`.

The learning objective is to predict generalized accelerations $\ddot{\boldsymbol{q}} = [\ddot{q}_1, \ddot{q}_2]^{\top}$ from state and parameters while preserving as much mechanical structure as possible.

In the training pipeline, each input timestep is represented as:

```text
[q1, q2, w1, w2, m1, m2, l1, l2]
```

and each target timestep is:

```text
[q1_tt, q2_tt]
```

## The Lagrangian and Its Derivative

The model learns a structured Lagrangian

$$L(\boldsymbol{q}, \dot{\boldsymbol{q}}, \boldsymbol{\theta}) = T(\boldsymbol{q}, \dot{\boldsymbol{q}}, \boldsymbol{\theta}) - V(\boldsymbol{q}, \boldsymbol{\theta})$$

and uses automatic differentiation to recover accelerations through the Euler-Lagrange equations. In the code, this happens inside `LagrangianNN.__call__`, where gradients and Jacobians of the learned Lagrangian are used to solve for `q_tt`.

This can be achieved using the chain rule to obtain $\ddot{q}$ explicitly as in [Cranmer et al., Eq. (6)](https://arxiv.org/pdf/2003.04630):

$$\ddot{\boldsymbol{q}} = (\nabla_{\dot{\boldsymbol{q}}}\,\nabla_{\dot{\boldsymbol{q}}}^{\top}\,L )^{-1}\, [ \nabla_{\boldsymbol{q}}\,L  - (\nabla_{\boldsymbol{q}}\,\nabla_{\dot{\boldsymbol{q}}}^{\top}\, L)\,\dot{\boldsymbol{q}}] $$

The forward pass therefore does not directly regress accelerations with an unconstrained network as in $(\boldsymbol{q}, \boldsymbol{\theta}) \rightarrow \ddot{q}$. Instead, it builds a differentiable mechanics model and computes accelerations from that model.

That gives the architecture more structure and makes it easier to inject mechanical priors into the network design.
The fact that the model is built in `JAX` allows you to take the Lagrangian and perform the necessary partial derivatives w.r.t. $q,\, \dot{q}$ naturally.

_Note_: with an abuse of notation, we define the system state vector composed of two positions and two velocities as $\boldsymbol{q}$, which should not be confused with the 2-element array of positions $q_1$, $q_2$.


## What I Took From Prior Work

Conceptually, the repo borrows:

- from Lagrangian Neural Networks (Cranmer et al., 2020): learn a Lagrangian and derive accelerations by autodiff
- from structured Lagrangian modeling in the DeLaN style (Lutter et al., 2019): split kinetic and potential terms, and enforce a well-behaved matrix structure for the kinetic part

The extra twist in this repo is parameter conditioning through FiLM on the kinetic branch.

## Normalized Coordinates

One practical framing detail matters for the rest of the docs:

- angles stay in their natural angular form
- velocities and physical parameters are normalized for training
- acceleration targets are normalized as well

So the repo talks about _normalized_ kinetic energy, _normalized_ potential energy, _normalized_ Lagrangian, and _normalized_ Hamiltonian.

The structure is still mechanical, but the learned energies are not being claimed as exact physical energies in original units.

For the implementation details, go to [How The Model Works](how-the-model-works.md).

## References

Interesting Reads:

- [Deep Lagrangian Networks (Lutter et al., ICLR 2019)](https://arxiv.org/abs/1907.04490). [Code](https://github.com/milutter/deep_lagrangian_networks)
- [Lagrangian Neural Networks (Cranmer et al., 2020)](https://arxiv.org/abs/2003.04630)

More Advanced Implementaions: 

- [Context-Aware DeLaN / CaDeLaC (Schulze et al., 2025)](https://arxiv.org/abs/2506.15249)
- [Floating-Base DeLaN / FeLaN (Schulze et al., 2025)](https://arxiv.org/abs/2510.17270)



