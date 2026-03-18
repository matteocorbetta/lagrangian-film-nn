# Learning the Lagrangian: Building a Parameter-Conditioned Neural Network for Double Pendulum Dynamics

## A technical story of building a physics-informed neural network from scratch, all the bugs along the way, and what actually worked.

---

## Overview

This project set out to answer a question: can a single neural network learn the equations of motion of a double pendulum across a family of different physical configurations — different masses, different rod lengths — and then simulate new configurations it has never seen?

The answer turned out to be yes, but getting there required building the architecture correctly from first principles, fixing a series of subtle and not-so-subtle bugs, and making several wrong turns before arriving at something that works.

The final model generalizes to pendulum configurations with masses up to 65% above the training maximum and rod lengths 40% above the training maximum, while conserving energy to within ~15% over 15-second rollouts on held-out test systems. This is a first-principles implementation of a parameter-conditioned Lagrangian Neural Network (LNN) trained entirely from trajectory data.

---

## Background and Motivation

### The Double Pendulum

The double pendulum is a canonical chaotic system. It consists of two pendulum bobs connected in series, governed by the Lagrangian:

```
L(q, q_dot, params) = T(q, q_dot, params) - V(q, params)
```

where:
- `T` is the kinetic energy: `T = 0.5*(m1+m2)*l1²*w1² + 0.5*m2*(l2²*w2² + 2*l1*l2*w1*w2*cos(t1-t2))`
- `V` is the potential energy: `V = -(m1+m2)*g*l1*cos(t1) - m2*g*l2*cos(t2)`
- `q = [t1, t2]` are the angles, `q_dot = [w1, w2]` are the angular velocities
- `params = [m1, m2, l1, l2]` are the physical parameters

The equations of motion follow from the Euler-Lagrange equations:

```
d/dt(dL/dq_dot) - dL/dq = 0
```

which gives: `M(q) * q_tt = f(q, q_dot)` where M is the mass matrix.

### Why a Neural Network?

The standard approach computes the EOM analytically. The neural network approach learns them from trajectory data — useful when the system is unknown or too complex to derive analytically. The physics-informed structure (Lagrangian formulation) imposes inductive biases that improve generalization and ensure physically meaningful behavior.

### Why Parameter Conditioning?

A naive LNN learns the dynamics of a single system with fixed masses and lengths. To generalize across a family of systems, the network must be conditioned on the physical parameters. We use FiLM (Feature-wise Linear Modulation) to condition the mass matrix network on the system parameters.

---

## Architecture

### High-Level Design

The model is split into three components:

1. **Kinetic net**: learns the Cholesky decomposition of the mass matrix `M(q, p)`
2. **Potential net**: learns the potential energy `V(q, p)`
3. **FiLM net**: conditions the kinetic net on physical parameters `p = [m1, m2, l1, l2]`

The Lagrangian is then `L = T - V = 0.5 * q_dot^T M(q,p) q_dot - V(q,p)`.

Accelerations are derived analytically via the Euler-Lagrange equations using JAX automatic differentiation:

```python
l_q     = jax.grad(L, argnums=0)(q, q_dot)
l_qt_fn = jax.grad(L, argnums=1)
l_qt_q  = jax.jacobian(l_qt_fn, argnums=0)(q, q_dot)
l_qt_qt = jax.jacobian(l_qt_fn, argnums=1)(q, q_dot)
rhs     = l_q - l_qt_q @ q_dot
q_tt    = jnp.linalg.solve(l_qt_qt, rhs)
```

### Guaranteeing Positive Definite Mass Matrix

The physical mass matrix M must be symmetric positive definite (SPD). An unconstrained MLP has no reason to produce SPD outputs. We use a Cholesky decomposition:

```python
# kinetic net outputs 3 values: [L11, L21, L22]
L = [[softplus(L11),  0        ],
     [L21,            softplus(L22)]]
M = L.T @ L + 1e-6 * I   # guaranteed SPD
T = 0.5 * q_dot @ M @ q_dot
```

`softplus` on the diagonal entries ensures they are positive, making `M = L^T L` positive definite by construction.

### FiLM Conditioning

FiLM modulates the hidden layers of the kinetic net using parameters-dependent scale and shift:

```
h_i = gamma_i(p) * softplus(W_i h_{i-1} + b_i) + beta_i(p)
```

where `[gamma_i, beta_i]` are outputs of a small MLP that takes `p = [m1, m2, l1, l2]` as input. This allows the mass matrix to depend on physical parameters without retraining.

The potential net does NOT use FiLM — instead, `p` is concatenated directly to the trig-encoded coordinates as input. This turned out to be critical for training stability (see Bugs section).

### Coordinate Transformation

Angles are transformed to `[sin(q), cos(q)]` pairs before entering any network. This avoids the discontinuity at `±π` and makes the inputs periodic and smooth.

---

## Data Generation

### Simulation

Trajectories are generated using JAX's `odeint` with `rtol=atol=1e-10`, ensuring near-exact ground truth. Each trajectory covers 20 seconds at `dt=0.005s` (4000 timesteps).

### Parameter Sampling

Parameters are sampled to avoid pathological dynamics:

```python
m1 ~ lognormal(0, 0.3)          # bulk in [0.55, 1.82]
m2 = m1 * uniform(0.2, 1.0)    # m2/m1 bounded
l1 ~ uniform(0.9, 2.0)
l2 = l1 * uniform(0.9, 1.1)    # near-equal lengths
```

Early attempts used `lognormal(0, 1.1)` for lengths, producing `l2/l1` ratios up to 45 — pathological dynamics with accelerations in the hundreds that corrupted training targets.

### Low-Energy Initial Conditions

Initial conditions are rejection-sampled to ensure the pendulum oscillates rather than rotates:

```python
H0 = T(q0, q_dot0) + V(q0)
V_max = (m1+m2)*g*l1 + m2*g*l2  # PE at upright equilibrium
accept if H0 < V_max
```

This prevents spinning solutions that produce unbounded angles and make rollout evaluation meaningless.

### Target Construction

Accelerations are computed via `numpy.gradient` (second-order finite differences) on the velocity columns of each trajectory. `dt=0.005` was chosen to make finite difference error negligible relative to the loss scale.

---

## Training

### Loss Function

The total loss combines two terms:

```
loss = huber_loss(q_tt_pred, q_tt_gt) + lambda * energy_conservation_loss
```

The **energy conservation loss** penalizes variance of the Hamiltonian along a trajectory chunk:

```python
H = T + V  # computed from model's T and V
loss_ec = jnp.var(H)  # should be ~0 for a conservative system
```

This requires batching at the trajectory level — each batch is a single contiguous chunk from one trajectory. Mixing trajectories in a batch makes this loss meaningless.

### Batch Construction

```python
# sample 1 trajectory, take a contiguous chunk of BATCH_SIZE timesteps
traj_idx = random.randint(0, n_trajectories)
time_idx = random.randint(0, T - BATCH_SIZE)
x_batch = X[traj_idx, time_idx:time_idx + BATCH_SIZE]
```

### Optimizer

AdamW with cosine decay schedule:
- Initial LR: 3e-3
- Final LR: 3e-5 (alpha=0.01)
- Gradient clipping: global norm 1.0
- Steps: 50,000

### Normalization

All inputs are normalized by training set mean and std. Angles are passed through unchanged (mean=0, std=1 by construction). Acceleration targets are normalized globally:

```python
dXdt_mean = jnp.mean(dXdt_train, axis=(0, 1))  # shape [2]
dXdt_std  = jnp.std(dXdt_train, axis=(0, 1))   # shape [2]
```

The rollout integrates in physical space, normalizing only at the model call boundary.

---

## The Bug Log

This section documents every significant bug encountered, in roughly chronological order. Most of these caused training to appear to work (loss going down) while the model was learning garbage.

### Bug 1: FiLM activation applied to output layer

**Symptom:** Training appeared to work but rollout was wrong.

**Cause:** The initial `compute_lagrangian` iterated over `enumerate(lagrangian_net.layers)` which includes the output layer. FiLM conditioning was applied to the output layer, and `softplus` was applied to the output — constraining the Lagrangian to `(0, ∞)` and preventing it from representing negative potential energy regions.

**Fix:** Separate the loop:
```python
for i in range(n_hidden):
    h = net.layers[i](h)
    h = softplus(h)
    h = gamma * h + beta
h = net.layers[n_hidden](h)  # output layer: no FiLM, no activation
```

### Bug 2: `training_loop` not returning the trained model

**Symptom:** Every rollout looked terrible regardless of training duration.

**Cause:** `training_loop` mutated `model` locally but returned only `loss_history`. The caller kept using the original randomly initialized model for all evaluation.

**Fix:** Return `model` from `training_loop`. This single bug invalidated many hours of evaluation.

### Bug 3: NaN in film_net due to zero-std normalization

**Symptom:** NaN in `film_params` immediately.

**Cause:** When training on a single trajectory (single parameter set), the std of the parameter columns of X was zero, causing division by zero in normalization.

**Fix:** Explicitly set mean=0, std=1 for parameter columns when `len(params) == 1`.

### Bug 4: `dXdt_mean` shape wrong

**Symptom:** Inconsistent acceleration normalization across timesteps.

**Cause:** `dXdt_mean = jnp.mean(dXdt_train, axis=0)` produced shape `[T, 2]` — a different normalization constant for each timestep. This gave the model inconsistent targets.

**Fix:** `dXdt_mean = jnp.mean(dXdt_train, axis=(0, 1))` — shape `[2]`, global normalization.

### Bug 5: Extreme parameter ranges causing training instability

**Symptom:** Loss all over the place with 2000 trajectories, training with 10 worked fine.

**Cause:** `l2/l1` ratios up to 45 and `m2/m1` up to 3 produced trajectories with `max|dXdt|` in the hundreds, dominating the Huber loss and corrupting gradients for all other trajectories.

**Fix:** Tighten parameter sampling ranges, enforce minimum lengths, add per-trajectory loss diagnostics to identify outliers.

### Bug 6: Indefinite mass matrix

**Symptom:** Training loss converged to ~0.05 but rollout diverged immediately. One-step acceleration check showed model output 1000x too small.

**Cause:** The unconstrained Lagrangian MLP learned a Lagrangian where `d²L/dq_dot²` (the mass matrix) had a negative eigenvalue (`-0.0002`). The `jnp.linalg.solve` with an indefinite matrix produces nonsensical accelerations.

**Diagnosis:**
```python
l_qt_qt = jax.jacobian(l_qt_fn, argnums=1)(q, q_dot)
print(jnp.linalg.eigvalsh(l_qt_qt))
# [-0.00020634  0.00049789]  ← negative eigenvalue
```

**Fix:** Restructure the architecture to guarantee SPD mass matrix via Cholesky decomposition. This required splitting the single Lagrangian MLP into separate kinetic and potential networks.

### Bug 7: Flat potential energy — vanishing gradients through FiLM

**Symptom:** After fixing the mass matrix, `l_q = dL/dq` was ~0.001 (should be O(1)). Model predicted near-zero acceleration everywhere. `dV/dq` model was 1000x smaller than ground truth.

**Cause:** The potential net used FiLM conditioning with `softplus` activations. Hidden layer gradient norms were 0.002-0.004 while the output layer was 0.6-0.8 — classic vanishing gradient through the FiLM+softplus stack. The potential net hidden layers received essentially no gradient signal.

**Diagnosis:**
```python
loss_fn = lambda m: potential_loss(m, ...)
grads = eqx.filter_grad(loss_fn)(model)
pot_leaves = jax.tree_util.tree_leaves(eqx.filter(grads.potential_net, eqx.is_array))
print([float(jnp.mean(jnp.abs(g))) for g in pot_leaves])
# [0.002, 0.004, 0.013, 0.027, 0.630, 0.879]
```

**Fix:** Remove FiLM from the potential net entirely. Concatenate `p` directly to the trig-encoded input and use a standard MLP with `tanh` activation. The `tanh` activation (vs `softplus`) also improved gradient flow.

### Bug 8: Rollout in wrong coordinate space

**Symptom:** Initial conditions in rollout plots didn't match ground truth.

**Cause:** The rollout was passing normalized states to the integrator but plotting against physical ground truth, or vice versa. The `rk4_step` function needed to integrate in physical space while only normalizing at the model call boundary.

**Fix:** Integrate entirely in physical space. Normalize state at the model call boundary, denormalize accelerations before integration:
```python
def f(state_physical):
    state_norm = (state - X_mean[:4]) / X_std[:4]
    q_tt_norm  = model(state_norm[:2], state_norm[2:], p_norm)
    q_tt_phys  = q_tt_norm * dXdt_std + dXdt_mean
    return jnp.concatenate([state[2:], q_tt_phys])
```

### Bug 9: Train/test split not seeded

**Symptom:** Rollouts in `inference.py` looked wrong even for trajectories the model had trained on.

**Cause:** `np.random.choice` in `train_test_split` used the global random state, producing a different split every run. The normalization stats computed from `Xtrain` were different between training and inference, so the model received out-of-distribution inputs.

**Fix:** Use `np.random.default_rng(seed)` with a fixed seed in both `train.py` and `inference.py`.

### Bug 10: Per-batch V normalization defeating the potential loss

**Symptom:** `potential_loss` was large but the potential net didn't respond.

**Cause:** Normalizing V within each batch:
```python
V_gt_norm   = (V_gt - jnp.mean(V_gt)) / jnp.std(V_gt)
V_model_norm = (V_model - jnp.mean(V_model)) / jnp.std(V_model)
```
When `V_model` had near-zero variance (flat potential), `std(V_model) ≈ 1e-8`, dividing by it produced huge values that swamped the gradient signal.

**Fix:** Use global normalization stats computed once from the full training set:
```python
V_gt_norm   = (V_gt   - V_mean) / V_std
V_model_norm = (V_model - V_mean) / V_std
```

---

## Key Design Decisions

### Why separate T and V networks instead of one Lagrangian network?

The unconstrained Lagrangian network cannot guarantee a positive definite mass matrix. Splitting into T and V allows the Cholesky constraint on T while leaving V unconstrained. The Euler-Lagrange equations still use autodiff through the full `L = T - V`.

### Why FiLM only on the kinetic net?

FiLM was originally applied to both networks. The potential net with FiLM suffered from vanishing gradients through the modulation layers. Removing FiLM from V and concatenating `p` directly gave the potential net a direct, short gradient path from the loss to the network weights.

This also makes physical sense: the kinetic energy structure `T = 0.5 * q_dot^T M(q,p) q_dot` has a specific parameter dependence (mass matrix entries scale as `m*l²`) that FiLM can learn to modulate. The potential energy `V = -(m1+m2)*g*l1*cos(t1)` has a simpler additive structure that direct concatenation handles naturally.

### Why energy conservation loss instead of direct V/M supervision?

Direct supervision requires knowing the analytical V and M — which defeats the purpose of learning from data. The energy conservation loss `var(H(t))` is a universal constraint that holds for any conservative system without requiring knowledge of the specific form of T or V. It provides gradient signal that pushes the model toward physically consistent T-V decomposition without using system-specific knowledge.

### Why single-trajectory batches?

Energy conservation is a property of a single trajectory — H should be constant along one system's evolution. Mixing timesteps from different trajectories (different parameter sets) makes `var(H)` meaningless since different systems have different energy levels.

### Why `tanh` for the potential net?

`softplus` saturates asymmetrically (output always positive, gradient approaches 1 for large positive inputs). `tanh` has symmetric saturation and better gradient flow for deep networks in the regime where inputs are not too large. Empirically, switching to `tanh` was the change that allowed `dV/dq` to grow from O(0.001) to O(0.1).

---

## Results

### Training Set Performance

After 50,000 training steps, the model tracks ground truth double pendulum trajectories for 10-15 seconds with correct frequency, amplitude, and phase across all four state variables `[q1, q2, w1, w2]`.

### Test Set Performance (Held-Out Parameter Combinations)

On trajectories with parameter combinations never seen during training:
- Correct oscillation frequency across all test trajectories
- Phase tracking for 8-12 seconds before Lyapunov divergence
- Failure mode: near-separatrix initial conditions where the pendulum crosses the energy barrier into spinning — a regime the model was not trained on

### Out-of-Distribution Generalization

The model was tested on parameter combinations outside the training ranges:

| Test | m1 | m2 | l1 | l2 | Result |
|------|----|----|----|----|--------|
| Heavy masses | 2.5 | 1.5 | 1.2 | 1.3 | Tracks ~10s ✓ |
| Long rods | 1.0 | 0.8 | 2.5 | 2.3 | Tracks ~10s ✓ |
| Both OOD | 3.0 | 2.0 | 2.8 | 2.5 | Tracks ~8s ✓ |
| Small system | 0.3 | 0.2 | 0.5 | 0.6 | Wrong frequency ✗ |

The model generalizes well in the direction of larger parameters (FiLM learns to modulate proportionally) but fails for much smaller parameters where the frequency regime is qualitatively different.

### Energy Conservation

The model Hamiltonian H(t) shows bounded oscillations without monotonic drift across all test trajectories. The ground truth integrator (`odeint` with `rtol=atol=1e-10`) conserves energy to 1e-8 (machine precision for float64). The model conserves energy to within ~10-30% in normalized units, with no secular accumulation over 15-second rollouts.

---

## Relationship to Prior Work

**DeLaN (Lutter et al., ICLR 2019)** introduced the structured Lagrangian network with separate T and V components and Cholesky decomposition for the mass matrix. The core architecture here follows the same principles, independently arrived at through the debugging process.

**LNN (Cranmer et al., 2020)** applied unconstrained Lagrangian networks to the double pendulum. The unconstrained approach has the indefinite mass matrix problem described in Bug 6.

**Novel contribution:** Parameter conditioning via FiLM on the kinetic network, combined with direct parameter concatenation for the potential network, allowing a single model to generalize across a family of double pendulum configurations. Neither DeLaN nor LNN address multi-system generalization. The energy conservation loss applied to single-trajectory batches as a training signal (rather than just an evaluation metric) is also not found in the reviewed literature.

---

## What Didn't Work

- **Scalar FiLM (γ, β per layer instead of per neuron):** Insufficient capacity for 2000 diverse trajectories. Abandoned in favor of per-neuron FiLM on the kinetic net.
- **`reduce_on_plateau` LR scheduler:** Created sinusoidal oscillations in the loss as LR cycled between reductions. Replaced with cosine decay.
- **Huber loss alone (without energy conservation):** Model found flat-V degenerate solutions with near-zero acceleration predictions everywhere.
- **Direct V and M supervision:** Philosophically inconsistent (requires knowing the answer) and practically caused the same scale mismatch problems as the unsupervised approach.
- **`lognormal(0, 1.1)` parameter sampling:** Produced extreme `l2/l1` ratios up to 45, making multi-trajectory training impossible.

---

## Implementation Details

**Framework:** JAX + Equinox  
**Integrator:** RK4 via `lax.scan` (JIT-compiled)  
**Data format:** HDF5 with streaming writes  
**Precision:** float64 throughout  
**Training time:** ~2 hours on Apple M-series CPU for 50k steps  
**Model size:** ~200k parameters (128-wide, 2-layer kinetic net + potential net + 32-wide FiLM net)

---

## Future Work

- Extend to dissipative systems (non-conservative forces, friction) using the Rayleigh dissipation function
- Scale to higher-DOF systems (n-link pendulum)
- Test with real sensor data (noisy observations, partial state)
- Symplectic integration for better long-term energy conservation
- Comparison against DeLaN baseline on standardized benchmarks
