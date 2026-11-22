# Normalization And The Euler–Lagrange Structure

This note explains why the current input/target normalization prevents the model from recovering the true kinetic and potential energies, even when its trajectory predictions are accurate, and proposes a normalization that keeps the Lagrangian structure intact.

**Summary.** The model applies the Euler–Lagrange equations to *normalized* velocities, as if they were time derivatives of the angles. That is only valid when the normalization is a pure rescaling of time: one scale $k$ shared by all velocity components, accelerations scaled by $k^2$, and no mean subtraction. The current pipeline standardizes velocities and accelerations independently, per component, with mean subtraction. Under that normalization, the true Lagrangian (up to a constant factor) is **not** a solution of the training problem. The network is forced to learn a distorted $\tilde M(q)$ and $\tilde V(q)$ that compensate for each other, which matches the systematic errors seen in the T–V decomposition plots.

---

## 1. Notation

Physical system, with generalized coordinates $q \in \mathbb{R}^n$ ($n = 2$ for the double pendulum):

$$
L(q, \dot q) = \tfrac12\, \dot q^\top M(q)\, \dot q \;-\; V(q).
$$

For any symmetric matrix field $A(q)$ and vector $w$, define the quadratic *Coriolis/centrifugal* term

$$
C[A](q, w) \;=\; \sum_k w_k\, \partial_k A(q)\, w \;-\; \tfrac12\, \nabla_q\!\left(w^\top A(q)\, w\right),
\qquad \partial_k = \partial/\partial q_k .
$$

$C[A]$ is linear in $A$ and homogeneous of degree 2 in $w$: $C[A](q, \lambda w) = \lambda^2 C[A](q, w)$.

The Euler–Lagrange (EL) equations of $L$ are

$$
\frac{d}{dt}\big(M \dot q\big) - \nabla_q T + \nabla V = 0
\quad\Longleftrightarrow\quad
M(q)\,\ddot q + C[M](q, \dot q) + \nabla V(q) = 0 .
\tag{1}
$$

For the double pendulum, $M_{12} = m_2 l_1 l_2 \cos(q_1 - q_2) \neq 0$ and $\partial_k M \neq 0$, so $C[M]$ is non-zero. It produces the $\dot q_j^2 \sin(q_1 - q_2)$ terms in `DoublePendulum.analytical_state_transition`.

**Identifiability.** $L$ and $cL + d$ ($c \neq 0$) give identical equations of motion. So the most we can hope to recover is $c\,T$ and $c\,V + d$, with **one shared** $c$. Inequivalent Lagrangians are tied to extra conserved quantities (Currie & Saletan, 1966). We assume the chaotic double pendulum, which conserves only energy, has none within the $T - V$ form.

---

## 2. What the model computes

`LagrangianNN` ([model.py](../src/lagrangian_film_nn/lnn/model.py)) receives raw angles $q$, a velocity-like input $v$ and normalized parameters $p$. It builds

$$
\tilde L(q, v) = \tfrac12\, v^\top \tilde M(q)\, v - \tilde V(q),
\qquad \tilde M = \Lambda^\top \Lambda \ \ (\text{Cholesky}),
$$

and `__call__` returns

```python
rhs  = l_q - l_qt_q @ q_t          # q_t is v here
q_tt = solve(l_qt_qt, rhs)
```

With $\partial \tilde L/\partial q = \tfrac12\nabla_q(v^\top\tilde M v) - \nabla\tilde V$, $\ \partial^2\tilde L/\partial v\,\partial q \cdot v = \sum_k v_k\,\partial_k\tilde M\, v$ and $\partial^2\tilde L/\partial v^2 = \tilde M$, the output $\hat a$ satisfies

$$
\tilde M(q)\,\hat a + C[\tilde M](q, v) + \nabla\tilde V(q) = 0 .
\tag{2}
$$

Equation (2) is the EL equation of $\tilde L$ **under the assumption $dq/d\tau = v$** for some time variable $\tau$, with $\hat a = dv/d\tau$. The chain-rule factor $\sum_k v_k\,\partial_k \tilde M$ comes from $d\tilde M/d\tau = \sum_k \partial_k\tilde M\, dq_k/d\tau$, and the code uses $v$ for $dq/d\tau$.

---

## 3. What the model is trained to match

`normalize_data` ([train_utils.py](../src/lagrangian_film_nn/train_utils.py)) computes, from the training split:

- velocities: $\mu = $ `X_mean[2:4]`, $S_v = \mathrm{diag}(s_1, s_2)$ with $s_i = $ `X_std[2+i]`
- accelerations: $\nu = $ `dXdt_mean`, $S_a = \mathrm{diag}(\sigma_1, \sigma_2)$ with $\sigma_i = $ `dXdt_std[i]`

Angles are passed through unchanged (`X_mean[:2] = 0`, `X_std[:2] = 1`). Training then fits

$$
v = S_v^{-1}(\dot q - \mu), \qquad
\hat a(q, v) \;\stackrel{!}{=}\; a^*(q, v) := S_a^{-1}\big(\ddot q - \nu\big),
$$

and `rk4_step` ([simulate.py](../src/lagrangian_film_nn/simulate.py)) inverts the same maps at rollout time.

Substituting the true dynamics (1), with $\dot q = S_v v + \mu$, gives the target

$$
a^*(q, v) = -\,S_a^{-1} M^{-1}\Big(C[M](q,\, S_v v + \mu) + \nabla V\Big) - S_a^{-1}\nu .
\tag{3}
$$

**The question:** is there a pair $(\tilde M, \tilde V)$ for which (2) reproduces (3) exactly, and is it the true pair up to a shared scale?

---

## 4. Matching conditions

Solving (2) for $\hat a$ and comparing with (3), both sides are polynomials in $v$. They must agree degree by degree, for every $q$.

**Model side** (from (2)):

$$
\hat a = \underbrace{-\tilde M^{-1}\nabla\tilde V}_{\text{degree }0}
\;\underbrace{-\;\tilde M^{-1} C[\tilde M](q, v)}_{\text{degree }2} .
$$

There is **no degree-1 term**: the model's $\tilde L$ has no part linear in $v$.

**Target side.** Since $C[M]$ is quadratic, expand $C[M](q, S_v v + \mu) = C[M](q, S_v v) + 2B(q; S_v v, \mu) + C[M](q, \mu)$, where $B$ is the symmetric bilinear form associated with $C[M]$:

$$
a^* = \underbrace{-S_a^{-1}\big(M^{-1}(\nabla V + C[M](q,\mu)) + \nu\big)}_{\text{degree }0}
\;\underbrace{-\,2\,S_a^{-1}M^{-1}B(q; S_v v, \mu)}_{\text{degree }1}
\;\underbrace{-\,S_a^{-1}M^{-1}C[M](q, S_v v)}_{\text{degree }2} .
$$

### 4.1 Velocity mean $\mu \neq 0$ is not representable

The target's degree-1 term must vanish because the model has none. $B(q;\cdot,\mu)$ is a non-zero linear map whenever $\partial_k M \neq 0$ and $\mu \neq 0$. So **any non-zero velocity mean creates a component of the target that the model cannot represent at all.**

Physically, a linear-in-velocity term would correspond to a gyroscopic or "magnetic" force, which the $T - V$ architecture deliberately excludes.

The acceleration mean $\nu$ appears only in the degree-0 term. It acts as a constant force, which a single scalar potential $\tilde V$ can absorb only when $\tilde M S_a^{-1}\nu$ is a gradient field. For $q$-dependent $\tilde M$ that is generally false.

### 4.2 Isotropic case: the time-scale condition $\sigma = s^2$

Take the most favourable case: $\mu = \nu = 0$, $S_v = s I$, $S_a = \sigma I$. Using the homogeneity of $C$:

$$
a^* = -\frac{1}{\sigma}\,M^{-1}\Big(s^2\, C[M](q, v) + \nabla V\Big).
$$

Try the "correct" answer: the true mass matrix up to a scale, $\tilde M = \kappa M$. Since $C$ is linear in its matrix argument, $C[\tilde M] = \kappa\, C[M]$, and

$$
\hat a = -M^{-1}\Big(C[M](q, v) + \tfrac{1}{\kappa}\nabla\tilde V\Big).
$$

Matching degree by degree:

- degree 2: $\;M^{-1}C[M] = \dfrac{s^2}{\sigma}\,M^{-1}C[M] \;\Longrightarrow\; \boxed{\,r := \dfrac{s^2}{\sigma} = 1\,}$
- degree 0: $\;\tilde V = \dfrac{\kappa}{\sigma}\,V$

If $r = 1$, the solution is $\tilde M = \kappa M$, $\tilde V = \kappa V/s^2$. Equivalently,

$$
\tilde L(q, v) = \frac{\kappa}{s^2}\, L(q,\, s v) = c\, L(q, \dot q), \qquad c = \kappa/s^2 ,
$$

so the learned $T$ and $V$ are the true ones with one shared scale, as desired.

If $r \neq 1$, **no $\tilde M$ proportional to $M$ satisfies the degree-2 condition.** An exact solution would need a metric $\tilde M$ whose Coriolis term equals $r$ times that of $M$, i.e. whose Levi-Civita connection is $r\,\Gamma[M]$. A scaled connection is not metric-compatible with $M$, and in more than one dimension it is generally not the connection of any metric. (In 1-D it is: $\tilde M \propto M^r$, so the problem is specifically about coupled degrees of freedom.) Whether or not some exotic $\tilde M$ exists, it is **not** $\kappa M$. Therefore the learned kinetic energy cannot be proportional to the true one.

**Interpretation as a time rescaling.** Equation (2) is the EL equation in a time variable $\tau$ with $dq/d\tau = v$. Since $\dot q = s v$, that means $\tau = s\,t$, and the acceleration consistent with (2) is $dv/d\tau = \ddot q / s^2$. Training instead asks for $\ddot q/\sigma$. The two agree only if $\sigma = s^2$, i.e. the acceleration scale must be the square of the velocity scale. Otherwise the target is the true $\tau$-acceleration multiplied by $r$: the Coriolis part is scaled by $r$ while the model's structure fixes it at 1, and the potential part can absorb any factor.

### 4.3 Per-component scales: the shared-time condition $s_1 = s_2$

With $\mu = \nu = 0$ but $S_v = K = \mathrm{diag}(s_1, s_2)$, the natural candidate is the true kinetic energy rewritten in $v$: $T = \tfrac12 (Kv)^\top M (Kv)$, i.e. $\tilde M = \kappa\, K M K$. Its Coriolis term is

$$
C[\tilde M](q, v) = \kappa\Big(\sum_k v_k\, K\,\partial_k M\, K v \;-\; \tfrac12\nabla_q\big((Kv)^\top M (Kv)\big)\Big),
$$

whereas matching (3) requires

$$
C[\tilde M](q, v) = \tilde M\, S_a^{-1} M^{-1}\, C[M](q, Kv)
= \kappa\, K M K S_a^{-1} M^{-1}\Big(\sum_k (Kv)_k\, \partial_k M\, Kv - \tfrac12 \nabla_q\big((Kv)^\top M (Kv)\big)\Big).
$$

Even with the best choice $S_a = K^2$ ($\sigma_i = s_i^2$), the two differ in three places:

1. **Chain-rule factor.** The model uses $v_k$ where the physics has $(Kv)_k = s_k v_k$, because the model believes $dq_k/d\tau = v_k$.
2. **Gradient term.** On the model side it carries no left factor $K$.
3. **Prefactor.** $K M K S_a^{-1} M^{-1} = K M K^{-1} M^{-1}$, which equals $K$ only if $K$ commutes with $M$. With $M_{12} \neq 0$ that requires $s_1 = s_2$.

All three discrepancies disappear if and only if $K = sI$. Time is shared by all coordinates, so all velocity components must be scaled by the **same** factor, and likewise all acceleration components.

### 4.4 Interaction with the energy-conservation loss

`energy_conservation_loss` penalizes variation of the model's $\tilde H = \tfrac12 v^\top\tilde M v + \tilde V$ along data trajectories. The data conserve the true $H = \tfrac12 s^2 v^\top M v + V$ (isotropic case, $\mu = 0$). So the energy loss is minimized by $\tilde M = \kappa s^2 M$, $\tilde V = \kappa V$, i.e. proportional to the truth.

- With $r = 1$ this is the same family the acceleration loss selects in §4.2 ($\kappa' = \kappa s^2$), so the two losses agree.
- With $r \neq 1$ the acceleration loss cannot be satisfied by any $\tilde M \propto M$, while the energy loss pulls towards exactly that. **The two loss terms conflict**, and the optimum is a compromise that satisfies neither.

---

## 5. Consequences for this project

| Normalization property | Condition for exact $\tilde L = cL$ | Current pipeline |
|---|---|---|
| velocity mean | $\mu = 0$ | `X_mean[2:4]` subtracted |
| acceleration mean | $\nu = 0$ | `dXdt_mean` subtracted |
| shared velocity scale | $s_1 = s_2$ | per-component `X_std[2:4]` |
| shared acceleration scale | $\sigma_1 = \sigma_2$ | per-component `dXdt_std` |
| time-consistent scales | $\sigma = s^2$ | $s$ and $\sigma$ set independently |

Expected symptoms, all present in the `kinpot_decomposition` results of `model_new_K2_s123_T512_20261121_155307`:

- **Distorted $\tilde V$:** the relative weight of $\cos q_1$ and $\cos q_2$ is wrong (error map zero on two curves, peaks at $(0,\pm\pi)$ and $(\pm\pi,0)$).
- **Distorted $\tilde M$:** the learned $T$ depends on $q_1 + q_2$, although the true $M$ depends only on $q_1 - q_2$.
- **Wrong frequencies out of distribution:** the small-oscillation frequency is set by $\tilde M^{-1}\nabla^2\tilde V$, which is wrong whenever $\tilde M$ and $\tilde V$ are not proportional to the truth. Inside the training data, the compensating errors largely cancel in the accelerations, but less so outside it.
- **Low training loss regardless:** the network is a flexible map from state to acceleration and can fit the targets approximately on the data. The structure only fails when we read off $T$ and $V$ separately.

### How to measure the mismatch for a trained model

```python
import numpy as np

with np.load('results/models/<model_name>.npz', allow_pickle=True) as run:
    ns = run['norm_stats'].item()
s, mu     = np.asarray(ns['X_std'][2:4]), np.asarray(ns['X_mean'][2:4])
sigma, nu = np.asarray(ns['dXdt_std']),   np.asarray(ns['dXdt_mean'])

print('r_i = s_i^2 / sigma_i :', s**2 / sigma)   # should be 1, 1
print('s1 / s2               :', s[0] / s[1])     # should be 1
print('sigma1 / sigma2       :', sigma[0] / sigma[1])
print('mu, nu                :', mu, nu)          # should be ~0
```

---

## 6. Proposed fix: normalize time, not variables

Choose a single time scale $t_0$ and define the dimensionless time $\tau = t / t_0$. Then

$$
v = t_0\,\dot q = \frac{dq}{d\tau},
\qquad
a = t_0^2\,\ddot q = \frac{d^2 q}{d\tau^2},
$$

with no mean subtraction, and the same $t_0$ for every component. This is exactly the transformation under which (2) is the correct EL equation. The EL equations are covariant under time rescaling, and $L(q, \dot q)$ becomes $L(q, v/t_0) = \tfrac{1}{t_0^2}\big(\tfrac12 v^\top M v - t_0^2 V\big)$, which is again of the $T - V$ form.

### Choice of $t_0$

- **Data-driven (recommended):** $t_0 = 1/\bar s$, with $\bar s = \sqrt{\operatorname{mean}(\dot q^2)}$ pooled over both components and all training samples (an RMS, not a std, since there is no mean subtraction). Then $v$ has unit RMS.
- **Physics-based:** $t_0 = \sqrt{\bar l / g}$ for a typical length $\bar l$. This is interpretable and does not depend on the data split.

### Changes in `normalize_data`

The existing `norm_stats` interface can represent the fix exactly, so `rk4_step`, `plot_energy` and `kinpot_decomposition` need no change:

| entry | current | proposed |
|---|---|---|
| `X_mean[2:4]` | per-component mean of $\dot q$ | `0` |
| `X_std[2:4]` | per-component std of $\dot q$ | `s_bar`, `s_bar` |
| `dXdt_mean` | per-component mean of $\ddot q$ | `0` |
| `dXdt_std` | per-component std of $\ddot q$ | `s_bar**2`, `s_bar**2` |
| `X_mean[:2]`, `X_std[:2]` (angles) | `0`, `1` | unchanged |
| `X_mean[4:]`, `X_std[4:]` (parameters) | standardized | unchanged |

Parameters $p$ can keep their standardization: they enter only through the FiLM conditioning, not through time derivatives, so they cannot break the EL structure.

### Side effects to handle

- **Loss scale.** The targets $a = \ddot q/\bar s^2$ no longer have unit variance, and one component may dominate. If needed, use explicit per-component loss weights, $\sum_i w_i\,\ell(\hat a_i - a_i)$. Loss weights only change how residuals are weighted, not which $\tilde L$ fits exactly, so they do not break the structure. The Huber `delta=1.0` may need retuning to the new target scale.
- **Checkpoints.** Models trained with the old `norm_stats` remain loadable, since rollouts invert whatever stats are stored. Their $T$ and $V$ should not be interpreted physically.

### Validation after the fix

With time normalization, $\tilde L(q, v) = c\,L(q, \dot q)$ is an exact solution, and the model's quantities map back to physical units as

$$
M_{\text{model}}(q) = \frac{\tilde M(q)}{\bar s^{2}} \approx c\, M(q),
\qquad
\tilde V(q) \approx c\, V(q) + d .
$$

The decomposition test should check this **with one shared $c$**, instead of z-scoring $T$ and $V$ separately (which allows independent scales and hides errors in the relative size of $T$ and $V$):

1. Fit $(c, d)$ by least squares: $\tilde V(q) \approx c\,V(q) + d$ over the $(q_1, q_2)$ grid.
2. With the same $c$, compare $M_{\text{model}}(q)$ with $c\,M(q)$ entrywise ($M_{11}$, $M_{12}$, $M_{22}$). This avoids testing $T$ at a single arbitrary velocity.
3. Report both residuals and $c$ itself. A $c$ that varies strongly with $p$ is itself a warning sign.

---

## 7. Relation to existing approaches

- **Deep Lagrangian Networks** (Lutter, Ritter & Peters, ICLR 2019) use the same structure as this model: a Cholesky-parameterized $M(q)$ and a separate $V(q)$, combined through the EL equations. The argument above applies to any such structured model. Its outputs are tied to its inputs by $d/dt$, so the inputs and targets cannot be scaled independently.
- **Lagrangian Neural Networks** (Cranmer et al., 2020) learn an unstructured $L(q, \dot q)$ and differentiate it the same way. A general $L$ could absorb a degree-1 term, but the $\sigma = s^2$ and shared-scale conditions still apply, because the EL operator still assumes $v = dq/d\tau$.
- **Standard practice.** Standardizing inputs and targets independently is harmless for black-box regressors, because nothing in the model links them. Physics-structured models instead need *nondimensionalization*: a consistent choice of units ($t_0$, and if desired a mass and length scale) under which the governing equations keep their form. Section 6 is the minimal version of that.

---

## References

- M. Lutter, C. Ritter, J. Peters. *Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning.* ICLR 2019.
- M. Cranmer, S. Greydanus, S. Hoyer, P. Battaglia, D. Spergel, S. Ho. *Lagrangian Neural Networks.* ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations.
- D. G. Currie, E. J. Saletan. *q-Equivalent Particle Hamiltonians.* J. Math. Phys. 7, 967 (1966).
