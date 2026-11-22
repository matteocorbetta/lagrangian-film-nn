# Generate double pendulum trajectories
from typing import ClassVar, Sequence
from dataclasses import dataclass
import numpy as np
from jax import config
config.update('jax_enable_x64', True)
import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx

from lagrangian_film_nn import utils

# Utility functions
# ==================
def _make_rng(random_seed: int) -> jax.Array:
    return jax.random.PRNGKey(random_seed)

# CLASSES
# ========

@dataclass(frozen=True)
class PhysicalParameterRanges:
    """Sampling distribution of the double-pendulum physical parameters.

    m1 ~ m1_median * exp(m1_log_std * N(0, 1))   (log-normal, median m1_median)
    m2 = m1 * U(mass_ratio_range)
    l1 ~ U(l1_range)
    l2 = l1 * U(length_ratio_range)

    Attributes:
        m1_median (float): Median of m1.
        m1_log_std (float): Std of log(m1); ~95% of samples lie in median * exp(±2 * m1_log_std).
        mass_ratio_range (tuple[float, float]): Bounds of m2 / m1.
        l1_range (tuple[float, float]): Bounds of l1.
        length_ratio_range (tuple[float, float]): Bounds of l2 / l1.
    """
    
    m1_median: float = 1.0
    m1_log_std: float = 0.3
    mass_ratio_range: tuple[float, float] = (0.2, 1.0)
    l1_range: tuple[float, float] = (0.9, 2.0)
    length_ratio_range: tuple[float, float] = (0.9, 1.1)
    def __post_init__(self):
        if self.m1_median <= 0 or self.m1_log_std < 0:
            raise ValueError("m1_median must be > 0 and m1_log_std >= 0.")
        for name in ("mass_ratio_range", "l1_range", "length_ratio_range"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must satisfy 0 < min <= max, got {(lo, hi)}.")


@dataclass(frozen=True)
class ParameterSupport:
    """Region covered by a set of training parameters, as a box in the sampling coordinates
    z = (log m1, m2/m1, l1, l2/l1), where each coordinate is sampled independently
    (see `PhysicalParameterRanges`). A box in z, unlike one on (m1, m2, l1, l2), excludes
    combinations never seen in training, such as m2 > m1.

    Attributes:
        z_min (np.ndarray): [4] lower bound of the box in z.
        z_max (np.ndarray): [4] upper bound of the box in z.
    """
    z_min: np.ndarray
    z_max: np.ndarray
    COORD_NAMES: ClassVar[tuple[str, ...]] = ('m1', 'm2/m1', 'l1', 'l2/l1')

    @staticmethod
    def to_sampling_coords(p: np.ndarray) -> np.ndarray:
        """Maps physical parameters to sampling coordinates.

        Args:
            p (np.ndarray): [N, 4] or [4] parameters (m1, m2, l1, l2).

        Returns:
            np.ndarray: [N, 4] coordinates (log m1, m2/m1, l1, l2/l1).
        """
        p = np.atleast_2d(np.asarray(p, dtype=float))
        return np.stack([np.log(p[:, 0]), p[:, 1] / p[:, 0], p[:, 2], p[:, 3] / p[:, 2]], axis=1)

    @classmethod
    def from_params(cls, params: np.ndarray) -> "ParameterSupport":
        """Builds the support as the per-coordinate min/max of the given parameters.

        Args:
            params (np.ndarray): [N, 4] training parameters (m1, m2, l1, l2).

        Returns:
            ParameterSupport: Box enclosing all given parameters in z.
        """
        z = cls.to_sampling_coords(params)
        return cls(z_min=z.min(axis=0), z_max=z.max(axis=0))

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Per-coordinate distance outside the box, in units of the box width (0 = inside).

        Args:
            p (np.ndarray): [N, 4] or [4] parameters (m1, m2, l1, l2).

        Returns:
            np.ndarray: [N, 4] non-negative distances.
        """
        z = self.to_sampling_coords(p)
        width = np.where(self.z_max > self.z_min, self.z_max - self.z_min, 1.0)
        return np.maximum(0.0, np.maximum(self.z_min - z, z - self.z_max)) / width

    def is_ood(self, p: np.ndarray) -> np.ndarray:
        """Flags parameters lying outside the support in any coordinate.

        Args:
            p (np.ndarray): [N, 4] or [4] parameters (m1, m2, l1, l2).

        Returns:
            np.ndarray: [N] boolean mask.
        """
        return np.any(self.distance(p) > 0.0, axis=1)

    def describe(self) -> str:
        """Human-readable bounds of the support, with m1 shown in physical units.

        Returns:
            str: e.g. 'm1∈[0.42,2.31], m2/m1∈[0.20,0.99], l1∈[0.71,2.48], l2/l1∈[0.90,1.10]'.
        """
        lo, hi = self.z_min.copy(), self.z_max.copy()
        lo[0], hi[0] = np.exp(lo[0]), np.exp(hi[0])
        return ", ".join(f"{n}∈[{a:.2f},{b:.2f}]" for n, a, b in zip(self.COORD_NAMES, lo, hi))


# SYSTEM INITIALIZATION
# =====================
def angular_state_initial_conditions(n_samples : int, 
                                     n_pendulums: int = 2, 
                                     q0_range: Sequence[float] = [-jnp.pi / 2.0, jnp.pi/2.0],
                                     qt0_range: Sequence[float] = [-jnp.pi/5.0, jnp.pi/5.0],
                                     random_seed: int = 123456789, 
                                     data_type=jnp.float32) -> jax.Array:
    key = _make_rng(random_seed)
    q_key, q_t_key = jax.random.split(key)
    q0 = jax.random.uniform(
        q_key,
        shape=(n_samples, n_pendulums),
        minval=jnp.asarray(q0_range[0]),
        maxval=jnp.asarray(q0_range[1]),
        dtype=data_type,
    )
    q0_t = jax.random.uniform(
        q_t_key,
        shape=(n_samples, n_pendulums),
        minval=jnp.asarray(qt0_range[0]),
        maxval=jnp.asarray(qt0_range[1]),
        dtype=data_type,
    )
    x0 = jnp.concatenate((q0, q0_t), axis=1)
    return x0


def mass_length_samples(n_samples: int,
                        ranges: PhysicalParameterRanges = PhysicalParameterRanges(),
                        random_seed: int = 123456789) -> dict:
    """Samples masses and lengths of a double pendulum.

    Args:
        n_samples (int): Number of parameter sets to draw.
        ranges (PhysicalParameterRanges): Sampling distribution of each parameter.
        random_seed (int): Seed of the JAX PRNG.

    Returns:
        dict: {'mass': [n_samples, 2] array of (m1, m2), 'length': [n_samples, 2] array of (l1, l2)}.
    """
    m1_key, mass_ratio_key, l1_key, length_ratio_key = jax.random.split(_make_rng(random_seed), 4)

    m1 = ranges.m1_median * jnp.exp(ranges.m1_log_std * jax.random.normal(m1_key, shape=(n_samples,)))
    m2 = m1 * jax.random.uniform(mass_ratio_key, shape=(n_samples,),
                                 minval=ranges.mass_ratio_range[0], maxval=ranges.mass_ratio_range[1])

    l1 = jax.random.uniform(l1_key, shape=(n_samples,),
                            minval=ranges.l1_range[0], maxval=ranges.l1_range[1])
    l2 = l1 * jax.random.uniform(length_ratio_key, shape=(n_samples,),
                                 minval=ranges.length_ratio_range[0], maxval=ranges.length_ratio_range[1])

    return {'mass': jnp.stack([m1, m2], axis=1), 'length': jnp.stack([l1, l2], axis=1)}

# DOUBLE PENDULUM CLASS
# =====================
class DoublePendulum(eqx.Module):
    """
    Double pendulum code assuming the following state variables:
    
    x = [theta1, theta2, omega1, omega2]
    theta1: rad, angle of pendulum 1 from downward vertical
    theta2: rad, angle of pendulum 2 from downward vertical
    omega1: rad/s, angular velocity of pendulum 1
    omega2: rad/s, angular velocity of pendulum 2

    analytical state transition returns the state vector derivative d/dt x: [omega1, omega2, omegadot1, omegadot2]
    """

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = utils.GRAVITY

    @jit
    def kinetic_energy(self, q, q_t):
        (theta1, theta2), (omega1, omega2) = q, q_t
        T1 = 0.5 * self.m1 * (self.l1 * omega1)**2
        T2 = 0.5 * self.m2 * ((self.l1 * omega1)**2 + (self.l2 * omega2)**2 + 2 * self.l1 * self.l2 * omega1 * omega2 * jnp.cos(theta1 - theta2))
        T  = T1 + T2
        return T
    
    @jit
    def potential_energy(self, q):
        if len(q) == 2:
            (theta1, theta2) = q
        else:
            theta1, theta2 = q[:, 0], q[:, 1]
        y1 = - self.l1 * jnp.cos(theta1)
        y2 = y1 - self.l2 * jnp.cos(theta2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        return V
    
    def lagrangian_fn(self, q, q_t):
        T = self.kinetic_energy(q, q_t)
        V = self.potential_energy(q)
        return T - V
    
    def hamiltonian_fn(self, q, q_t):
        T = self.kinetic_energy(q, q_t)
        V = self.potential_energy(q)
        return T + V
    
    def to_cartesian(self, q: jax.Array):
        """Convert angles to Cartesian coordinates."""
        theta1, theta2 = q
        x1 = self.l1 * jnp.sin(theta1)
        y1 = -self.l1 * jnp.cos(theta1)
        x2 = x1 + self.l2 * jnp.sin(theta2)
        y2 = y1 - self.l2 * jnp.cos(theta2)
        return x1, y1, x2, y2
    
    @staticmethod
    def is_low_energy(q, q_t, m1, m2, l1, l2, g=9.81):
        theta1, theta2 = q
        omega1, omega2 = q_t

        # PE at unstable equilibrium (both up)
        V_max = (m1 + m2) * g * l1 + m2 * g * l2
        
        # Total energy at initial condition
        T = 0.5 * m1 * (l1 * omega1)**2 + 0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2 + \
                                                      2 * l1 * l2 * omega1 * omega2 * jnp.cos(theta1 - theta2))
        V = -(m1 + m2) * g * l1 * jnp.cos(theta1) - m2 * g * l2 * jnp.cos(theta2)
        H = T + V
        
        return H < V_max

    @jit
    def analytical_state_transition(self, state, t):
        """
        1 - a1 * a2 in the denominator goes to zero when theta1 - theta2 = ±π/2 (cos → 0 kills it) — actually it's cos²(t1-t2) that drives the singularity. 
        """
        theta1, theta2, omega1, omega2 = state
        
        a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * jnp.cos(theta1 - theta2)
        a2 = (self.l1 / self.l2) * jnp.cos(theta1 - theta2)
        
        f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (omega2**2) * jnp.sin(theta1 - theta2) - (self.g / self.l1) * jnp.sin(theta1)
        f2 = (self.l1 / self.l2) * (omega1**2) * jnp.sin(theta1 - theta2) - (self.g / self.l2) * jnp.sin(theta2)
        
        omegadot1 = (f1 - a1 * f2) / (1 - a1 * a2)
        omegadot2 = (f2 - a2 * f1) / (1 - a1 * a2)
        return jnp.stack([omega1, omega2, omegadot1, omegadot2])



if __name__ == '__main__':

    print('Double Pendulum Data')
    