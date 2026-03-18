# Generate double pendulum trajectories
from jax import config
config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx


# CONSTANTS
# ==========
GRAVITY = 9.806


def _make_rng(random_seed: int) -> jax.Array:
    return jax.random.PRNGKey(random_seed)

# SYSTEM INITIALIZATION
# =====================
def angular_state_initial_conditions(n_samples : int, n_pendulums: int = 2, random_seed: int = 123456789, data_type=jnp.float32) -> jax.Array:
    key = _make_rng(random_seed)
    q_key, qdot_key = jax.random.split(key)
    q0 = jax.random.uniform(
        q_key,
        shape=(n_samples, n_pendulums),
        minval=-jnp.pi / 2,
        maxval=jnp.pi / 2,
        dtype=data_type,
    )
    q0_dot = jax.random.uniform(
        qdot_key,
        shape=(n_samples, n_pendulums),
        minval=-jnp.pi / 5,
        maxval=jnp.pi / 5,
        dtype=data_type,
    )
    x0 = jnp.concatenate((q0, q0_dot), axis=1)
    return x0

def mass_length_samples(n_samples: int, n_pendulums: int = 2, random_seed: int = 123456789) -> dict:
    key = _make_rng(random_seed)
    m1_key, ratio_key, l1_key, l2_key = jax.random.split(key, 4)
    
    bob_mass = []
    m1 = jnp.exp(0.3 * jax.random.normal(m1_key, shape=(n_samples,)))
    bob_mass.append(m1)
    for _ in range(n_pendulums-1):
        ratio = jax.random.uniform(ratio_key, shape=(n_samples,), minval=0.2, maxval=1.0)  # m2/m1 bounded
        bob_mass.append(m1 * ratio)
    bob_mass = jnp.stack(bob_mass).T

    stick_length = []
    l1 = jax.random.uniform(l1_key, shape=(n_samples,), minval=0.9, maxval=2.0)
    l2 = l1 * jax.random.uniform(l2_key, shape=(n_samples,), minval=0.9, maxval=1.1)
    l2 = jnp.clip(l2, a_min=0.8)
    stick_length.append(l1)
    stick_length.append(l2)
    stick_length = jnp.stack(stick_length).T

    return {'mass': bob_mass, 'length': stick_length}

# DOUBLE PENDULUM CLASS
# =====================
class DoublePendulum(eqx.Module):
    """
    Double pendulum code assuming the following state variables:
    
    x = [t1, t2, w1, w2]
    t1: rad, angle of pendulum 1 from downward vertical
    t2: rad, angle of pendulum 2 from downward vertical
    w1: rad/s, angular velocity of pendulum 1
    w2: rad/s, angular velocity of pendulum 2

    analytical state transition returns the state vector derivative d/dt x: [w1, w2, g1, g2]
    """

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = GRAVITY

    @jit
    def kinetic_energy(self, q, q_dot):
        (t1, t2), (w1, w2) = q, q_dot
        T1 = 0.5 * self.m1 * (self.l1 * w1)**2
        T2 = 0.5 * self.m2 * ((self.l1 * w1)**2 + (self.l2 * w2)**2 + 2 * self.l1 * self.l2 * w1 * w2 * jnp.cos(t1 - t2))
        T  = T1 + T2
        return T
    
    @jit
    def potential_energy(self, q):
        if len(q) == 2:
            (t1, t2) = q
        else:
            t1, t2 = q[:, 0], q[:, 1]
        y1 = - self.l1 * jnp.cos(t1)
        y2 = y1 - self.l2 * jnp.cos(t2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        return V
    
    def lagrangian_fn(self, q, q_dot):
        T = self.kinetic_energy(q, q_dot)
        V = self.potential_energy(q)
        return T - V
    
    def hamiltonian_fn(self, q, q_dot):
        T = self.kinetic_energy(q, q_dot)
        V = self.potential_energy(q)
        return T + V
    
    def to_cartesian(self, q: jax.Array):
        """Convert angles to Cartesian coordinates."""
        q1, q2 = q
        x1 = self.l1 * jnp.sin(q1)
        y1 = -self.l1 * jnp.cos(q1)
        x2 = x1 + self.l2 * jnp.sin(q2)
        y2 = y1 - self.l2 * jnp.cos(q2)
        return x1, y1, x2, y2
    
    @staticmethod
    def is_low_energy(q, q_dot, m1, m2, l1, l2, g=9.81):
        t1, t2 = q
        w1, w2 = q_dot

        # PE at unstable equilibrium (both up)
        V_max = (m1 + m2) * g * l1 + m2 * g * l2
        
        # Total energy at initial condition
        T = 0.5 * m1 * (l1*w1)**2 + 0.5 * m2 * ((l1*w1)**2 + (l2*w2)**2 + 2*l1*l2*w1*w2*jnp.cos(t1-t2))
        V = -(m1 + m2) * g * l1 * jnp.cos(t1) - m2 * g * l2 * jnp.cos(t2)
        H = T + V
        
        return H < V_max

    @jit
    def analytical_state_transition(self, full_state, t):
        """
        1 - a1 * a2 in the denominator goes to zero when t1 - t2 = ±π/2 (cos → 0 kills it) — actually it's cos²(t1-t2) that drives the singularity. 
        """
        t1, t2, w1, w2 = full_state
        
        a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * jnp.cos(t1 - t2)
        a2 = (self.l1 / self.l2) * jnp.cos(t1 - t2)
        
        f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (w2**2) * jnp.sin(t1 - t2) - (self.g / self.l1) * jnp.sin(t1)
        f2 = (self.l1 / self.l2) * (w1**2) * jnp.sin(t1 - t2) - (self.g / self.l2) * jnp.sin(t2)
        
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)
        return jnp.stack([w1, w2, g1, g2])



if __name__ == '__main__':

    print('Double Pendulum Data')
    