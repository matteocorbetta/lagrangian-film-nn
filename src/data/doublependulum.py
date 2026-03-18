# Generate double pendulum trajectories
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from jax import config
config.update('jax_enable_x64', True)
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx


# CONSTANTS
# ==========
GRAVITY = 9.806

# SYSTEM INITIALIZATION
# =====================
def angular_state_initial_conditions(n_samples : int, n_pendulums: int = 2, random_seed: int = 123456789, data_type=np.float32) -> np.array:
    rs = RandomState(MT19937(SeedSequence(random_seed)))
    q0 = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(n_samples, n_pendulums)).astype(data_type)
    q0_dot = np.random.uniform(low=-np.pi/5, high=np.pi/5, size=(n_samples, n_pendulums)).astype(data_type)
    x0 = np.concatenate((q0, q0_dot), axis=1)
    return x0

def mass_length_samples(n_samples: int, n_pendulums: int = 2, random_seed: int = 123456789) -> np.array:
    rs = RandomState(MT19937(SeedSequence(random_seed)))
    
    bob_mass = []
    m1 = np.random.lognormal(mean=0., sigma=0.3, size=n_samples)
    bob_mass.append(m1)
    for _ in range(n_pendulums-1):
        ratio = np.random.uniform(0.2, 1.0, size=n_samples)  # m2/m1 bounded
        bob_mass.append(m1 * ratio)
    bob_mass = np.stack(bob_mass).T

    stick_length = []
    l1 = np.random.uniform(low=0.9, high=2.0, size=n_samples)
    l2 = l1 * np.random.uniform(0.9, 1.1, size=n_samples)
    l2 = np.clip(l2, a_min=0.8, a_max=None)
    stick_length.append(l1)
    stick_length.append(l2)
    stick_length = np.stack(stick_length).T

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
    
    def to_cartesian(self, q: jnp.array):
        """Convert angles to Cartesian coordinates."""
        q1, q2 = q
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        x2 = x1 + self.l2 * np.sin(q2)
        y2 = y1 - self.l2 * np.cos(q2)
        return x1, y1, x2, y2
    
    @staticmethod
    def is_low_energy(q, q_dot, m1, m2, l1, l2, g=9.81):
        t1, t2 = q
        w1, w2 = q_dot

        # PE at unstable equilibrium (both up)
        V_max = (m1 + m2) * g * l1 + m2 * g * l2
        
        # Total energy at initial condition
        T = 0.5 * m1 * (l1*w1)**2 + 0.5 * m2 * ((l1*w1)**2 + (l2*w2)**2 + 2*l1*l2*w1*w2*np.cos(t1-t2))
        V = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
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


def integrate_trajectory(times, doublependulum, rtol=1e-10, atol=1e-10):
    
    def diffrax_ode_fun(t, y, args):
        dp_instance = args
        return dp_instance.analytical_state_transition(y, t)
    
    term = diffrax.ODETerm(diffrax_ode_fun)
    solver = diffrax.Dopri5() # An adaptive Runge-Kutta 4/5 solver

    # Initial time, final time
    t0 = times[0]
    t1 = times[-1]

    # Error control (adaptive step sizing)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Specify when to save the output (at the `times` array points)
    saveat = diffrax.SaveAt(ts=jnp.asarray(times))

     # Solve the ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0=0.001, # Initial guess for the step size
        y0=x0,
        args=doublependulum, # Pass the DoublePendulum instance as args
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None # Allow as many steps as needed for the given tolerance
    )

    # The trajectory is in sol.ys
    x_t = sol.ys
    # --- End Diffrax Integration ---

    return x_t


if __name__ == '__main__':

    # GENERATE DOBULE PENDULUM DATA
    # =================================
    import matplotlib.pyplot as plt
    from jax.experimental.ode import odeint
    import diffrax

    # Define simulation variables:
    t_span = [0, 40]    # time span
    num_steps = int( 200 * (t_span[1] - t_span[0]) )    # number of time steps
    times = np.linspace(t_span[0], t_span[1], num_steps)
    
    # Initialize state vector x(t=0)
    x0 = np.array([1*np.pi/7, 1*np.pi/4, 0.0, 0.0], dtype=np.float64)

    # Initialize double pendulum
    dp = DoublePendulum(m1=1.0, m2=0.5, l1=1.0, l2=0.5)

    # Integrate dobule pendulum analytical equation
    x_t = integrate_trajectory(times, dp, rtol=1e-10, atol=1e-10)

    # x_t = odeint(dp.analytical_state_transition, x0, t=times, rtol=1e-10, atol=1e-10)

    # Extract variables for plotting
    t1, t2, w1, w2 = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]

    t1_wrapped = (t1 + np.pi) % (2.0 * np.pi) - np.pi
    t2_wrapped = (t2 + np.pi) % (2.0 * np.pi) - np.pi

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(times, t1_wrapped, '-', label='pendulum 1')
    ax1.plot(times, t2_wrapped, '--', label='pendulum 2')
    ax1.set_ylabel('angle, rad')
    ax1.legend(fontsize=12)
    ax2 = fig.add_subplot(212)
    ax2.plot(times, w1, '-', label='pendulum 1')
    ax2.plot(times, w2, '--', label='pendulum 2')
    ax2.set_ylabel('angular velocity, rad/s')

    # Verifying correctness
    # ======================
    # reconstruct the energy at every time stamp
    q     = (t1_wrapped, t2_wrapped)
    q_dot = (w1, w2)

    def H_single(row):
        return dp.hamiltonian_fn((row[0], row[1]), (row[2], row[3]))
    
    H = jax.vmap(H_single)(x_t)
    H_max_drift = np.max(H) - np.min(H)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, H - H[0])
    ax.set_title("Total energy drift from initial value (should be around 0)")
    ax.set_ylabel("Hamiltonian (T+V) [J]")
    
    print(f"Maximum Energy drift: {H_max_drift:.2e}")


    plt.show()