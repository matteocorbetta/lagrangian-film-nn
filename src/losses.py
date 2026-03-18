import jax
import equinox as eqx
import jax.numpy as jnp

def energy_conservation_loss(model: eqx.Module, x: jax.Array, split_size: int = 2) -> jax.Array:
    """
    Calculates a loss term that penalizes deviations from energy conservation within trajectory chunks.

    This loss function leverages the physical principle that for an isolated Hamiltonian system,
    the total energy (Hamiltonian H = T + V) should remain constant over time. It computes the
    Hamiltonian for each timestep within a given batch chunk and penalizes the variance of these
    Hamiltonian values. A low variance implies better energy conservation.

    Args:
        model (eqx.Module): The neural network model, expected to be an instance of LagrangianNN
                            or a similar model that can provide `film_net`, `compute_cholesky_entries`,
                            and `compute_potential` methods.
        x (jax.Array): The input batch of state vectors, typically a 2D array of concatenated
                       generalized coordinates (q), generalized velocities (q_dot),
                       and system parameters (p). Shape: (batch_size * temporal_chunk_len, features).
        split_size (int, optional): The dimensionality of generalized coordinates (q), used for
                                    splitting the input vector `x`. Defaults to 2 for 2D systems.

    Returns:
        jax.Array: A scalar JAX array representing the variance of the Hamiltonian across timesteps
                   within each chunk. This variance is used as a loss component.
    """
    batch_q, batch_qt, batch_params = jnp.split(x, [split_size, split_size*2], axis=-1)
    
    def single_H(q: jax.Array, qt: jax.Array, p: jax.Array):
        """Computes the Hamiltonian (T + V) for a single timestep."""
        trig_q = jnp.array([jnp.sin(q[0]), jnp.cos(q[0]),
                            jnp.sin(q[1]), jnp.cos(q[1])])
        film_params = model.film_net(p).reshape(model.n_hidden, 2)
        chol = model.compute_cholesky_entries(trig_q, film_params)
        L = jnp.array([[jax.nn.softplus(chol[0]), 0.0],
                        [chol[1], jax.nn.softplus(chol[2])]])
        M = L.T @ L + jnp.eye(2) * 1e-6
        T = 0.5 * qt @ M @ qt
        V = model.compute_potential(trig_q, p)
        return T + V

    H = jax.vmap(single_H)(batch_q, batch_qt, batch_params)
    return jnp.var(H)  # should be near zero along a trajectory

def kinetic_loss(model, x, norm_stats, split_size=2):
    batch_q, batch_qt, batch_params = jnp.split(x, [split_size, split_size*2], axis=-1)
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    batch_p_phys = batch_params * X_std[4:] + X_mean[4:]

    def model_M(q, p_norm):
        trig_q = jnp.array([jnp.sin(q[0]), jnp.cos(q[0]),
                             jnp.sin(q[1]), jnp.cos(q[1])])
        film_params = model.film_net(p_norm).reshape(model.n_hidden, 2)
        h = model.apply_film(trig_q, film_params, model.kinetic_net)
        chol = model.kinetic_net.layers[model.n_hidden](h)
        L = jnp.array([[jax.nn.softplus(chol[0]), 0.0],
                        [chol[1], jax.nn.softplus(chol[2])]])
        return L.T @ L

    def gt_M(q_phys, p_phys):
        m1, m2, l1, l2 = p_phys
        cos_diff = jnp.cos(q_phys[0] - q_phys[1])
        M11 = (m1 + m2) * l1**2
        M12 = m2 * l1 * l2 * cos_diff
        M22 = m2 * l2**2
        return jnp.array([[M11, M12], [M12, M22]])

    M_model = jax.vmap(model_M)(batch_q, batch_params)
    M_gt    = jax.vmap(gt_M)(batch_q, batch_params * X_std[4:] + X_mean[4:])
    
    # normalize by gt scale
    M_std = jnp.std(M_gt) + 1e-8
    return jnp.mean(((M_model - M_gt) / M_std)**2)

def potential_loss(model, x, norm_stats, V_mean, V_std, split_size=2):
    batch_q, batch_qt, batch_params = jnp.split(x, [split_size, split_size*2], axis=-1)
    
    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    batch_q_phys = batch_q  # angles passthrough
    batch_p_phys = batch_params * X_std[4:] + X_mean[4:]
    
    def model_V(q, p_norm):
        trig_q = jnp.array([jnp.sin(q[0]), jnp.cos(q[0]),
                             jnp.sin(q[1]), jnp.cos(q[1])])
        h_pot = jnp.concatenate([trig_q, p_norm])
        V_m     = jnp.squeeze(model.potential_net(h_pot))
        return V_m
    
    def gt_V(q_phys, p_phys):
        y1 = -p_phys[2] * jnp.cos(q_phys[0])
        y2 = y1 - p_phys[3] * jnp.cos(q_phys[1])
        return (p_phys[0] + p_phys[1]) * 9.806 * y1 + p_phys[1] * 9.806 * y2
    
    V_model = jax.vmap(model_V)(batch_q, batch_params)
    V_gt    = jax.vmap(gt_V)(batch_q_phys, batch_p_phys)
    
    # normalize with global stats, not per-batch
    V_gt_norm    = (V_gt    - V_mean) / (V_std + 1e-8)
    V_model_norm = (V_model - V_mean) / (V_std + 1e-8)  # same scale target
    
    return jnp.mean((V_model_norm - V_gt_norm)**2)