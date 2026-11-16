
import jax
import jax.numpy as jnp



# TRIGONOMETRIC FUNCTIONS
# ========================
def sincos_factorization(q: jax.Array) -> jax.Array:
    trig_q = []
    for qi in q:
        trig_q.extend([jnp.sin(qi), jnp.cos(qi)])
    trig_q = jnp.array(trig_q)
    return trig_q


def trig_transform(q: jax.Array) -> jax.Array:
    """[q1, q2, ...] -> [sin q1, cos q1, sin q2, cos q2, ...], shape (2*len(q),)."""
    return jnp.stack([jnp.sin(q), jnp.cos(q)], axis=-1).reshape(-1)