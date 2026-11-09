
import jax.numpy as jnp


def sincos_factorization(q: jnp.Array) -> jnp.Array:
    trig_q = []
    for qi in q:
        trig_q.extend([jnp.sin(qi), jnp.cos(qi)])
    trig_q = jnp.array(trig_q)
    return trig_q