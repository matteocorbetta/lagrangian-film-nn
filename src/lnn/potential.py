"""
Create a potential energy pipe for the Lagrangian NN model
"""

from typing import Any
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class CouplingModel(eqx.Module):

    name: str = eqx.field(static=True)
    net: eqx.nn.MLP

    def __init__(self, 
                 in_size: int,
                 out_size: int,
                 width_size: int,
                 depth: int,
                 activation: Callable[..., Any],
                 key: jax.Array,
                 name: str = 'PotentialCouplingModel'
                 ) -> None:

        self.name = name
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key
        )

    def __call__(self, p: jax.Array) -> jax.Array:
        return self.net(p)


class FunctionalModel(eqx.Module):

    name: str = eqx.field(static=True)
    net: eqx.nn.MLP

    def __init__(self, 
                 in_size: int,  
                 out_size: int,
                 width_size: int,
                 depth: int,
                 activation: Callable[..., Any],
                 key: jax.Array,
                 name: str='PotentialFunctional'
                 ) -> None:

        self.name = name
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key
        )

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.net(q)



class Potential(eqx.Module):
    coupling: CouplingModel
    functional: FunctionalModel

    def __init__(self,
                 coupling: CouplingModel,
                 functional: FunctionalModel) -> None:
        self.coupling = coupling
        self.functional = functional

    def __call__(self, q: jnp.Array, p: jnp.Array) -> jnp.Array:
        return self.coupling(p) @ self.functional(trig(q))
