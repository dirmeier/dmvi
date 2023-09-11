import dataclasses
from typing import Callable, List, Optional

import haiku as hk
from jax import numpy as jnp
from tensorflow_probability.substrates.jax.bijectors.masked_autoregressive import (  # noqa: E501
    _make_dense_autoregressive_masks,
)

from ._masked_dense import MaskedDense


@dataclasses.dataclass
class MADEConfig:
    n_dim: int
    hidden_dims: List
    n_params: int
    activation: Callable
    w_init: Optional[hk.initializers.Initializer] = jnp.zeros
    b_init: Optional[hk.initializers.Initializer] = jnp.zeros


class MADE(hk.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        masks = _make_dense_autoregressive_masks(config.n_params, config.n_dim, config.hidden_dims)

        layers = []
        for mask in masks:
            layers.append(
                MaskedDense(
                    mask=mask.astype(jnp.float32),
                    w_init=self.config.w_init,
                    b_init=self.config.b_init,
                )
            )
        self.layers = tuple(layers)

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        output = y
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i < len(self.layers[1:]) - 1:
                output = self.config.activation(output)
        output = hk.Reshape((self.config.n_dim, self.config.n_params))(output)
        return output
