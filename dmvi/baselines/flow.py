import dataclasses

import chex
import distrax
import haiku as hk
from jax import lax
from jax import numpy as jnp

from dmvi._guide import Guide
from dmvi.baselines.nn.autoregressive_flow import MaskedAutoregressiveFlow
from dmvi.baselines.nn.made import MADE, MADEConfig
from dmvi.baselines.nn.permutation import Permutation


@dataclasses.dataclass
class FlowConfig:
    n_dim: int
    n_layers: int
    conditioner: dict


def unstack(x, axis=0):
    return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


def flow_fn(config):
    def _bijector_fn(params: chex.Array):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    layers = []
    order = jnp.arange(config.n_dim)

    for _ in range(config.n_layers):
        layer = MaskedAutoregressiveFlow(
            bijector_fn=_bijector_fn,
            conditioner=MADE(MADEConfig(**{**config.conditioner, "n_dim": config.n_dim})),
        )
        order = order[::-1]
        layers.append(layer)
        layers.append(Permutation(order, 1))

    base_distribution = distrax.Independent(
        distrax.Normal(loc=jnp.zeros(config.n_dim), scale=1.0),
        reinterpreted_batch_ndims=1,
    )
    tr = distrax.Transformed(base_distribution, distrax.Inverse(distrax.Chain(layers)))

    return tr


class Flow(Guide, hk.Module):
    def __init__(self, config):
        super().__init__()
        self._flow = flow_fn(config)

    def __call__(self, method="sample", **kwargs):
        return getattr(self, method)(**kwargs)

    def sample(self, sample_shape=(), **kwargs) -> chex.Array:
        return self._flow.sample(seed=hk.next_rng_key(), sample_shape=sample_shape)

    def evidence(self, z, **kwargs) -> chex.Array:
        return self._flow.log_prob(z)
