import dataclasses

import chex
import haiku as hk
from jax import numpy as jnp
from jax import random as jr
from jax.scipy import stats

from dmvi._guide import Guide


@dataclasses.dataclass
class ADVIConfig:
    n_dim: int


class ADVI(Guide, hk.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._n_dim = config.n_dim
        self.loc = hk.get_parameter("loc", [self._n_dim], init=jnp.zeros)
        self.log_scale = hk.get_parameter("log_scale", [self._n_dim], init=jnp.ones)

    def __call__(self, method="sample", **kwargs):
        return getattr(self, method)(**kwargs)

    def evidence(self, z, **kwargs):
        lp = stats.norm.logpdf(z, self.loc, jnp.exp(self.log_scale))
        return lp

    def sample(self, sample_shape=(1,)) -> chex.Array:
        z = jr.normal(hk.next_rng_key(), sample_shape + self.loc.shape)
        return jnp.exp(self.log_scale) * z + self.loc
