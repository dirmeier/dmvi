import dataclasses
from typing import Callable, List, Optional

import haiku as hk
import jax
from jax import numpy as jnp


def get_embedding(inputs, embedding_dim, max_positions=10000):
    assert len(inputs.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = inputs[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb


@dataclasses.dataclass
class ScoreModelConfig:
    output_dim: int
    hidden_dims: List
    use_linear_embedding: bool
    embedding_dim: int
    dropout_rate: float
    activation: Callable = jax.nn.gelu
    w_init: Optional[hk.initializers.Initializer] = hk.initializers.TruncatedNormal(0.001)
    b_init: Optional[hk.initializers.Initializer] = jnp.zeros


@dataclasses.dataclass
class ScoreModel(hk.Module):
    config: ScoreModelConfig

    def __call__(self, z, t, is_training):
        dropout_rate = self.config.dropout_rate if is_training else 0.0
        t = jnp.atleast_1d(jnp.squeeze(t))
        t_embedding = get_embedding(t, self.config.embedding_dim)
        if self.config.use_linear_embedding:
            t_embedding = self.config.activation(
                hk.Linear(
                    self.config.embedding_dim,
                    w_init=self.config.w_init,
                    b_init=self.config.b_init,
                )(t_embedding)
            )
        h = hk.Linear(
            self.config.embedding_dim,
            w_init=self.config.w_init,
            b_init=self.config.b_init,
        )(z)
        h += t_embedding

        for dim in self.config.hidden_dims:
            h = hk.Linear(
                dim,
                w_init=self.config.w_init,
                b_init=self.config.b_init,
            )(h)
            h = self.config.activation(h)

        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = hk.dropout(hk.next_rng_key(), dropout_rate, h)
        h = hk.Linear(
            self.config.output_dim,
            w_init=self.config.w_init,
            b_init=self.config.b_init,
        )(h)
        return h
