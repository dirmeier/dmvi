from collections import namedtuple
from typing import Callable, Tuple

import jax
import ml_collections
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd


class model_fn:
    def __new__(cls, n_dim, n_samples) -> Tuple[Callable, Callable, int]:
        _model = tfd.JointDistributionNamed(dict(
            loc=tfd.Independent(tfd.Normal(jnp.zeros(n_dim), 1.0), 1),
            y=lambda loc: tfd.Independent(tfd.Normal(loc, 1.0), 1),
        ))

        model_pinned = _model.experimental_pin(y=jnp.zeros(n_dim))
        init_params = model_pinned.sample_unpinned(seed=jr.PRNGKey(0))
        _, unravel_fn = jax.flatten_util.ravel_pytree(init_params)

        def _data_and_prior_fn(rng):
            param_rng, data_rng = jr.split(rng)
            prior_params = model_pinned.sample_unpinned(seed=param_rng)
            y = tfd.Normal(prior_params["loc"], 1.0).sample(seed=rng, sample_shape=(n_samples,))
            data_fn = namedtuple("data", "y")
            prior_fn = namedtuple("priors", "constrained unconstrained")
            return data_fn(y), \
                   prior_fn(
                       jax.flatten_util.ravel_pytree(prior_params)[0],
                       jax.flatten_util.ravel_pytree(prior_params)[0],
                   )

        def log_prob_fn(theta, y):
            theta = unravel_fn(theta)
            model_pinned = _model.experimental_pin(y=y)
            lp = model_pinned.unnormalized_log_prob(theta).sum()
            return lp

        return log_prob_fn, _data_and_prior_fn, init_params


def get_config():
    new_dict = lambda **kwargs: ml_collections.ConfigDict(initial_dictionary=kwargs)

    config = ml_collections.ConfigDict()
    config.name = "mean_model"

    config.data = new_dict(
        rng_key=42,
        n_dim=5,
        n_samples=1000,
        model_fn=model_fn
    )

    config.training = new_dict(
        batch_size=32,
        shuffle_data=True
    )

    return config
