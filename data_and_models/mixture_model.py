from collections import namedtuple
from typing import Callable, Tuple

import jax
import ml_collections
from jax import numpy as jnp
from jax import random as jr
from jax.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd


class model_fn:
    def __new__(cls, n_dim, n_samples) -> Tuple[Callable, Callable, int]:
        _model = tfd.JointDistributionNamed(dict(
            loc=tfd.Independent(
                tfd.Normal(
                    loc=jnp.stack([
                        -jnp.ones(n_dim),
                        jnp.zeros(n_dim),
                        jnp.ones(n_dim),
                    ]),
                    scale=1.0),
                reinterpreted_batch_ndims=2),
            scale=tfd.Independent(tfd.HalfNormal(jnp.ones((3, n_dim))), reinterpreted_batch_ndims=2),
            y=lambda loc, scale: tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=jnp.ones(3)),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=scale))
        ))

        model_pinned = _model.experimental_pin(y=jnp.zeros((n_samples, n_dim)))
        bijector = model_pinned.experimental_default_event_space_bijector()
        init_params = model_pinned.sample_unpinned(seed=jr.PRNGKey(0))
        init_params = bijector.inverse(init_params)
        _, unravel_fn = jax.flatten_util.ravel_pytree(init_params)

        def _data_and_prior_fn(rng):
            param_rng, data_rng = jr.split(rng)
            prior_params = model_pinned.sample_unpinned(seed=param_rng)
            y = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=jnp.ones(3)),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=prior_params["loc"],
                    scale_diag=prior_params["scale"])).sample(seed=rng, sample_shape=(n_samples,))
            data_fn = namedtuple("data", "y")
            prior_fn = namedtuple("priors", "constrained unconstrained")
            return data_fn(y), \
                   prior_fn(
                       jax.flatten_util.ravel_pytree(prior_params)[0],
                       jax.flatten_util.ravel_pytree(
                           bijector.inverse(prior_params))[0],
                   )

        def log_prob_fn(theta, y):
            theta = unravel_fn(theta)
            theta_forward = bijector.forward(theta)
            log_det = bijector.forward_log_det_jacobian(theta)
            lp = _model.log_prob(**theta_forward, y=y).sum() + log_det.sum()
            return lp

        return log_prob_fn, _data_and_prior_fn, init_params


def get_config():
    new_dict = lambda **kwargs: ml_collections.ConfigDict(
        initial_dictionary=kwargs)

    config = ml_collections.ConfigDict()
    config.name = "mixture_model"

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
