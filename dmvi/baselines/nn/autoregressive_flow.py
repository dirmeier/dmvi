from typing import Callable

import distrax
from chex import Array
from distrax._src.utils import math
from jax import numpy as jnp

from dmvi.baselines.nn.made import MADE


class MaskedAutoregressiveFlow(distrax.Bijector):
    """
    Masked autoregressive layer
    """

    def __init__(
        self,
        conditioner: MADE,
        bijector_fn: Callable,
        event_ndims_in: int = 1,
        event_ndims: int = 1,
        inner_event_ndims: int = 0,
    ):
        super().__init__(event_ndims_in)
        if event_ndims is not None and event_ndims < inner_event_ndims:
            raise ValueError(
                f"`event_ndims={event_ndims}` should be at least as"
                f" large as `inner_event_ndims={inner_event_ndims}`."
            )
        self._event_ndims = event_ndims
        self._inner_event_ndims = inner_event_ndims
        self.conditioner = conditioner
        self._inner_bijector = bijector_fn

    def forward_and_log_det(self, z: Array):
        shape = z.shape
        z = jnp.atleast_2d(z)
        y = jnp.zeros_like(z)
        for _ in jnp.arange(z.shape[-1]):
            params = self.conditioner(y)
            y, log_det = self._inner_bijector(params).forward_and_log_det(z)
        log_det = math.sum_last(log_det, self._event_ndims - self._inner_event_ndims)
        return y.reshape(shape), log_det

    def inverse_and_log_det(self, y: Array):
        shape = y.shape
        y = jnp.atleast_2d(y)
        params = self.conditioner(y)
        z, log_det = self._inner_bijector(params).inverse_and_log_det(y)
        log_det = math.sum_last(log_det, self._event_ndims - self._inner_event_ndims)
        return z.reshape(shape), log_det

    def forward(self, z: Array) -> Array:
        y, _ = self.forward_and_log_det(z)
        return y

    def inverse_and_likelihood_contribution(self, y: Array, **kwargs):
        return self.inverse_and_log_det(y)

    def forward_and_likelihood_contribution(self, z: Array, **kwargs):
        return self.forward_and_log_det(z)
