import haiku as hk
import numpy as np
from jax import numpy as jnp


class MonotonicLinear(hk.Linear):
    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision=None,
    ) -> jnp.ndarray:
        inputs = jnp.atleast_2d(inputs)
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        out = jnp.dot(inputs, jnp.abs(w), precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b
        return out
