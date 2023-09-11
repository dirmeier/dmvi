import dataclasses

import haiku as hk
import jax
from jax import numpy as jnp
from jax import random

from dmvi._guide import Guide
from dmvi.dpm_solver import DPMSolver
from dmvi.nn.score_model import ScoreModel


@dataclasses.dataclass
class NoiseScheduleConfig:
    name: str
    Tmax: float = 1.0
    cosine_s: float = 0.008
    b_min: float = 0.1
    b_max: float = 10.0


def get_log_sqrt_alpha_cumprod_fn(config):
    if config.name == "cosine":
        return lambda t: jnp.log(
            jnp.cos((t + config.cosine_s) / (1.0 + config.cosine_s) * jnp.pi * 0.5)
        )
    elif config.name == "linear":
        return lambda t: -0.25 * t**2 * (config.b_max - config.b_min) - 0.5 * t * config.b_min
    raise ValueError("could not find correct schedule")


class NoiseSchedule:
    def __init__(self, config):
        self.config = config
        self.Tmax = config.Tmax
        self._log_sqrt_alpha_cumprod_fn = get_log_sqrt_alpha_cumprod_fn(config)
        self.T0 = 1 / 1000
        self.log_sqrt_alphas_cumprod_0 = jnp.log(self._log_sqrt_alpha_cumprod_fn(0.0))

    def log_sqrt_alphas_cumprod(self, t):
        lsa = self._log_sqrt_alpha_cumprod_fn(t)
        if self.config.name == "cosine":
            lsa = lsa - self.log_sqrt_alphas_cumprod_0
        return lsa

    def sqrt_alphas_cumprod(self, t):
        return jnp.exp(self.log_sqrt_alphas_cumprod(t))

    def scale(self, t):
        return jnp.sqrt(1.0 - jnp.exp(2.0 * self.log_sqrt_alphas_cumprod(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.log_sqrt_alphas_cumprod(t)
        log_std = 0.5 * jnp.log(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        if self.config.name == "linear":
            tmp = (
                2.0
                * (self.config.b_max - self.config.b_min)
                * jnp.logaddexp(-2.0 * lamb, jnp.zeros((1,)))
            )
            delta = self.config.b_min**2 + tmp
            return (
                tmp
                / (jnp.sqrt(delta) + self.config.b_min)
                / (self.config.b_max - self.config.b_min)
            )
        else:
            log_alpha = -0.5 * jnp.logaddexp(-2.0 * lamb, jnp.zeros((1,)))
            t_fn = (
                lambda log_alpha_t: jnp.arccos(
                    jnp.exp(log_alpha_t + self.log_sqrt_alphas_cumprod_0)
                )
                * 2.0
                * (1.0 + self.config.cosine_s)
                / jnp.pi
                - self.config.cosine_s
            )
            t = t_fn(log_alpha)
        return t


@dataclasses.dataclass
class ContinuousDDPMConfig:
    n_dim: int
    solver_n_steps: int
    solver_order: int


class ContinuousDDPM(Guide, hk.Module):
    """
    Denoising probabilistic diffusion model

    References
    ----------
    [1] https://arxiv.org/abs/2107.00630
    """

    def __init__(self, config, score_model, noise_schedule):
        super().__init__()
        self._config = config
        self._n_dim = config.n_dim

        self._score_model: ScoreModel = score_model
        self._noise_schedule: NoiseSchedule = noise_schedule
        self._sample_fn: DPMSolver = DPMSolver(
            self._n_dim,
            self._score_model,
            self._noise_schedule,
            config.solver_n_steps,
            config.solver_order,
        )

    def __call__(self, method="evidence", **kwargs):
        return getattr(self, method)(**kwargs)

    def evidence(self, z, is_training=True):
        z = jnp.atleast_2d(z).reshape(-1, self._n_dim)
        evidence = -self._diffusion_loss(z, is_training)
        return evidence

    def _diffusion_loss(self, y, is_training):
        t = random.uniform(
            key=hk.next_rng_key(),
            shape=(y.shape[0],),
            minval=self._noise_schedule.T0,
            maxval=self._noise_schedule.Tmax,
        )

        noise = jax.random.normal(hk.next_rng_key(), y.shape)
        alpha = self._noise_schedule.sqrt_alphas_cumprod(t).reshape(-1, 1)
        perturbed_y = alpha * y + (1.0 - alpha) * noise

        eps = self._score_model(perturbed_y, t, is_training)
        loss = jnp.sum(jnp.square(noise - eps), axis=-1)
        return loss

    def sample(self, sample_shape=(1,), is_training=True):
        return self._sample_fn(sample_shape, is_training)
