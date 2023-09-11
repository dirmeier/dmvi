import dataclasses

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd

from dmvi._guide import Guide
from dmvi.dpm_solver import DPMSolver
from dmvi.nn.score_model import ScoreModel


def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = jnp.concatenate(
        [jnp.expand_dims(x, 2), jnp.tile(jnp.expand_dims(xp, 0), (N, 1, 1))],
        axis=2,
    )
    x_indices = jnp.argsort(all_x, axis=2)
    sorted_all_x = jnp.take_along_axis(all_x, x_indices, axis=2)
    x_idx = jnp.argmin(x_indices, axis=2)
    cand_start_idx = x_idx - 1
    start_idx = jnp.where(
        jax.lax.eq(x_idx, 0),
        jnp.array(1),
        jnp.where(
            jax.lax.eq(x_idx, K),
            jnp.array(K - 2),
            cand_start_idx,
        ),
    )
    end_idx = jnp.where(jax.lax.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = jnp.take_along_axis(sorted_all_x, jnp.expand_dims(start_idx, 2), axis=2).squeeze(2)
    end_x = jnp.take_along_axis(sorted_all_x, jnp.expand_dims(end_idx, 2), axis=2).squeeze(2)
    start_idx2 = jnp.where(
        jax.lax.eq(x_idx, 0),
        jnp.array(0),
        jnp.where(
            jax.lax.eq(x_idx, K),
            jnp.array(K - 2),
            cand_start_idx,
        ),
    )
    y_positions_expanded = jnp.tile(jnp.expand_dims(yp, 0), (N, 1, 1))
    start_y = jnp.take_along_axis(
        y_positions_expanded, jnp.expand_dims(start_idx2, 2), axis=2
    ).squeeze(2)
    end_y = jnp.take_along_axis(
        y_positions_expanded, jnp.expand_dims(start_idx2 + 1, 2), axis=2
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def cosine_alpha_schedule(n_timesteps, s=0.008):
    steps = n_timesteps + 2
    x = np.linspace(0, steps, steps)[1:-1]
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(1.0 - alphas, a_min=0.001, a_max=0.9946)
    return alphas


def linear_alpha_schedule(n_timesteps, b_min=1e-04, b_max=0.02):
    return 1.0 - jnp.linspace(b_min, b_max, n_timesteps)


def get_alphas(config):
    if config.name == "cosine":
        return cosine_alpha_schedule(config.n_timesteps, config.cosine_s)
    elif config.name == "linear":
        return linear_alpha_schedule(config.n_timesteps, config.b_min, config.b_max)
    raise ValueError("could not find correct schedule")


@dataclasses.dataclass
class NoiseScheduleConfig:
    name: str
    n_timesteps: int
    cosine_s: float = 0.008
    b_min: float = 1e-04
    b_max: float = 0.02
    Tmax: float = 1.0


class NoiseSchedule:
    """
    Code taken and adopted from:
    https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_jax.py#L7
    """

    def __init__(self, config):
        self._config = config
        self._alphas = get_alphas(config)
        self.n_diffusions = len(self._alphas)
        self.Tmax = config.Tmax
        self._t_array = jnp.linspace(0.0, self.Tmax, self.n_diffusions + 1)[1:]
        self.T0 = self._t_array[0]

        self._betas = 1.0 - self._alphas
        self._alphas_cumprod = jnp.cumprod(self._alphas)
        self._sqrt_alphas_cumprod = jnp.sqrt(self._alphas_cumprod)
        self._sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self._alphas_cumprod)
        self._log_sqrt_alphas_cumprod = 0.5 * jnp.log(self._alphas_cumprod)

    def t_array(self, t):
        ts = self._t_array[t]
        return ts

    def alphas_cumprod(self, t):
        ret = interpolate_fn(
            t.reshape((-1, 1)),
            self._t_array.reshape((1, -1)),
            self._alphas_cumprod.reshape((1, -1)),
        )
        return ret.reshape(-1)

    def alphas(self, t):
        ret = interpolate_fn(
            t.reshape((-1, 1)),
            self._t_array.reshape((1, -1)),
            self._alphas.reshape((1, -1)),
        )
        return ret.reshape(-1)

    def betas(self, t):
        ret = interpolate_fn(
            t.reshape((-1, 1)),
            self._t_array.reshape((1, -1)),
            self._betas.reshape((1, -1)),
        )
        return ret.reshape(-1)

    def log_sqrt_alphas_cumprod(self, t):
        ret = interpolate_fn(
            t.reshape((-1, 1)),
            self._t_array.reshape((1, -1)),
            self._log_sqrt_alphas_cumprod.reshape((1, -1)),
        )
        return ret.reshape(-1)

    def sqrt_alphas_cumprod(self, t):
        return jnp.exp(self.log_sqrt_alphas_cumprod(t))

    def scale(self, t):
        return jnp.sqrt(1.0 - jnp.exp(2.0 * self.log_sqrt_alphas_cumprod(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.log_sqrt_alphas_cumprod(t)
        log_std = 0.5 * jnp.log(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        log_alpha = -0.5 * jnp.logaddexp(jnp.zeros((1,)), -2.0 * lamb)
        t = interpolate_fn(
            log_alpha.reshape((-1, 1)),
            jnp.flip(self._log_sqrt_alphas_cumprod, 0).reshape((1, -1)),
            jnp.flip(self._t_array, 0).reshape((1, -1)),
        )
        return t


@dataclasses.dataclass
class DDPMConfig:
    n_dim: int
    n_diffusions: int
    solver_n_steps: int
    solver_order: int
    use_prior_loss: bool
    use_likelihood_loss: bool


class DDPM(Guide, hk.Module):
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
        self._n_diffusions = config.n_diffusions

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
        if self._config.use_prior_loss:
            prior_kl = self._prior_kl(z)
            evidence -= prior_kl
        if self._config.use_likelihood_loss:
            lp = self._log_likelihood(z, is_training)
            evidence += lp
        return evidence

    def _prior_kl(self, y):
        tT = jnp.atleast_1d(self._noise_schedule.t_array(self._n_diffusions - 1))
        q_T = self.forward_process(y, tT)
        p_T = tfd.Independent(tfd.Normal(jnp.zeros_like(y), 1), 1)
        kl = q_T.kl_divergence(p_T)
        return kl

    def _log_likelihood(self, y, is_training):
        t0 = jnp.atleast_1d(self._noise_schedule.t_array(0))
        z = self.forward_process(y, t0).sample(seed=hk.next_rng_key())
        sc = self._score_model(y, jnp.zeros(y.shape[0]), is_training)
        mu_inner = z - sc * (self._noise_schedule.betas(t0) / self._noise_schedule.scale(t0))
        mean = mu_inner / jnp.sqrt(self._noise_schedule.alphas(t0))
        scale = jnp.sqrt(self._noise_schedule.betas(t0))
        px = tfd.Independent(tfd.Normal(mean, scale), 1)
        lp = px.log_prob(y)
        return lp

    def _diffusion_loss(self, y, is_training):
        t = random.choice(
            key=hk.next_rng_key(),
            a=self._noise_schedule._t_array,
            shape=(y.shape[0],),
        )
        noise = jax.random.normal(hk.next_rng_key(), y.shape)
        perturbed_y = (
            self._noise_schedule.sqrt_alphas_cumprod(t).reshape(-1, 1) * y
            + (1.0 - self._noise_schedule.sqrt_alphas_cumprod(t)).reshape(-1, 1) * noise
        )
        eps = self._score_model(perturbed_y, t, is_training)
        loss = jnp.sum(jnp.square(noise - eps), axis=-1)
        return loss

    def forward_process(self, y, t):
        return tfd.Independent(
            tfd.Normal(
                self._noise_schedule.sqrt_alphas_cumprod(t) * y,
                self._noise_schedule.scale(t),
            ),
            1,
        )

    def sample(self, sample_shape=(1,), is_training=True):
        return self._sample_fn(sample_shape, is_training)
