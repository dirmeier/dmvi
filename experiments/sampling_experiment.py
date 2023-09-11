from functools import partial

import arviz as az
import blackjax as bj
import distrax
import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models.generator import data_and_model_fn
from experiments.experiment import Experiment


class SamplingExperiment(Experiment):
    def __init__(self, FLAGS, run_name):
        super().__init__(FLAGS, run_name)

    def fit(self):
        data_and_priors, train_itr, logprob_fn, init_params = self.get_data()
        self.save(
            self.training_outfile(),
            {
                "filename": self.file_name(),
                "data": data_and_priors[0]._asdict(),
                "priors": data_and_priors[1]._asdict(),
            },
        )

    def evaluate(self, checkpoint, **kwargs):
        (z_hat, traces), time = self.time("sampling", lambda: self.sample(checkpoint, **kwargs))

        self.save(
            self.posteriors_outfile(),
            {
                "samples": z_hat,
                "traces": traces,
                "elapsed_time_for_sampling": time,
            },
        )

    def sample(self, checkpoint, **kwargs):
        _, data = self._load(checkpoint)
        *_, logprob_fn, init_params = data_and_model_fn(self.data_config)
        len_theta = len(jax.flatten_util.ravel_pytree(init_params)[0])

        if self.FLAGS.config.name == "slice":
            lp_fn = partial(logprob_fn, y=data["y"])

            def lp(theta):
                return jax.vmap(lp_fn)(theta)

            z_hat, traces = self.sample_with_slice(lp, len_theta, **self.config.sampling)
        else:
            lp_fn = partial(logprob_fn, y=data["y"])
            lp = lambda theta: lp_fn(**theta)

            z_hat, traces = self.sample_with_nuts(lp, len_theta, **self.config.sampling)

        return z_hat, traces

    def sample_with_slice(self, lp, len_theta, n_chains, n_samples, n_warmup):
        initial_states = self._slice_init(len_theta, n_chains, lp)
        samples = tfp.mcmc.sample_chain(
            num_results=n_samples,
            current_state=initial_states,
            num_steps_between_results=1,
            kernel=tfp.mcmc.SliceSampler(lp, step_size=1, max_doublings=5),
            num_burnin_steps=n_warmup,
            trace_fn=None,
            seed=next(self.rng_seq),
        )
        # samples = samples[n_warmup:, ...]
        posterior = {f"var_{i}": samples[..., i].T for i in range(samples.shape[2])}

        traces = az.from_dict(posterior=posterior)
        thetas = samples.reshape(-1, len_theta)
        return thetas, traces

    def sample_with_nuts(self, lp, len_theta, n_chains, n_samples, n_warmup):
        def _inference_loop(rng_key, kernel, initial_state, n_samples):
            @jax.jit
            def _step(states, rng_key):
                keys = jax.random.split(rng_key, n_chains)
                states, infos = jax.vmap(kernel)(keys, states)
                return states, (states, infos)

            sampling_keys = jax.random.split(rng_key, n_samples)
            _, (states, infos) = jax.lax.scan(_step, initial_state, sampling_keys)
            return states, infos

        initial_states, kernel = self._nuts_init(len_theta, n_chains, lp)
        states, infos = _inference_loop(next(self.rng_seq), kernel, initial_states, n_samples)

        _ = states.position["theta"].block_until_ready()
        thetas = states.position["theta"][n_warmup:, :, :].reshape(-1, len_theta)
        traces = self._arviz_trace_from_states(states.position, infos, n_warmup)

        return thetas, traces

    def _slice_init(self, len_theta, n_chains, lp):
        initial_positions = distrax.Normal(jnp.zeros(len_theta), 1.0).sample(
            seed=next(self.rng_seq), sample_shape=(n_chains,)
        )

        return initial_positions

    def _nuts_init(self, len_theta, n_chains, lp):
        initial_positions = tfd.MultivariateNormalDiag(
            jnp.zeros(len_theta),
            jnp.ones(len_theta),
        ).sample(seed=next(self.rng_seq), sample_shape=(n_chains,))
        initial_positions = {"theta": initial_positions}

        init_keys = jr.split(next(self.rng_seq), n_chains)

        warmup = bj.window_adaptation(bj.nuts, lp)
        initial_states, kernel_params = jax.vmap(lambda seed, param: warmup.run(seed, param)[0])(
            init_keys, initial_positions
        )

        kernel_params = {k: v[0] for k, v in kernel_params.items()}
        _, kernel = bj.nuts(lp, **kernel_params)

        return initial_states, kernel

    @staticmethod
    def _arviz_trace_from_states(position, info, n_warmup):
        samples = {}
        for param in position.keys():
            ndims = len(position[param].shape)
            if ndims >= 2:
                samples[param] = jnp.swapaxes(position[param], 0, 1)[:, n_warmup:]
                divergence = jnp.swapaxes(info.is_divergent[n_warmup:], 0, 1)

            if ndims == 1:
                divergence = info.is_divergent
                samples[param] = position[param]

        trace_posterior = az.convert_to_inference_data(samples)
        trace_sample_stats = az.convert_to_inference_data(
            {"diverging": divergence}, group="sample_stats"
        )
        trace = az.concat(trace_posterior, trace_sample_stats)
        return trace
