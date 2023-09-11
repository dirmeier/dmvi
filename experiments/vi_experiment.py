import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp
from jax import random as jr
from optax import apply_updates

from data_and_models.generator import data_and_model_fn
from experiments import guides
from experiments.experiment import Experiment


class VIExperiment(Experiment):
    def __init__(self, FLAGS, run_name):
        super().__init__(FLAGS, run_name)

    def get_guide_and_params(self, init_params):
        logging.info(f"initializing model {self.config.name}")
        model_fn = guides.make_model(self.config)

        flat_z, _ = jax.flatten_util.ravel_pytree(init_params)
        model = model_fn(n_dim=len(flat_z))
        params = model.init(
            next(self.rng_seq),
            method="evidence",
            z=jr.normal(next(self.rng_seq), shape=(100, len(flat_z)), dtype=jnp.float32),
        )
        return model, params

    def fit(self):
        data_and_priors, train_itr, logprob_fn, init_params = self.get_data()
        guide, params = self.get_guide_and_params(init_params)

        n_params_leaves = jax.tree_util.tree_map(lambda x: np.prod(x.shape), params)
        n_params = jax.tree_util.tree_reduce(jnp.add, n_params_leaves)
        logging.info(f"total number of parameters: {n_params}")

        (params, losses), time = self.time(
            "fitting", lambda: self._fit(logprob_fn, guide, params, train_itr)
        )

        self.save(
            self.training_outfile(),
            {
                "params": params,
                "n_params": n_params,
                "filename": self.file_name(),
                "losses": losses,
                "data": data_and_priors[0]._asdict(),
                "priors": data_and_priors[1]._asdict(),
                "elapsed_time_for_training": time,
            },
        )

        return params, losses

    def _fit(self, logprob_fn, guide, params, train_itr):
        optimizer = self.get_optimizer()

        @jax.jit
        def step(rng, params, state, **batch):
            def loss_fn(params):
                sample_rng, evidence_rng = jr.split(rng)
                zs = guide.apply(
                    params,
                    rng=sample_rng,
                    method="sample",
                    sample_shape=(self.config.training.n_mc_samples,),
                )

                def _fn(z):
                    evidence = guide.apply(params, rng=evidence_rng, z=z, method="evidence")
                    lp = logprob_fn(z, **batch)
                    loss = -jnp.sum(lp) - jnp.sum(evidence)
                    return loss

                losses = jax.vmap(_fn)(zs)
                return jnp.mean(losses)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = apply_updates(params, updates)
            return new_params, new_state, loss

        state = optimizer.init(params)

        losses = np.zeros(self.config.training.n_iter)
        logging.info("training model")
        best_params = params
        best_loss = jnp.inf

        early_stopper = EarlyStopping(
            min_delta=self.config.training.early_stopping_delta,
            patience=self.config.training.early_stopping_patience,
        )
        for i in range(self.config.training.n_iter):
            loss = 0.0
            for j in range(train_itr.num_batches):
                batch = train_itr(j)
                params, state, batch_loss = step(next(self.rng_seq), params, state, **batch)
                loss += batch_loss
            losses[i] = loss
            _, early_stop = early_stopper.update(loss)

            if not jnp.isnan(loss) and loss < best_loss:
                best_params = params
                best_loss = loss
            if jnp.isnan(loss):
                logging.warning("breaking prematurely due to nan loss")
                break
            if early_stop.should_stop:
                logging.info(f"met early stopping criterion at {i}th iteration")
                break

        logging.info(f"finished after {i}th iteration")
        losses = jnp.vstack(losses)[: (i + 1), :]
        return best_params, losses

    def sample(self, checkpoint, **kwargs):
        params, train_itr = self._load(checkpoint)
        data_and_priors, *_, logprob_fn, init_params = data_and_model_fn(self.data_config)
        guide, _ = self.get_guide_and_params(init_params)

        z_hat = guide.apply(
            params,
            rng=next(self.rng_seq),
            method="sample",
            sample_shape=(self.config.sampling.sample_size,),
        )

        z_hat_mean = jnp.mean(z_hat, 0)
        unconstrained_prior = data_and_priors[1].unconstrained
        logging.info(f"mean z_hat: {z_hat_mean}")
        logging.info(f"unconstrained prior: {unconstrained_prior}")
        mse = jnp.mean(jnp.square(unconstrained_prior - z_hat_mean))
        logging.info(f"mse: {mse}")

        return z_hat

    def get_optimizer(self):
        optimizer = optax.adamw(
            learning_rate=self.config.training.optimizer.learning_rate,
            b1=self.config.training.optimizer.b1,
            b2=self.config.training.optimizer.b2,
            weight_decay=self.config.training.optimizer.weight_decay,
        )
        return optimizer
