from jax import random
from rmsyutls import as_batch_iterator


def data_and_model_fn(config):
    logprob_fn, data_fn, init_params = config.data.model_fn(n_dim=config.data.n_dim, n_samples=config.data.n_samples)

    data_key, batch_key, seed = random.split(random.PRNGKey(config.data.rng_key), 3)
    data, prior = data_fn(data_key)

    train_itr = as_batch_iterator(
        rng_key=batch_key,
        data=data,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_data,
    )

    return (data, prior), train_itr, logprob_fn, init_params
