import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "nuts"
    config.rng_seq_key = 23

    config.model = new_dict()

    config.sampling = new_dict(
        # We use the slice sampler of BlackJAX
        # n_samples includes burnin
        n_samples=10000,
        n_warmup=5000,
        n_chains=4,
    )

    return config
