import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "advi"
    config.rng_seq_key = 23

    config.model = new_dict()

    config.training = new_dict(
        n_iter=2000,
        early_stopping_patience=10,
        early_stopping_delta=10,
        n_mc_samples=5,
        optimizer=new_dict(
            name="adamw",
            learning_rate=0.001,
            b1=0.9,
            b2=0.999,
            weight_decay=0.0001,
        )
    )

    config.sampling = new_dict(
        sample_size=20000,
    )

    return config
