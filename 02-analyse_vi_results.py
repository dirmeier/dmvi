import pickle
import re

import chex
import jax
import pandas as pd
from absl import app, flags
from jax import numpy as jnp

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exact", None, "file with samples that are assumed to be 'exact', i.e., MCMC samples"
)
flags.DEFINE_string("approximate", None, "file with samples generated using variational ifnerence")
flags.mark_flags_as_required(["exact", "approximate"])


def open_pickle(fl):
    with open(fl, "rb") as handle:
        d = pickle.load(handle)
    return d


def check_params_and_priors(exact, approximate):
    chex.assert_trees_all_close(exact["data"]["y"], approximate["data"]["y"])
    chex.assert_trees_all_close(
        exact["priors"]["constrained"], approximate["priors"]["constrained"]
    )
    chex.assert_trees_all_close(
        exact["priors"]["unconstrained"], approximate["priors"]["unconstrained"]
    )


def parse_filename(filename):
    n_diffusions = n_solver_steps = n_solver_order = jnp.nan
    if "continuous_ddpm" in filename:
        reg = re.compile(r".*solver_n_steps_(\d+)-solver_order_(\d+).*")
        groups = reg.match(FLAGS.approximate)
        n_solver_steps = int(groups.group(1))
        n_solver_order = int(groups.group(2))
    elif "ddpm" in FLAGS.approximate:
        reg = re.compile(r".*n_diffusions_(\d+)-solver_n_steps_(\d+)-solver_order_(\d+).*")
        groups = reg.match(FLAGS.approximate)
        n_diffusions = int(groups.group(1))
        n_solver_steps = int(groups.group(2))
        n_solver_order = int(groups.group(3))
    return n_diffusions, n_solver_steps, n_solver_order


def as_df(params, samples):
    prior = params["exact"]["priors"]["unconstrained"]
    z_hat = samples["approximate"]["samples"]

    n_diffusions, n_solver_steps, n_solver_order = parse_filename(FLAGS.approximate)
    mse = jnp.mean(jnp.square(prior - z_hat))

    outputs = {k: [params["approximate"][k]] for k in ["n_params", "elapsed_time_for_training"]}
    outputs["elapsed_time_for_sampling"] = [samples["approximate"]["elapsed_time_for_sampling"]]
    outputs["mse"] = [mse]
    outputs["n_diffusions"] = [n_diffusions]
    outputs["n_solver_steps"] = [n_solver_steps]
    outputs["n_solver_order"] = [n_solver_order]

    df = pd.DataFrame.from_dict(outputs)
    return df


def main(argv):
    del argv

    samples_fl = {"approximate": FLAGS.approximate, "exact": FLAGS.exact}
    params_fl = jax.tree_map(lambda x: x.replace("posteriors", "params"), samples_fl)
    params = jax.tree_map(open_pickle, params_fl)
    samples = jax.tree_map(open_pickle, samples_fl)
    check_params_and_priors(**params)

    df = as_df(params, samples)
    outfile = FLAGS.approximate.replace("posteriors", "results")
    with open(outfile, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
