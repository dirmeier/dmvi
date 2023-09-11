import logging
import pathlib
import pickle

import jax
from absl import app, flags
from jax import numpy as jnp

FLAGS = flags.FLAGS
flags.DEFINE_string("folder", None, "folder with posterior and parame files")
flags.mark_flags_as_required(["folder"])


def open_pickle(fl):
    with open(fl, "rb") as handle:
        d = pickle.load(handle)
    return d


def check_posterior(file):
    dic = open_pickle(file)
    nans_there = jnp.isnan(dic["samples"])
    num_nans = jnp.sum(nans_there)
    ratio_ans = num_nans / jnp.prod(jnp.asarray(nans_there.shape))
    if num_nans != 0:
        logging.warning(f"file {file} contains {ratio_ans} nans")


def check_params(file):
    dic = open_pickle(file)
    if "params" not in dic:
        return
    nans_there = jax.tree_map(jnp.isnan, dic["params"])
    num_nans = jax.tree_map(jnp.sum, nans_there)
    num_nans = jax.tree_util.tree_reduce(jnp.add, num_nans)
    if num_nans != 0:
        logging.warning(f"file {file} contains nans in training parameters")


def main(argv):
    del argv
    dir = pathlib.Path(FLAGS.folder)
    for p in dir.rglob("*"):
        if p.is_file():
            if "posteriors" in p.stem:
                check_posterior(p)
            elif "params" in p.stem:
                check_params(p)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
