import os
import pickle
from abc import abstractmethod
from timeit import default_timer as timer

import haiku as hk
from absl import logging

from data_and_models.generator import data_and_model_fn


class Experiment:
    def __init__(self, FLAGS, run_name):
        self.FLAGS = FLAGS
        self.config = FLAGS.config
        self.data_config = FLAGS.data_config
        self.run_name = run_name
        self.rng_seq_key = self.config.rng_seq_key
        self.rng_seq = hk.PRNGSequence(self.rng_seq_key)

        logging.info(
            f"running experiment {self.data_config.name} "
            f"with model {self.config.name} and run name {self.run_name}"
        )

    def fit_and_evaluate(self):
        self.fit()
        logging.info("done fitting")
        self.evaluate(self.training_outfile())
        logging.info("done sampling")

    def time(self, fun_name, fn):
        start = timer()
        ret = fn()
        end = timer()
        logging.info(f"elapsed time for {fun_name}: {end - start}")
        return ret, end - start

    def get_data(self):
        data_and_priors, train_itr, logprob_fn, init_params = data_and_model_fn(self.data_config)
        return data_and_priors, train_itr, logprob_fn, init_params

    @abstractmethod
    def fit(self):
        pass

    def save(self, file_name, save_dict, **kwargs):
        with open(file_name, "wb") as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, checkpoint, **kwargs):
        z_hat, time = self.time("sampling", lambda: self.sample(checkpoint, **kwargs))

        self.save(
            self.posteriors_outfile(),
            {"samples": z_hat, "elapsed_time_for_sampling": time},
        )

    @abstractmethod
    def sample(self, checkpoint, **kwargs):
        pass

    def _load(self, outname):
        with open(outname, "rb") as handle:
            d = pickle.load(handle)
        return d.get("params", None), d["data"]

    def training_outfile(self):
        return self.file_name() + "-params.pkl"

    def posteriors_outfile(self):
        return self.file_name() + "-posteriors.pkl"

    def file_name(self):
        return os.path.join(self.FLAGS.outdir, self.run_name)
