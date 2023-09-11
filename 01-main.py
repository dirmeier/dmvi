import pathlib
from datetime import datetime

import jax
from absl import app, flags, logging
from jax.lib import xla_bridge
from ml_collections import config_flags

from experiments.sampling_experiment import SamplingExperiment
from experiments.vi_experiment import VIExperiment

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "training configuration", lock_config=False)
config_flags.DEFINE_config_file("data_config", None, "data configuration", lock_config=False)
flags.DEFINE_string("checkpoint", None, "parameter file of the trained guide")
flags.DEFINE_string("outdir", None, "out directory, i.e., place where results are written to")
flags.mark_flags_as_required(["config", "data_config", "outdir"])


def init_and_log_jax_env():
    tm = datetime.now().strftime("%Y-%m-%d-%H%M")
    logging.info("file prefix: %s", tm)
    logging.info("----- Checking JAX installation ----")
    logging.info(jax.devices())
    logging.info(jax.default_backend())
    logging.info(xla_bridge.get_backend().platform)
    logging.info("------------------------------------")
    return tm


def get_run_name(config, data_config):
    run_data_id = (
        f"{data_config.name}-n_dim_{data_config.data.n_dim}-n_samples_{data_config.data.n_samples}"
    )
    run_model_id = f"{config.name}"
    if "diffusion_model" in config.model and "n_diffusions" in config.model.diffusion_model:
        run_model_id += f"-n_diffusions_{config.model.diffusion_model.n_diffusions}"
    if "diffusion_model" in config.model:
        run_model_id += f"-solver_n_steps_{config.model.diffusion_model.solver_n_steps}-solver_order_{config.model.diffusion_model.solver_order}"
    run_name = run_data_id + "-" + run_model_id + f"-seed_{config.rng_seq_key}"
    return run_name


def main(argv):
    del argv

    outdir = pathlib.Path(FLAGS.outdir)
    run_name = get_run_name(FLAGS.config, FLAGS.data_config)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if not (outdir / run_name).exists():
        (outdir / run_name).mkdir(parents=True)
    logging.get_absl_handler().use_absl_log_file("absl_logging", (outdir / run_name))

    if FLAGS.config.name in ["nuts", "slice"]:
        experiment = SamplingExperiment(FLAGS, run_name)
    else:
        experiment = VIExperiment(FLAGS, run_name)
    experiment.fit_and_evaluate()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
