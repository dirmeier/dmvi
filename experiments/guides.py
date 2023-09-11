from functools import partial

import haiku as hk

from dmvi import Flow, continuous_ddpm, ddpm
from dmvi.baselines.advi import ADVI, ADVIConfig
from dmvi.baselines.flow import FlowConfig
from dmvi.nn.score_model import ScoreModel, ScoreModelConfig


def dpm_fn(config, n_dim):
    def _dm(method, **kwargs):
        score_model = ScoreModel(ScoreModelConfig(**{**config.score_model, "output_dim": n_dim}))
        noise_schedule = ddpm.NoiseSchedule(
            ddpm.NoiseScheduleConfig(
                **{
                    **config.noise_schedule,
                    "n_timesteps": config.diffusion_model.n_diffusions,
                }
            )
        )
        ddpm_config = ddpm.DDPMConfig(**{**config.diffusion_model, "n_dim": n_dim})
        model = ddpm.DDPM(
            config=ddpm_config,
            score_model=score_model,
            noise_schedule=noise_schedule,
        )
        return model(method, **kwargs)

    diffusion = hk.transform(_dm)
    return diffusion


def continuous_dpm_fn(config, n_dim):
    def _dm(method, **kwargs):
        score_model = ScoreModel(ScoreModelConfig(**{**config.score_model, "output_dim": n_dim}))
        noise_schedule = continuous_ddpm.NoiseSchedule(
            continuous_ddpm.NoiseScheduleConfig(**config.noise_schedule)
        )
        ddpm_config = continuous_ddpm.ContinuousDDPMConfig(
            **{**config.diffusion_model, "n_dim": n_dim}
        )
        model = continuous_ddpm.ContinuousDDPM(
            config=ddpm_config,
            score_model=score_model,
            noise_schedule=noise_schedule,
        )
        return model(method, **kwargs)

    diffusion = hk.transform(_dm)
    return diffusion


def flow_fn(config, n_dim):
    def _flow(method, **kwargs):
        flow = Flow(FlowConfig(**{"n_dim": n_dim, **config}))
        return flow(method, **kwargs)

    flow = hk.transform(_flow)
    return flow


def advi_fn(config, n_dim):
    def _advi(method, **kwargs):
        advi = ADVI(ADVIConfig(**{"n_dim": n_dim}))
        return advi(method, **kwargs)

    flow = hk.transform(_advi)
    return flow


def make_model(config):
    if config.name == "ddpm":
        return partial(dpm_fn, config=config.model)
    elif config.name == "continuous_ddpm":
        return partial(continuous_dpm_fn, config=config.model)
    elif config.name == "flow":
        return partial(flow_fn, config=config.model)
    elif config.name == "advi":
        return partial(advi_fn, config=config.model)
    raise ValueError(f"guide '{config.name}' not implemented/known")
