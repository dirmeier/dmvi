RESULTS__ = "results"
CONFIG__ = "config"
DATA_CONFIG__ = "data_config"

RESULTS_PATH__ = config[RESULTS__].rstrip("/") + "/"
CONFIG_PATH__ = config[CONFIG__].rstrip("/") + "/"
DATA_CONFIG_PATH__ = config[DATA_CONFIG__].rstrip("/") + "/"

seed = [1, 2, 3, 4, 5]

rule all:
    input:
        expand(CONFIG_PATH__ + "{model}.py", model=["advi", "ddpm", "flow"]),
        expand(DATA_CONFIG_PATH__ + "{data}.py", data=["mean_model",
                                                       "hierarchical_model_1",
                                                       "hierarchical_model_2",
                                                       "hierarchical_model_3",
                                                       "hierarchical_model_4",
                                                       "hierarchical_model_5",
                                                       "mixture_model"]),
        # mean model
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-{outfile}.pkl",
               data=["mean_model"], data_ndim=[5], data_nsamples=[100, 1000], n_diffusions=[50, 100], solver_steps=[20, 10], solver_order=[1, 3], seed=seed, outfile=["params", "posteriors", "results"]),
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-{outfile}.pkl",
               data=["mean_model"], data_ndim=[5], data_nsamples=[100, 1000], model=["advi", "flow"], seed=seed, outfile=["params", "posteriors", "results"]),

        # mixture model
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-{outfile}.pkl",
               data=["mixture_model", "mixture_with_scale_model", "mixture_with_scale_and_pi_model"], data_ndim=[2], data_nsamples=[100, 1000], n_diffusions=[50, 100], solver_steps=[20, 10], solver_order=[1, 3], seed=seed, outfile=["params", "posteriors", "results"]),
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-{outfile}.pkl",
               data=["mixture_model", "mixture_with_scale_model", "mixture_with_scale_and_pi_model"], data_ndim=[2], data_nsamples=[100, 1000], model=["advi", "flow"], seed=seed, outfile=["params", "posteriors", "results"]),

        # hierarchical models
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-{outfile}.pkl",
               data=[f"hierarchical_model_{idx}" for idx in range(10)], data_ndim=[1], data_nsamples=[100, 1000], n_diffusions=[50, 100], solver_steps=[20, 10], solver_order=[1, 3], seed=seed, outfile=["params", "posteriors", "results"]),
        expand(RESULTS_PATH__ + "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-{outfile}.pkl",
               data=[f"hierarchical_model_{idx}" for idx in range(10)], data_ndim=[1], data_nsamples=[100, 1000], model=["advi", "flow"], seed=seed, outfile=["params", "posteriors", "results"]),

ruleorder: run_ddpm > run_vi
ruleorder: run_analyze_ddpm > run_analyze_vi


rule run_ddpm:
    input:
        config = CONFIG_PATH__ + "ddpm.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py"
    output:
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-posteriors.pkl",
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-params.pkl"
    shell: " \
            python 01-main.py \
                --outdir={RESULTS_PATH__}/{wildcards.data} \
                --config={input.config} \
                --data_config={input.data_config} \
                --config.rng_seq_key={wildcards.seed} \
                --config.model.diffusion_model.n_diffusions={wildcards.n_diffusions} \
                --config.model.diffusion_model.solver_n_steps={wildcards.solver_steps} \
                --config.model.diffusion_model.solver_order={wildcards.solver_order} \
                --data_config.data.n_dim={wildcards.data_ndim} \
                --data_config.data.n_samples={wildcards.data_nsamples}"

rule run_vi:
    input:
        config = CONFIG_PATH__ + "{model}.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py"
    output:
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-posteriors.pkl",
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-params.pkl"
    shell: " \
            python 01-main.py  \
                    --outdir={RESULTS_PATH__}/{wildcards.data} \
                    --config={input.config} \
                    --data_config={input.data_config} \
                    --config.rng_seq_key={wildcards.seed} \
                    --data_config.data.n_dim={wildcards.data_ndim}\
                    --data_config.data.n_samples={wildcards.data_nsamples}"

rule run_analyze_ddpm:
    input:
        exact = RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-advi-seed_1-posteriors.pkl",
        approximate = RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-posteriors.pkl",
    output:
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-ddpm-n_diffusions_{n_diffusions}-solver_n_steps_{solver_steps}-solver_order_{solver_order}-seed_{seed}-results.pkl",
    shell: " \
            python 02-analyse_vi_results.py  \
                    --approximate={input.approximate} \
                    --exact={input.exact}"

rule run_analyze_vi:
    input:
        exact = RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-advi-seed_1-posteriors.pkl",
        approximate = RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-posteriors.pkl",
    output:
        RESULTS_PATH__ + \
            "{data}/{data}-n_dim_{data_ndim}-n_samples_{data_nsamples}-{model}-seed_{seed}-results.pkl",
    shell: " \
            python 02-analyse_vi_results.py  \
                    --approximate={input.approximate} \
                    --exact={input.exact}"
