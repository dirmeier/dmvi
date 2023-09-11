# Diffusion models for probabilistic programming

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

## About

This repository implements the Python code for the NeurIPS workshop submission "Diffusion models for probabilistic programming".
Folder structure is as following:

- `configs` contains configuration files for the different inferential algorithms,
- `data_and_models` contains generative experimental models used for validating the inferential algorithms,
- `dmvi` contains the soucre code of the developed method and baseline implementations,
- `experiments` contains source code with the logic to run the experiments,
- `.*py` files are entry point scripts that execute the experiments.

## Installation

To run the experiments, we make use of venv for dependency management and Snakemake as workflow manager.
To install all requires dependencies and setup an environment via

```bash
python -m venv <envname>
<path/to/envname>/bin/activate {envname}
pip install -r requirements.txt
```

## Usage

You can either run experiments manually or use Snakemake to run everything in an automated fashion.

### Manual execution (not recommended)

If you want to run jobs manually, call either of

```bash
# runs flow/advi/nuts
python 01-main.py  \
  --outdir=results/mean_model \
  --config=configs/{flow/advi/nuts/slice}.py \
  --data_config=data_and_models/{mean_model/mixture_model/...}.py \
  --config.rng_seq_key=3 \
  --data_config.data.n_dim=100 \
  --data_config.data.n_samples=100


 # runs ddpm
python 01-main.py  \
  --outdir=results/mean_model \
  --config=configs/ddpm.py \
  --data_config=data_and_models/mean_model.py \
  --config.rng_seq_key=3 \
  --data_config.data.n_dim=100 \
  --data_config.data.n_samples=100 \
  --config.model.diffusion_model.n_diffusions=100 \
  --config.model.diffusion_model.solver_n_steps=20 \
  --config.model.diffusion_model.solver_order=1
```

### Automatic execution (recommended)

If you run all experiments from the manuscript and its appendix you can do it automatically using Snakemake.

On a HPC cluster you can use

```bash
snakemake --cluster {sbatch/qsub/bsub} --jobs N_JOBS  --configfile=snake_config.yaml
```

where `--cluster {sbatch/qsub/bsub}` specifies the command your cluster uses for job management and `--jobs N_JOBS` sets the number of jobs submitted at the same time.
For instance, to run on a SLURM cluster:

```bash
snakemake --cluster "sbatch --mem-per-cpu=4096 --time=4:00:00" --jobs 100  --configfile=snake_config.yaml
```

In the above scenario, Snakemake would run 100 jobs with 4Gb memory and a time limit of 4h each and resubmit jobs once less than 100 jobs are queued/running.
We ran all experiments using these resources.

On a single desktop computer, run all experiments sequentially via

```bash
snakemake  --configfile=snake_config.yaml
```

## Citation

If you want to cite the manuscript, please use the following quotation

```
@article{dirmeier2023diffusion,
  title={Diffusion models for probabilistic programming},
  author={Simon Dirmeier and Fernando Perez-Cruz},
  journal={ArXiv},
  year={2023}
}
```
## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
