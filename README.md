# Multivariate quantum reservoir computing with discrete and continuous variable systems

[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--6457-d45815.svg)](https://doi.org/10.18419/darus-6457)

Code and data accompanying

> Tobias Fellner, Jonas Merklinger and Christian Holm, *Multivariate quantum reservoir computing with discrete and continuous variable systems*.
> Institute for Computational Physics, University of Stuttgart.

## What this is

Everything needed to reproduce the paper's quantitative results: the simulation code for
both reservoir models and the classical baseline, the 32 runs behind the figures
as both raw per-seed output and aggregated results, the scripts that draw the figures, and
the numbers behind each drawn point as machine-readable CSV.

The paper studies two quantum reservoirs -- a continuous-variable network of damped
coupled harmonic oscillators, and a discrete-variable tilted transverse-field Ising chain
-- driven by multivariate inputs under three encodings, together with a classical echo
state network of matched size as a baseline. Two tasks are measured: the degree-2 mixing
capacity, which asks how much of the product of two input channels a reservoir's readout
can express, and one-step-ahead prediction of the Lorenz-63 system.

Reproduction comes at three depths, and `reproduce_all.sh` runs each of them:

| Tier | Command | From | Cost |
|---|---|---|---|
| figures | `./reproduce_all.sh --tier figures` | the aggregated results in `data/` | minutes |
| average | `./reproduce_all.sh --tier average` | the raw per-seed results in `data/Results_run/`, then the figures | minutes |
| simulate | `./reproduce_all.sh --tier simulate` | the grids: generates 59,514 jobs and stops | see below |

The deposit carries five main-text figures and three supplementary
ones. The paper's Figure 1 is a drawn schematic -- the reservoir-computing setup in its
panel (a) and the three encoding schemes in (b) to (d) -- with no underlying data. No
script produces it, and it is therefore not here. Its absence is the only gap between the
paper's figures and this deposit.

## Layout

```
.
|-- README.md                     this file
|-- LICENSE                       MIT, covering code/
|-- LICENSE-DATA                  CC BY 4.0, covering data/ and figures/
|-- requirements.txt              the 9 pinned Python dependencies
|-- reproduce_all.sh              the three-tier reproduction driver
|-- .gitignore                    what a run of this deposit leaves behind
|-- code/
|   |-- run_general_job.py        one simulation, plus its shell wrapper
|   |-- run_optuna_job.py         one hyperparameter study, plus its shell wrapper
|   |-- prepare_jobs.py           grid.yaml -> argument file and submission scripts
|   |-- average_runs_general.py   per-seed pickles -> aggregated pickle
|   |-- average_runs_optuna.py    per-study JSONs -> combined pickle
|   |-- utils/                    the reservoir models, tasks, metrics and grid machinery
|   `-- PlottingScripts/          one script per figure, plus extract_correlations.py
|-- experiments/
|   |-- parameter_scans/          12 grids, each with a README
|   |-- hyperparameter_studies/   18 grids, each with a README
|   `-- finite_shots/             2 grids, each with a README
|-- data/
|   |-- Results_run/              32 run directories of raw per-seed results
|   |-- Results_averaged/         32 aggregated result pickles, one per run
|   `-- SourceData/               the source-data CSVs behind the figures, and their README
`-- figures/                      the 8 figure PDFs, under the filenames the scripts write
```

`experiments/` sits beside `code/` rather than inside it because a `grid.yaml` is the
specification of a run rather than part of the program: `prepare_jobs.py` takes
`--config <any path>`, so nothing depends on where a grid lives. Every grid directory
carries a `README.md` stating what that run measures, its parameter axes, its run ID, the
files it produced and the figure that reads it.

## Prerequisites

- **Python 3.12.** The dependency pins below are the versions the results were produced
  with.
- **The nine pinned dependencies.** From the deposit root:

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

  `reproduce_all.sh` activates `.venv` if it finds one there, and otherwise falls back to
  whatever `python3` is on the path.
- **A LaTeX installation with the Computer Modern fonts.** Every figure script typesets its
  labels through LaTeX (`text.usetex`), so without one they fail with a LaTeX error rather
  than a missing-package message. A TeX Live install including `cm-super` and
  `dvipng`/`ghostscript` is enough.
- **bash 4 or newer.** Tier `average` falls through into tier `figures` with `;&`, which
  earlier bash does not parse. This matters mainly on macOS, whose `/bin/bash` is 3.2;
  install a current bash and run the script with it.

Nothing else is needed. There is no build step, no compiled extension and no configuration
file to fill in.

## Reproducing the figures

```bash
./reproduce_all.sh --tier figures
```

This reads the aggregated results in `data/Results_averaged/` and the source data in
`data/SourceData/`, and rewrites the 8 PDFs in `figures/` together with the
CSV behind each of them. It takes minutes and touches nothing else. One script,
`ensemble_sweep_mixing.py`, also drops a PNG preview beside its PDF; that preview is not
part of the deposited figure set, nothing reads it, and `reproduce_all.sh` deletes it once
the figures are drawn.

The deposited figures keep the filenames the scripts write rather than being renamed
`Figure1.pdf`, `Figure2.pdf` and so on, because the numbering belongs to the paper and the
filenames belong to the code. The map between them:

| Figure | File in `figures/` | Written by | What it shows |
|---|---|---|---|
| 2 | `combined_best_score_nNorm_False_dNorm_False.pdf` | `figure2_encoding_comparison_mixing.py` | Degree-2 mixing capacity of the tuned reservoirs against the number of sites and the input dimension, scaled by the readout dimension. |
| 3 | `mixing_capacity_comparison_heatmap_first_moment_2.pdf` | `figure3_heatmap_mixing.py` | Degree-2 mixing capacity over the encoding-strength / coupling-strength plane. |
| 4 | `lorenz_x1_encoding_comparison.pdf` | `figure4_encoding_comparison_lorenz.py` | One-step-ahead Lorenz-63 prediction error against the number of encoded input dimensions. |
| 5 | `combined_1d_mixing_lorenz.pdf` | `figure5_combined_mixing_lorenz.py` | Mixing capacity (a) and Lorenz-63 prediction error (b) along the coupling sweep, each beside the quantum resource of its platform, annotated with the correlation statistics. |
| 6 | `ensemble_sweep_mixing.pdf` | `ensemble_sweep_mixing.py` | Degree-2 mixing capacity against the size of the measurement ensemble, up to the exact infinite-shot limit. |
| S1 | `combined_best_score_nNorm_True_dNorm_True.pdf` | `figure2_encoding_comparison_mixing.py` | Figure 2's data under the alternative normalisation: capacity per input pair, with no readout-dimension factor. Written by the same script as Figure 2. |
| S2 | `nrmse_comparison_heatmap.pdf` | `figureS2_heatmap_lorenz.py` | One-step-ahead Lorenz-63 prediction error over the encoding-strength / coupling-strength plane. |
| S3 | `resource_performance_scatter.pdf` | `figureS3_resource_scatter.py` | Per-seed quantum resource against per-seed performance at fixed coupling strength, the data behind Figure 5's across-seed correlation. |

Each script can also be run on its own, from `code/`:

```bash
cd code
python PlottingScripts/figure3_heatmap_mixing.py
```

## Re-aggregating from the raw data

```bash
./reproduce_all.sh --tier average
```

This goes one level deeper: it re-derives everything in `data/Results_averaged/` from the
per-seed pickles and per-study JSONs in `data/Results_run/`, re-derives the two correlation
CSVs from the same raw data, and then redraws the figures. The aggregated files it writes
have exactly the names the deposited ones have, so the figure step that follows reads what
was just recomputed.

It costs minutes rather than hours: the work is reading files, not simulating. The
twelve parameter scans hold 4,940 per-seed pickles each, and
`extract_correlations.py` reads all of them again, one at a time, which is the single
slowest step of the tier.

Two things about the order are worth knowing, because getting either wrong produces a
plausible-looking figure drawn from the wrong numbers.

`PlottingScripts/extract_correlations.py` is the only script in the deposit that reads the
raw per-seed data. It writes `data/SourceData/Figure5_correlations.csv`, which supplies the
correlation coefficients Figure 5 prints in its panel titles, and
`data/SourceData/FigureS3.csv`, which is the entire content of Figure S3. Both figure
scripts read those files rather than recomputing them, which is what lets tier `figures`
run without the raw data being present at all. In tier `average` the driver therefore runs
`extract_correlations.py` after the aggregation and before any figure script; in tier
`figures` the deposited CSVs are used as they stand.

The aggregation itself is per run, and the driver names the deposited run ID for each grid
explicitly:

```bash
cd code
python average_runs_general.py --config ../experiments/parameter_scans/<slug>/grid.yaml --run-id <run id>
```

The run ID is not optional here. Without it the averaging scripts look for a `runs.yaml`
ledger beside the grid, which records commits of a repository a reader of this deposit
cannot fetch and is deliberately not deposited. Each grid's own README states the run ID
its results carry, and the table further down lists all 32.

## Re-running the simulations

```bash
./reproduce_all.sh --tier simulate
```

This expands every grid into an argument file and three submission scripts -- a sequential
local runner, an HTCondor submit file and a Slurm array script, all written under
`code/jobs/` -- and then stops, printing the three commands. It deliberately does not
choose a backend or start anything.

| Tree | Grids | Jobs | |
|---|---|---|---|
| `experiments/parameter_scans/` | 12 | 59,280 | one simulation per grid point: encoding strength x coupling strength x seed |
| `experiments/hyperparameter_studies/` | 18 | 198 | one two-stage study per job |
| `experiments/finite_shots/` | 2 | 36 | one two-stage study per ensemble size |
| **total** | **32** | **59,514** | |

Be aware of what that commits you to. The cost per job is wildly uneven: one
discrete-variable study at six qubits runs about fourteen hours, because the master
equation is integrated over a Hilbert space that grows exponentially with the number of
qubits, while the matching continuous-variable study finishes in minutes -- the Gaussian
reservoir evolves a covariance matrix whose size grows only quadratically. The
59,514 jobs in total are a cluster-scale undertaking, not a
laptop one, and the results of running them are already deposited under
`data/Results_run/`.

Each generated run allocates a fresh run ID from the current date and time, so re-running
this tier never overwrites the deposited results; it writes new directories beside them.
To aggregate a run of your own, pass its run ID to the averaging script as shown above.

### The hyperparameter-search protocol

Each hyperparameter study is one job and runs in two stages, and the deposited grids
mostly do not restate the defaults, so they are written out here.

**Search.** 100 trials of Optuna's tree-structured Parzen estimator, the
first 10 of them drawn at random before the estimator starts
modelling. Each trial is scored on the mean over the ten tuning seeds
(0-9) -- the degree-2 mixing capacity for the mixing task, the
validation-block prediction error for the Lorenz task. The sampler is seeded from the grid
point itself, so a study repeats exactly.

**Held-out evaluation.** The hyperparameters of the best trial are then evaluated once
more, on twenty realizations (1000-1019) that are disjoint from the tuning seeds
and, for the Lorenz task, on a test block the search never saw either. **That held-out
score, not the search optimum, is the number the figures report**, as its mean and its
sample standard deviation across those realizations. The search's own optimum is kept
in the result alongside it, so the gap between the two is readable rather than hidden.

A grid may declare its own budget with a top-level `n_trials:`, which its own README then states: `ensemble_sweep_cv`, `ensemble_sweep_dv` search 300 trials with 30 startup trials. Every other study runs the default above.

## The deposited runs

One row per run. The directory holds the grid and a README describing it; the run ID names
the directory under `data/Results_run/` and appears in the aggregated pickle's filename.

| Directory | Run ID | Jobs | Read by |
|---|---|---|---|
| `experiments/finite_shots/ensemble_sweep_cv` | `0904224422` | 18 | Figure 6 |
| `experiments/finite_shots/ensemble_sweep_dv` | `0904224422` | 18 | Figure 6 |
| `experiments/hyperparameter_studies/optuna_lorenz_esn_dense` | `0903163853` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_esn_fill` | `0903163854` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_esn_one_to_one` | `0903163855` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_gaussian_dense` | `0903163854` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_gaussian_fill` | `0903163855` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_gaussian_one_to_one` | `0903163856` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_tfim_dense` | `0903163855` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_tfim_fill` | `0903163856` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_lorenz_tfim_one_to_one` | `0903163857` | 6 | Figure 4 |
| `experiments/hyperparameter_studies/optuna_mixing_esn_dense` | `0903151601` | 20 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_esn_fill` | `0903151602` | 14 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_esn_one_to_one` | `0903151603` | 14 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_gaussian_dense` | `0903012721` | 20 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_gaussian_fill` | `0903012720` | 14 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_gaussian_one_to_one` | `0903012719` | 14 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_tfim_dense` | `0903012720` | 20 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_tfim_fill` | `0903012719` | 14 | Figure 2, Figure S1 |
| `experiments/hyperparameter_studies/optuna_mixing_tfim_one_to_one` | `0903012718` | 14 | Figure 2, Figure S1 |
| `experiments/parameter_scans/lorenz_gaussian_dense` | `0906114152` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/lorenz_gaussian_fill` | `0906114151` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/lorenz_gaussian_one_to_one` | `0906114150` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/lorenz_tfim_dense` | `0906114153` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/lorenz_tfim_fill` | `0906114152` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/lorenz_tfim_one_to_one` | `0906114151` | 4,940 | Figure 5, Figure S2, Figure S3 |
| `experiments/parameter_scans/mixing_gaussian_dense` | `0905130005` | 4,940 | Figure 3, Figure 5, Figure S3 |
| `experiments/parameter_scans/mixing_gaussian_fill` | `0905130004` | 4,940 | Figure 3, Figure 5, Figure S3 |
| `experiments/parameter_scans/mixing_gaussian_one_to_one` | `0905130003` | 4,940 | Figure 3, Figure 5, Figure S3 |
| `experiments/parameter_scans/mixing_tfim_dense` | `0905130006` | 4,940 | Figure 3, Figure 5, Figure S3 |
| `experiments/parameter_scans/mixing_tfim_fill` | `0905130005` | 4,940 | Figure 3, Figure 5, Figure S3 |
| `experiments/parameter_scans/mixing_tfim_one_to_one` | `0905130004` | 4,940 | Figure 3, Figure 5, Figure S3 |

## The code

| Module | |
|---|---|
| `code/run_general_job.py` | One simulation: builds the reservoir, drives it with the task's input, reads it out, and writes one per-seed result pickle. Every parameter arrives as a positional argument, in the order the grid declares its axes. |
| `code/run_optuna_job.py` | One two-stage hyperparameter study: a search over the tuning seeds followed by a single evaluation of the selected hyperparameters on the held-out seeds. Writes one JSON per grid point. |
| `code/prepare_jobs.py` | Expands a grid.yaml into an argument file and three submission scripts -- a sequential local runner, an HTCondor submit file and a Slurm array script -- and records the resolved grid in the results directory. |
| `code/average_runs_general.py` | Aggregates the per-seed pickles of a parameter scan over its seeds, grouping by every grid axis except the seed. |
| `code/average_runs_optuna.py` | Combines the per-study JSONs of a hyperparameter run into one table. |
| `code/utils/qrc_gaussian.py` | The continuous-variable reservoir: a network of damped, coupled harmonic oscillators, evolved through its covariance matrix, read out by homodyne detection of the position quadratures. |
| `code/utils/qrc_spin.py` | The discrete-variable reservoir: a tilted transverse-field Ising chain under amplitude damping, evolved as a Lindblad master equation, read out through Pauli expectation values. |
| `code/utils/esn.py` | The classical echo state network baseline, sized and driven to match the discrete-variable reservoir's readout dimension and driven fraction. |
| `code/utils/ipc.py` | Information processing capacity: the orthogonal-polynomial decomposition that turns a reservoir's states into the mixing capacity the paper reports. |
| `code/utils/lorenz63.py` | The Lorenz-63 system, integrated with a fourth-order Runge-Kutta scheme, and the trajectories the prediction task is built on. |
| `code/utils/prediction.py` | The linear readout: ridge regression from reservoir states onto the target, and the normalised root-mean-square error over the ten-step prediction horizon. |
| `code/utils/data.py` | The random input sequences the mixing-capacity task drives the reservoirs with. |
| `code/utils/shots.py` | Finite measurement ensembles: the exact sampling distribution of each readout -- a Wishart draw for the continuous-variable covariances, a multinomial draw for the Pauli measurements -- rather than an additive-noise surrogate. |
| `code/utils/shotguard.py` | Refuses to aggregate results whose measurement-ensemble size or search budget differ, which would otherwise be pooled silently into one output. |
| `code/utils/correlation.py` | The two correlation measures Figure 5 annotates: one across the coupling sweep, one across seeds at fixed coupling, with the Fisher pooling of the latter. |
| `code/utils/gridconfig.py` | Reads a grid.yaml, expands its axes into ordered parameter combinations, and checks that the axis order matches the target script's argument order. |
| `code/utils/provenance.py` | Allocates a run ID and records the resolved grid, the job count and the timestamp in run_config.json beside the results. |
| `code/PlottingScripts/_common.py` | Shared by every figure script: the deposit's paths, the plotting style, the run each figure reads, and the source-data writer. |
| `code/PlottingScripts/extract_correlations.py` | The one script that reads the raw per-seed data, deriving the correlation statistics Figure 5 annotates and the scatter Figure S3 plots. |

Every module listed is reachable from a script that produced a deposited result; there is
no dead code here. The grid file is the single specification of a run: `prepare_jobs.py`
expands it into jobs, `run_general_job.py` or `run_optuna_job.py` executes one job, and the
matching averaging script aggregates the results back into one file per run.

The order of the axes in a `grid.yaml` is load-bearing. Both job scripts read their
parameters as positional arguments, so the axis order is the argument order;
`utils/gridconfig.py` checks it against the target script's signature before any job is
written and refuses a grid whose axes are out of order.

## Known behaviour

`average_runs_general.average_results` takes its list of output keys from the first
per-seed result in a group rather than from the union over the group's seeds. A seed whose
degree-2 mixing capacity is exactly zero writes no `first_moment_breakdown_2` entry at all,
so for a group that mixes such a seed with normal ones, whether the aggregated output
carries `first_moment_breakdown_2_mean` and `first_moment_breakdown_2_std` depends on which
seed the directory listing happens to return first.

Re-aggregating can therefore make one of those keys appear or vanish relative to the
pickles deposited here. That is the order-dependence, not a computation error, and no
figure in the paper reads those keys.

## Licensing

| | |
|---|---|
| `code/` | MIT -- see `LICENSE` |
| `data/` and `figures/` | Creative Commons Attribution 4.0 International (CC BY 4.0) -- see `LICENSE-DATA` |

Both cover derivative work, and both ask only that the authors be credited. If you use the
code or the data, please cite the paper named at the top of this file.
