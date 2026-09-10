# ensemble_sweep_dv

Degree-2 mixing capacity of the discrete-variable (tilted transverse-field Ising) reservoir at n = 4, d = 2, as a function of measurement-ensemble size (shots per readout setting, from 1e4 to 1e12 plus the exact infinite-shot limit) across all three encodings, each point from a 300-trial search-then-holdout study over 20 holdout seeds.

Task `optuna_mixing_capacity_qrc_tilted_tfim`.

| | |
|---|---|
| Run ID | `0904224422` |
| Jobs | 18 |
| Raw results | `data/Results_run/optuna_mixing_capacity_qrc_tilted_tfim_results_0904224422/` |
| Aggregated results | `data/Results_averaged/optuna_mixing_capacity_qrc_tilted_tfim_results_0904224422_combined.pkl` |
| Read by | Figure 6 |

Hyperparameter search budget: 300 trials, 30 random startup trials.

## Grid

| Axis | Values |
|---|---|
| `n` | 4 |
| `d` | 2 |
| `encoding_mode` | dense, fill, one_to_one |
| `n_shots` | 10000, 1000000, 100000000, 10000000000, 1000000000000, inf |

The grid file above argues for its 300-trial search budget by comparing it against 100-trial studies at the same operating point. Those 100-trial studies are not part of this deposit.

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/finite_shots/ensemble_sweep_dv/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_optuna.py --config ../experiments/finite_shots/ensemble_sweep_dv/grid.yaml --run-id 0904224422
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
