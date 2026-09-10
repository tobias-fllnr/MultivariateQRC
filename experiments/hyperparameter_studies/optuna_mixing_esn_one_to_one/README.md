# optuna_mixing_esn_one_to_one

Two-stage search-then-holdout tuning of input scaling, spectral radius, leak rate and bias scale for the degree-2 mixing capacity of the classical echo state network baseline under local encoding, reporting the score on realizations the search never saw.

Task `optuna_mixing_capacity_esn`.

| | |
|---|---|
| Run ID | `0903151603` |
| Jobs | 14 |
| Raw results | `data/Results_run/optuna_mixing_capacity_esn_results_0903151603/` |
| Aggregated results | `data/Results_averaged/optuna_mixing_capacity_esn_results_0903151603_combined.pkl` |
| Read by | Figure 2, Figure S1 |

## Grid

| Axis | Values |
|---|---|
| `n` | 2, 3, 4, 5, 6 |
| `d` | 2, 3, 4, 5 |
| `encoding_mode` | one_to_one |

Filter: `n >= d`

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/hyperparameter_studies/optuna_mixing_esn_one_to_one/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_optuna.py --config ../experiments/hyperparameter_studies/optuna_mixing_esn_one_to_one/grid.yaml --run-id 0903151603
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
