# optuna_mixing_tfim_fill

Two-stage search-then-holdout tuning of encoding strength, coupling strength and decay rate for the degree-2 mixing capacity of the discrete-variable (tilted transverse-field Ising) reservoir under clustered encoding, reporting the score on realizations the search never saw.

Task `optuna_mixing_capacity_qrc_tilted_tfim`.

| | |
|---|---|
| Run ID | `0903012719` |
| Jobs | 14 |
| Raw results | `data/Results_run/optuna_mixing_capacity_qrc_tilted_tfim_results_0903012719/` |
| Aggregated results | `data/Results_averaged/optuna_mixing_capacity_qrc_tilted_tfim_results_0903012719_combined.pkl` |
| Read by | Figure 2, Figure S1 |

## Grid

| Axis | Values |
|---|---|
| `n` | 2, 3, 4, 5, 6 |
| `d` | 2, 3, 4, 5 |
| `encoding_mode` | fill |

Filter: `n >= d`

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/hyperparameter_studies/optuna_mixing_tfim_fill/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_optuna.py --config ../experiments/hyperparameter_studies/optuna_mixing_tfim_fill/grid.yaml --run-id 0903012719
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
