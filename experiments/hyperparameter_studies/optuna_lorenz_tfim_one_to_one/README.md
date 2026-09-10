# optuna_lorenz_tfim_one_to_one

Two-stage search-then-holdout tuning of encoding strength, coupling strength and decay rate for the one-step-ahead Lorenz-63 prediction error of the discrete-variable (tilted transverse-field Ising) reservoir under local encoding, for input dimension d in {1, 2, 3}, reporting the score on realizations the search never saw.

Task `optuna_lorenz63_qrc_tilted_tfim`.

| | |
|---|---|
| Run ID | `0903163857` |
| Jobs | 6 |
| Raw results | `data/Results_run/optuna_lorenz63_qrc_tilted_tfim_results_0903163857/` |
| Aggregated results | `data/Results_averaged/optuna_lorenz63_qrc_tilted_tfim_results_0903163857_combined.pkl` |
| Read by | Figure 4 |

## Grid

| Axis | Values |
|---|---|
| `n` | 6 |
| `d` | 1, 2, 3 |
| `target` | all_1, x_1 |
| `encoding_mode` | one_to_one |

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/hyperparameter_studies/optuna_lorenz_tfim_one_to_one/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_optuna.py --config ../experiments/hyperparameter_studies/optuna_lorenz_tfim_one_to_one/grid.yaml --run-id 0903163857
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
