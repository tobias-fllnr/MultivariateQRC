# optuna_lorenz_gaussian_dense

Two-stage search-then-holdout tuning of encoding strength, coupling strength and decay rate for the one-step-ahead Lorenz-63 prediction error of the continuous-variable (Gaussian) reservoir under global encoding, for input dimension d in {1, 2, 3}, reporting the score on realizations the search never saw.

Task `optuna_lorenz63_qrc_gaussian`.

| | |
|---|---|
| Run ID | `0903163854` |
| Jobs | 6 |
| Raw results | `data/Results_run/optuna_lorenz63_qrc_gaussian_results_0903163854/` |
| Aggregated results | `data/Results_averaged/optuna_lorenz63_qrc_gaussian_results_0903163854_combined.pkl` |
| Read by | Figure 4 |

## Grid

| Axis | Values |
|---|---|
| `n` | 6 |
| `d` | 1, 2, 3 |
| `target` | all_1, x_1 |
| `encoding_mode` | dense |

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/hyperparameter_studies/optuna_lorenz_gaussian_dense/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_optuna.py --config ../experiments/hyperparameter_studies/optuna_lorenz_gaussian_dense/grid.yaml --run-id 0903163854
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
