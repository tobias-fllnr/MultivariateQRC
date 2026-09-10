# lorenz_gaussian_fill

One-step-ahead Lorenz-63 prediction error of the continuous-variable (Gaussian) reservoir under clustered encoding of the full three-component (d = 3) Lorenz state, swept over encoding strength and coupling strength at a decay rate taken from the matching hyperparameter study.

Task `lorenz63_qrc_gaussian`.

| | |
|---|---|
| Run ID | `0906114151` |
| Jobs | 4,940 |
| Raw results | `data/Results_run/lorenz63_qrc_gaussian_results_0906114151/` |
| Aggregated results | `data/Results_averaged/lorenz63_qrc_gaussian_results_0906114151_averaged.pkl` |
| Read by | Figure 5, Figure S2, Figure S3 |

## Grid

| Axis | Values |
|---|---|
| `n` | 6 |
| `d` | 3 |
| `encoding_mode` | fill |
| `dt` | 1.0 |
| `encoding_strength` | 13 values from 0.0001 to 1.0 |
| `coupling_strength` | 19 values from 1e-05 to 10.0 |
| `gamma` | 18.0 |
| `seed` | 20 values from 0 to 19 |

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/parameter_scans/lorenz_gaussian_fill/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_general.py --config ../experiments/parameter_scans/lorenz_gaussian_fill/grid.yaml --run-id 0906114151
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
