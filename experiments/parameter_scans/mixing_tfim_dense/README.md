# mixing_tfim_dense

Degree-2 mixing capacity of the discrete-variable (tilted transverse-field Ising) reservoir under global encoding of a 2-dimensional input, swept over encoding strength and coupling strength at a decay rate taken from the matching hyperparameter study.

Task `mixing_capacity_qrc_tilted_tfim`.

| | |
|---|---|
| Run ID | `0905130006` |
| Jobs | 4,940 |
| Raw results | `data/Results_run/mixing_capacity_qrc_tilted_tfim_results_0905130006/` |
| Aggregated results | `data/Results_averaged/mixing_capacity_qrc_tilted_tfim_results_0905130006_averaged.pkl` |
| Read by | Figure 3, Figure 5, Figure S3 |

## Grid

| Axis | Values |
|---|---|
| `n` | 6 |
| `d` | 2 |
| `encoding_mode` | dense |
| `dt` | 1.0 |
| `encoding_strength` | 13 values from 0.0001 to 1.0 |
| `coupling_strength` | 19 values from 1e-05 to 10.0 |
| `gamma` | 3.0 |
| `seed` | 20 values from 0 to 19 |

## Reproducing this run

```bash
cd code
python prepare_jobs.py --config ../experiments/parameter_scans/mixing_tfim_dense/grid.yaml
```

That writes an argument file and three submission scripts — a sequential local runner, an
HTCondor submit file and a Slurm array script. Then aggregate:

```bash
python average_runs_general.py --config ../experiments/parameter_scans/mixing_tfim_dense/grid.yaml --run-id 0905130006
```

The run ID is the one whose results this deposit carries, and naming it is required: with
no `--run-id` the averaging script looks for a `runs.yaml` ledger beside the grid, which
this deposit does not contain. A fresh run allocates its own ID.
