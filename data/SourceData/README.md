# Source data

The numbers behind each drawn point, one CSV per figure. Each row is one plotted point and
each value is what the figure draws: every slice, normalisation and sign convention is
already applied, so a panel can be redrawn from the CSV alone, with no access to the
pickles under `Results_averaged/` and no need to rerun anything. Floats carry full double
precision.

Most of the files are written by the figure script that draws the figure they belong to.
Two are not. `Figure5_correlations.csv` and `FigureS3.csv` are derived by
`code/PlottingScripts/extract_correlations.py`, the only script in the deposit that reads
the raw per-seed results under `Results_run/`, and the figure scripts read them back rather
than recomputing them. Figure 5 only reads `Figure5_correlations.csv`, printing its
coefficients in the panel titles. `figureS3_resource_scatter.py`, on the other hand, reads
`FigureS3.csv`, plots it, and then writes it out again as its own source data -- so two
scripts write that file, but the numbers in it always come from
`extract_correlations.py`.

`model` takes three values, not two: `DV-QRC` for the discrete-variable reservoir,
`CV-QRC` for the continuous-variable one, and `ESN` for the classical echo state network
that both are compared against -- a reservoir of matched readout size with no quantum
degrees of freedom, which is the baseline the paper's claims are measured against. All
three appear in `Figure2.csv`, `Figure4.csv` and `FigureS1.csv`. The other six files carry
only the two quantum models, because the figures they belong to sweep or plot quantities
with no classical counterpart: the Hamiltonian's encoding and coupling strengths, the
negativity and the squeezing, and the size of the measurement ensemble. `encoding` is
`Local`, `Clustered` or `Global` for all three models, matching the panel titles rather
than the internal parameter names. The figures print no panel letters
except Figure 5's (a) and (b), so there is no panel column: a `(model, encoding)` pair
identifies a panel, and Figure 5 and Figure S3 add `subpanel`.

**Every `_sd` column is a standard deviation across 20 realizations, never a standard error
of the mean.** For the figures built on the parameter scans (3, 5, S2) it is the population
standard deviation over the grid's 20 seeds. For the figures built on the hyperparameter
studies (2, 4, 6, S1) it is the sample standard deviation over the 20 held-out
realizations that the search never saw.

| File | Columns | Slice and transform |
|---|---|---|
| `Figure2.csv` | `model, encoding, D, n, C_mix_mean, C_mix_sd` | `n <= 6`, `D <= 5`. Capacity multiplied by the readout dimension: `n(n+1)/2` for CV-QRC, `3n` for DV-QRC. |
| `Figure3.csv` | `model, encoding, gamma, encoding_strength, coupling_strength, C_mix` | Capacity multiplied by the readout dimension at `n = 6`: 21 for CV-QRC, 18 for DV-QRC. One row per grid cell that holds data; cells the figure leaves blank are absent. |
| `Figure4.csv` | `model, encoding, encoded_dimensions, NRMSE_mean, NRMSE_sd` | Target `x_1`. `encoded_dimensions` is 1, 2 or 3, drawn as x, xy, xyz. |
| `Figure5.csv` | `subpanel, model, encoding, gamma, coupling_strength, primary_quantity, primary_mean, primary_sd, secondary_quantity, secondary_mean, secondary_sd, rho_sweep, rho_fixed, rho_fixed_ci_low, rho_fixed_ci_high` | (a) `encoding_strength == 0.1`, capacity multiplied by the readout dimension as in Figure 3. (b) `encoding_strength == 1.0`, `d == 3`, one-step-ahead NRMSE. The secondary quantity is negativity for DV-QRC and squeezing for CV-QRC. The four `rho_*` columns repeat that panel's row of `Figure5_correlations.csv`, which is what the panel title prints. |
| `Figure5_correlations.csv` | `subpanel, model, encoding, resource_quantity, performance_quantity, gamma, rho_sweep, rho_fixed, ci_low, ci_high, k_couplings, n_seeds, peak_offset_decades` | One row per panel of Figure 5, twelve in all. `rho_sweep` is the Spearman correlation between the seed-mean resource and the seed-mean performance along the coupling sweep. `rho_fixed` is the Fisher-pooled Spearman correlation across seeds at fixed coupling, pooled over the `k_couplings` couplings at which the resource is present, with `ci_low`/`ci_high` its 95% interval. Performance is signed so that larger is better: the scaled mixing capacity in (a), minus the one-step-ahead NRMSE in (b). `peak_offset_decades` is the distance in decades of coupling strength between the resource peak and the performance peak. |
| `Figure6.csv` | `model, encoding, n_shots_per_setting, n_settings, readout_dimension, mixing_capacity_mean, mixing_capacity_sd, n_realizations` | `n == 4`, `D == 2`. Capacity multiplied by `readout_dimension`, which is `n(n+1)/2` for CV-QRC and `3n` for DV-QRC. `n_shots_per_setting` is `inf` on the exact-limit row; `n_settings` is 1 for the CV homodyne readout and 3 for the DV Pauli readout, so the total number of measurements per time step is the product of the two. |
| `FigureS1.csv` | `model, encoding, D, n, C_mix_mean, C_mix_sd` | `n <= 6`, `D <= 5`. The same data as `Figure2.csv` under the other normalisation: capacity divided by `D(D-1)/2`, with no readout-dimension factor. |
| `FigureS2.csv` | `model, encoding, gamma, encoding_strength, coupling_strength, NRMSE` | `d == 3`, one-step-ahead NRMSE. One row per grid cell that holds data. |
| `FigureS3.csv` | `subpanel, model, encoding, coupling_strength, seed, resource, performance` | One row per (coupling strength, seed) pair for each of the twelve curves of Figure 5, so 12 x 19 x 20 rows. These are the per-seed points `rho_fixed` is computed from; `performance` carries the same sign convention and scaling as in `Figure5_correlations.csv`. |

"One-step-ahead NRMSE" is element 0 of the length-10 prediction-horizon axis the result
files carry. Entry `i-1` of that axis is the error at a horizon of `i` time steps, and each
entry is already averaged over the predicted components.
