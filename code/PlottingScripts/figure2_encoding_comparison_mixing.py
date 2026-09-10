r"""Figures 2 and S1 — mixing capacity against system size, one line per encoded dimension.

Reads the nine hyperparameter-optimised mixing-capacity runs pinned in `_common.RUN_IDS`:
the six quantum studies of `experiments/hyperparameter_studies/` (0903012718–0903012721)
and the three ESN studies (0903151601–0903151603).

Two things to know before reading the figure:

  - **The plotted number is the holdout, not the search optimum.** Every point is
    `holdout_score_mean` over the twenty `EVAL_SEEDS` realizations no trial saw, banded by
    `holdout_score_sd` — the sample SD across those realizations, ddof=1. The best of 100
    trials on the ten tuning seeds is inflated by that selection, so it is not what is
    reported; the payloads deliberately carry no such key. Every plotted point is asserted
    to rest on all twenty realizations; see `assert_full_holdout`.
  - **Three rows, one per model.** The classical ESN baseline joins the DV and CV
    reservoirs as a third row, so the quantum result is read against a classical reservoir
    on the identical grid, metric and protocol rather than against nothing.

This script loops over all four combinations of (is_n_normalized, is_d_normalized) and
saves each. Only two are cited, and only those two are written here:

  Figure 2   nNorm False, dNorm False -> the capacity is multiplied by n(n+1)/2 for CV-QRC
             and 3n for DV-QRC, the observable count as a function of system size, and is
             not divided by anything.
  Figure S1  nNorm True, dNorm True   -> no n factor, and divided by D(D-1)/2. All six
             frames start at d == 2, so that divisor is never zero.

The loop is kept whole because the colour normalisation over d is computed once before it,
across all nine frames.

Restricted to n <= 6 and d <= 5 by the MAX_N and MAX_D that main() defines below — which
is also the extent of the grids themselves, so nothing is optimised and then clipped
away.

The output filenames say `best_score` for historical reasons: the manuscript cites
`combined_best_score_nNorm_*_dNorm_*.pdf`. The name is the file's, not the metric's.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figure2_encoding_comparison_mixing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    PLOTS_DIR,
    SOURCE_DATA_DIR,
    apply_paper_style,
    assert_full_holdout,
    load_averaged,
    write_source_data,
)

# (is_n_normalized, is_d_normalized) -> the SourceData sheet name for that combination.
# The two absent combinations are the ones no \includegraphics cites.
CITED = {(False, False): "Figure2", (True, True): "FigureS1"}


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()

    df2_one_to_one_gaussian = load_averaged("optuna_mixing", "gaussian", "one_to_one")
    df2_fill_gaussian = load_averaged("optuna_mixing", "gaussian", "fill")
    df2_dense_gaussian = load_averaged("optuna_mixing", "gaussian", "dense")
    df2_one_to_one_spin = load_averaged("optuna_mixing", "tfim", "one_to_one")
    df2_fill_spin = load_averaged("optuna_mixing", "tfim", "fill")
    df2_dense_spin = load_averaged("optuna_mixing", "tfim", "dense")
    df2_one_to_one_esn = load_averaged("optuna_mixing", "esn", "one_to_one")
    df2_fill_esn = load_averaged("optuna_mixing", "esn", "fill")
    df2_dense_esn = load_averaged("optuna_mixing", "esn", "dense")

    # 1. Group the dataframes by model (Row) and experiment type (Column). The two
    #    quantum rows come first and the classical baseline is appended underneath.
    models_data = [
        ("DV-QRC", [
            ("Local", df2_one_to_one_spin),
            ("Clustered", df2_fill_spin),
            ("Global", df2_dense_spin)
        ]),
        ("CV-QRC", [
            ("Local", df2_one_to_one_gaussian),
            ("Clustered", df2_fill_gaussian),
            ("Global", df2_dense_gaussian)
        ]),
        ("ESN", [
            ("Local", df2_one_to_one_esn),
            ("Clustered", df2_fill_esn),
            ("Global", df2_dense_esn)
        ])
    ]

    for model_name, experiments in models_data:
        for exp_name, exp_df in experiments:
            assert_full_holdout(exp_df, f"{model_name} {exp_name}")

    # metrics and threshold setup
    # (mean column, band column, filename label). The label stays "best_score" because it
    # names the file the manuscript cites, not the metric -- see the module docstring. The
    # columns are the holdout's: best_score_mean / best_score_std do not exist in these
    # payloads, by design, so a stale read fails loudly here.
    metrics = [
        ("holdout_score_mean", "holdout_score_sd", "best_score"),
        # ("first_and_second_moment_2_mean", "first_and_second_moment_2_std", "first_and_second_moment_2"),
        # ("first_moment_3_mean", "first_moment_3_std", "first_moment_3"),
        # ("first_and_second_moment_3_mean", "first_and_second_moment_3_std", "first_and_second_moment_3"),
    ]

    # Set your limits here
    MAX_N = 6
    MAX_D = 5

    num_figs = 0

    for mean_col, std_col, metric_label in metrics:
        
        # Find global min and max 'd' across ALL models and experiments
        all_d_vals = []
        for model_name, experiments in models_data:
            for _, exp_df in experiments:
                df_filtered = exp_df.dropna(subset=[mean_col, std_col])
                df_filtered = df_filtered[(df_filtered["n"] <= MAX_N) & (df_filtered["d"] <= MAX_D)]
                all_d_vals.extend(df_filtered["d"].unique())
            
        if not all_d_vals:
            continue  # Skip if no data at all for this metric
            
        vmin_d, vmax_d = min(all_d_vals), max(all_d_vals)
        
        # Initialize the viridis colormap and normalizer
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=vmin_d, vmax=vmax_d) if vmin_d != vmax_d else mcolors.Normalize(vmin=vmin_d-1, vmax=vmax_d+1)
        
        # Loop through all 4 combinations of normalization
        for is_n_normalized in [True, False]:
            for is_d_normalized in [True, False]:
                
                source_rows: list[dict] = []
                
                # Create a 3 row, 3 column figure: DV-QRC, CV-QRC, ESN.
                # Removed sharex=True so models have independent x-ranges.
                fig, axes = plt.subplots(3, 3, figsize=(8, 6.75), sharey='row')
                anything_plotted_in_fig = False
                
                # Dictionary to collect unique legend handles
                legend_handles = {}
                
                # Loop through rows (Models: Spin QRC, Gaussian QRC)
                for row_idx, (model_name, experiments) in enumerate(models_data):
                    
                    # Loop through columns (Experiments: Local, Clustered, Global)
                    for col_idx, (exp_name, exp_df) in enumerate(experiments):
                        ax = axes[row_idx, col_idx]
                        
                        # Drop rows where either mean or std is NaN
                        plot_df = exp_df.dropna(subset=[mean_col, std_col]).copy()
                        
                        # Filter the dataframe to limit n and d
                        plot_df = plot_df[(plot_df["n"] <= MAX_N) & (plot_df["d"] <= MAX_D)]
                        
                        d_vals = sorted(plot_df["d"].unique())
                        
                        if len(d_vals) == 0:
                            ax.set_title(f"{model_name}: {exp_name}\n(No data)")
                            continue
                        
                        anything_plotted_in_ax = False

                        for d_val in d_vals:
                            d_df = plot_df[plot_df["d"] == d_val].sort_values(by="n")
                            
                            if d_df.empty:
                                continue
                                
                            x = d_df["n"]
                            y_mean = d_df[mean_col].copy()
                            y_std = d_df[std_col].copy()
                            
                            factor = 1.0
                            if not is_n_normalized:
                                # The readout dimension, per model: CV reads the upper
                                # triangle of the n x n q-covariance, DV reads 3n local
                                # Pauli expectations, and the ESN has 3n nodes -- matched
                                # to that DV readout, which is what makes the baseline
                                # comparable rather than merely present.
                                if "CV-QRC" in model_name:
                                    factor = factor * (x * (x + 1) / 2)
                                elif "DV-QRC" in model_name or model_name == "ESN":
                                    factor = factor * (3 * x)

                            if is_d_normalized:
                                factor = factor / (d_val * (d_val - 1) / 2)
                            
                            y_mean = y_mean * factor
                            y_std = y_std * factor
                            
                            line_color = cmap(norm(d_val))
                            
                            # Plot mean line and standard deviation
                            line = ax.plot(x, y_mean, marker='o', label=f"D={d_val:g}", color=line_color)
                            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=line_color)
                            for n_value, mean_value, sd_value in zip(x, y_mean, y_std):
                                source_rows.append({
                                    "model": model_name,
                                    "encoding": exp_name,
                                    "D": int(d_val),
                                    "n": int(n_value),
                                    "C_mix_mean": float(mean_value),
                                    "C_mix_sd": float(sd_value),
                                })
                            
                            # Save the line handle for the global legend
                            if d_val not in legend_handles:
                                legend_handles[d_val] = line[0]
                            
                            anything_plotted_in_ax = True
                            anything_plotted_in_fig = True

                        # Formatting for the specific subplot
                        if anything_plotted_in_ax:
                            # Write the Model Name and Encoding Method on every subplot
                            ax.set_title(f"{model_name}: {exp_name}", pad=10)
                            
                            # Set x-labels for all plots since they are now independent
                            ax.set_xlabel("n")
                            
                            # Force integer ticks on the x-axis
                            # ax.set_xticks(range(2, MAX_N + 1))
                            ax.grid(True, linestyle='--', alpha=0.6)
                            
                            # Set y-labels (Metric ONLY) on the far left column
                            if col_idx == 0:
                                ax.set_ylabel(r"$C_\mathrm{mix}$")

                # Handle the overall figure
                if anything_plotted_in_fig:
                    # Add a single master legend to the center-right of the figure
                    sorted_d_keys = sorted(legend_handles.keys())
                    handles = [legend_handles[d] for d in sorted_d_keys]
                    labels = [f"D={d:g}" for d in sorted_d_keys]
                    
                    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.00, 0.5))
                    
                    # Use rect to compress the subplots slightly to the left, leaving 10% space for the legend
                    plt.tight_layout(rect=[0, 0, 0.9, 1])
                    sheet = CITED.get((is_n_normalized, is_d_normalized))
                    if sheet is None:
                        plt.close(fig)
                        continue

                    plt.savefig(
                        out_dir
                        / f"combined_{metric_label}_nNorm_{is_n_normalized}"
                        f"_dNorm_{is_d_normalized}.pdf",
                        bbox_inches='tight',
                    )
                    plt.close(fig)
                    write_source_data(sheet, source_rows, out_dir=source_data_dir)

                    num_figs += 1
                else:
                    plt.close(fig)

    print(f"\nGenerated {num_figs} combined figure(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
