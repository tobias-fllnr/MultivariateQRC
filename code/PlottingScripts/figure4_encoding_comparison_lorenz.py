r"""Figure 4 — Lorenz-63 x-component NRMSE against the number of encoded dimensions.

Reads the nine Lorenz-63 hyperparameter studies of experiments/hyperparameter_studies/:
0903163856, 0903163855, 0903163854 (CV-QRC, local / clustered / global), 0903163857,
0903163856, 0903163855 (DV-QRC, likewise) and 0903163855, 0903163854, 0903163853 (ESN,
likewise).

Three properties of the figure are worth stating:

  - **The plotted number is the holdout, not the search optimum.** Every point is
    `holdout_score_mean` over the twenty `EVAL_SEEDS` realizations no trial saw, scored on
    a test block the search never read either, and banded by `holdout_score_sd` — the
    sample SD across those realizations, ddof=1. The best of 100 trials on the ten tuning
    seeds is inflated by that selection, so it is not what is reported; those keys are
    deliberately absent from the payloads, so a script still reading them fails loudly
    rather than plotting the wrong number. `assert_full_holdout` refuses any point that
    does not rest on all twenty realizations.
  - **Three panels, one per model.** The classical ESN baseline joins the DV and CV
    reservoirs, at `3n = 18` nodes — matched exactly to the DV local observable block, and
    within three features of the CV `q_only` covariance readout at n = 6.
  - **The protocol is teacher-forced one-step-ahead prediction**, which is what the
    caption and the Methods section state.

**`target == "x_1"` is the whole figure.** The grids sweep two targets: `x_1`, the
one-step-ahead NRMSE of the x component alone, and `all_1`, the mean over the three
components. Only `x_1` is plotted here — `all_1` exists so its optimised decay rate can set
the operating point of the Lorenz parameter scans that Figure 5(b) and Supplementary
Figure S2 read. Both live in the same pickle, so the filter is not a convenience: without
it every point would average two different metrics, and `assert_single_target` is what makes
dropping the filter an error instead of a wrong plot.

The x axis is d, the number of encoded Lorenz dimensions, drawn with tick labels x, xy,
xyz. The y axis is logarithmic and shared across the three panels.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figure4_encoding_comparison_lorenz.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.cm as cm
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

# The one-step-ahead x-component target. `run_optuna_job.get_nrmse_element` splits this on
# '_' and reads horizon index int(suffix) - 1 of the length-10 arrays
# `prediction_multi_step` fills, where entry i-1 holds the i-step-ahead error -- so the
# suffix 1 is one step ahead, roughly 1/12 of a Lyapunov time. Named here rather than
# written inline three times so the figure has one place that says which target it plots.
TARGET = "x_1"

# The mean and SD columns of the holdout protocol. Not best_score_mean / best_score_std:
# those keys named a best-of-100 search value and are absent from every payload read here.
MEAN_COLUMN = "holdout_score_mean"
SD_COLUMN = "holdout_score_sd"


def assert_single_target(frame, label: str) -> None:
    """Refuses a frame that still carries more than one target after filtering.

    The pickles hold `x_1` and `all_1` side by side, and the two are different metrics --
    the x component against its own standard deviation, versus the mean of three
    per-component NRMSEs. Averaging them would produce a plausible-looking wrong number
    rather than an error, so the filter is checked rather than trusted.
    """
    targets = set(frame["target"].dropna().unique())
    if targets != {TARGET}:
        raise ValueError(
            f"{label}: expected target {TARGET!r} alone after filtering, got "
            f"{sorted(targets)} -- the {TARGET!r} filter is missing or wrong"
        )


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()

    df2_one_to_one_gaussian = load_averaged("optuna_lorenz", "gaussian", "one_to_one")
    df2_fill_gaussian = load_averaged("optuna_lorenz", "gaussian", "fill")
    df2_dense_gaussian = load_averaged("optuna_lorenz", "gaussian", "dense")
    df2_one_to_one_spin = load_averaged("optuna_lorenz", "tfim", "one_to_one")
    df2_fill_spin = load_averaged("optuna_lorenz", "tfim", "fill")
    df2_dense_spin = load_averaged("optuna_lorenz", "tfim", "dense")
    df2_one_to_one_esn = load_averaged("optuna_lorenz", "esn", "one_to_one")
    df2_fill_esn = load_averaged("optuna_lorenz", "esn", "fill")
    df2_dense_esn = load_averaged("optuna_lorenz", "esn", "dense")

    source_rows: list[dict] = []

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

    # Every frame must rest on all twenty held-out realizations before anything is drawn,
    # so a short point fails the figure rather than appearing in one panel of it.
    for model_name, experiments in models_data:
        for exp_name, exp_df in experiments:
            assert_full_holdout(exp_df, f"{model_name} {exp_name}")

    # 1. Find all unique experiment names (encoding methods) to ensure consistent coloring
    all_experiments = []
    for model_name, experiments in models_data:
        for exp_name, exp_df in experiments:
            if exp_name not in all_experiments:
                all_experiments.append(exp_name)

    # 2. Generate consistent colors from the viridis colormap based on encoding method
    colors = cm.viridis(np.linspace(0, 1, len(all_experiments)))
    exp_color_map = {exp: color for exp, color in zip(all_experiments, colors)}

    # 3. One column per model: DV-QRC, CV-QRC, ESN. The figure keeps its single-column
    #    width, so the panels narrow rather than the fonts shrinking. Each panel carries
    #    only three x positions, so it takes the narrowing without crowding.
    fig, axes = plt.subplots(1, 3, figsize=(5, 2.9), sharey=True)
    legend_handles = {}

    # 4. Loop through the Models (side-by-side subplots)
    for ax_idx, (model_name, experiments) in enumerate(models_data):
        ax = axes[ax_idx]

        anything_plotted_in_ax = False

        # Loop through the experiments
        for exp_name, exp_df in experiments:

            # Strictly filter for the 'x_1' target and drop NaNs
            plot_df = exp_df[exp_df['target'] == TARGET].dropna(
                subset=[MEAN_COLUMN, SD_COLUMN]
            ).copy()

            if plot_df.empty:
                continue

            assert_single_target(plot_df, f"{model_name} {exp_name}")

            # Ensure data is sorted by the x-axis dimension
            target_data = plot_df.sort_values(by='d')

            x = target_data['d']
            y = target_data[MEAN_COLUMN]
            std = target_data[SD_COLUMN]

            color = exp_color_map[exp_name]

            # Plot the mean line and fill +/- one holdout SD for this encoding method
            line = ax.plot(x, y, marker='o', label=exp_name, color=color)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.2)
            for d_value, mean_value, sd_value in zip(x, y, std):
                source_rows.append({
                    "model": model_name,
                    "encoding": exp_name,
                    "encoded_dimensions": int(d_value),
                    "NRMSE_mean": float(mean_value),
                    "NRMSE_sd": float(sd_value),
                })

            # Save the line handle for the master legend
            if exp_name not in legend_handles:
                legend_handles[exp_name] = line[0]

            anything_plotted_in_ax = True

        # 5. Format the specific subplot
        if anything_plotted_in_ax:
            ax.set_title(f"{model_name}", pad=10)
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Hardcode the x-axis ticks and assign the string labels
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["x", "xy", "xyz"])

            # All three subplots need the x-label in a side-by-side layout
            ax.set_xlabel("Encoded dimensions")

            # Only put the y-label on the left subplot (index 0)
            if ax_idx == 0:
                ax.set_ylabel("NRMSE")
        else:
            ax.set_title(f"{model_name}\n(No '{TARGET}' data)")

    # 6. Add the master legend and adjust layout
    if legend_handles:
        # Retain the exact order of the experiments as they were discovered
        handles = [legend_handles[exp] for exp in all_experiments if exp in legend_handles]
        labels = [exp for exp in all_experiments if exp in legend_handles]

        # Below the panels, not inside one of them. With a third model the panels are
        # narrow enough that a legend in axes[0] would sit on top of its own lines.
        fig.legend(
            handles, labels, loc='lower center', ncol=len(labels),
            bbox_to_anchor=(0.5, -0.02), frameon=False,
        )

        plt.tight_layout(rect=(0, 0.06, 1, 1))
        plt.savefig(out_dir / "lorenz_x1_encoding_comparison.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
        print(f"No valid '{TARGET}' data found to plot across any models.")

    write_source_data("Figure4", source_rows, out_dir=source_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
