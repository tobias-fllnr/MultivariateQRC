r"""Figure 3 — mixing capacity across encoding strength and coupling strength.

Reads the six mixing-capacity parameter scans of `experiments/parameter_scans/`:
0905130003, 0905130004, 0905130005 (CV-QRC, one-to-one / fill / dense) and 0905130004,
0905130005, 0905130006 (DV-QRC, likewise).

**gamma is the tuned one**, so this figure and Figure 2 describe reservoirs at the same
operating point. Each panel uses the gamma of the best trial at n = 6, d = 2 in the
matching study of `experiments/hyperparameter_studies/`, rounded to the nearest integer:
CV 6, 5, 3 and DV 7, 4, 3 for local / clustered / global.

The mixing capacity is multiplied by 18 for DV-QRC and 21 for CV-QRC, the observable count
for each platform at n = 6. Each row of panels carries its own colour scale, computed
across that row's three panels after the multiplier is applied.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figure3_heatmap_mixing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    PLOTS_DIR,
    SOURCE_DATA_DIR,
    apply_paper_style,
    load_averaged,
    write_source_data,
)


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()

    df2_one_to_one_gaussian = load_averaged("mixing", "gaussian", "one_to_one")
    df2_fill_gaussian = load_averaged("mixing", "gaussian", "fill")
    df2_dense_gaussian = load_averaged("mixing", "gaussian", "dense")
    df2_one_to_one_spin = load_averaged("mixing", "tfim", "one_to_one")
    df2_fill_spin = load_averaged("mixing", "tfim", "fill")
    df2_dense_spin = load_averaged("mixing", "tfim", "dense")

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
        ])
    ]

    metrics = [
        ("first_moment_2_mean", "first_moment_2_std", "first_moment_2"),
        # ("negativity_mean", "negativity_std", "negativity"),
    ]

    threshold = None  # set to a float to filter weak grids

    for mean_col, std_col, metric_label in metrics:
        
        # 1. Determine global coordinate grids across ALL DataFrames to ensure perfectly aligned axes
        all_enc_vals = set()
        all_cs_vals = set()
        
        for row_label, cols in models_data:
            for col_label, df in cols:
                if "encoding_strength" in df.columns:
                    all_enc_vals.update(df["encoding_strength"].dropna().unique())
                if "coupling_strength" in df.columns:
                    all_cs_vals.update(df["coupling_strength"].dropna().unique())

        # Sort the coordinates to ensure the grid is strictly ordered
        enc_vals = sorted(list(all_enc_vals))
        cs_vals = sorted(list(all_cs_vals))

        if not enc_vals or not cs_vals:
            print(f"\nNo valid coordinates found for metric: {metric_label}. Skipping.")
            continue

        # --- Calculate clean log ticks based on the global grid ---
        x_ticks = [idx for idx, val in enumerate(cs_vals) if np.isclose(np.log10(val), np.round(np.log10(val)), atol=1e-5)]
        x_labels = [f"$10^{{{int(np.round(np.log10(val)))}}}$" for idx, val in enumerate(cs_vals) if np.isclose(np.log10(val), np.round(np.log10(val)), atol=1e-5)]

        y_ticks = [idx for idx, val in enumerate(enc_vals) if np.isclose(np.log10(val), np.round(np.log10(val)), atol=1e-5)]
        y_labels = [f"$10^{{{int(np.round(np.log10(val)))}}}$" for idx, val in enumerate(enc_vals) if np.isclose(np.log10(val), np.round(np.log10(val)), atol=1e-5)]
        # ----------------------------------------------------------

        # 2. Set up the 2x3 Figure
        fig, axes = plt.subplots(
            nrows=2, 
            ncols=3,
            figsize=(8, 5),  # Made slightly wider to accommodate the two colorbars comfortably
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True 
        )

        print(f"\n--- Processing Metric: {metric_label} ---")

        # 3. Iterate through the models_data structure row by row
        for i, (row_label, cols) in enumerate(models_data):
            
            # Determine the C_mix multiplier based on the row label
            if row_label == "DV-QRC":
                multiplier = 18
            elif row_label == "CV-QRC":
                multiplier = 21
            else:
                multiplier = 1
                
            # Calculate row-specific minimums and maximums for the color scale
            row_vmin = float('inf')
            row_vmax = float('-inf')
            
            for col_label, df in cols:
                if mean_col in df.columns:
                    # Apply the multiplier to the valid data to find the correct vmin/vmax
                    valid_data = df[mean_col].dropna() * multiplier
                    if not valid_data.empty:
                        row_vmin = min(row_vmin, valid_data.min())
                        row_vmax = max(row_vmax, valid_data.max())
                        
            # Fallback if a row is completely empty
            if row_vmin == float('inf'):
                row_vmin, row_vmax = 0, 1  
                
            row_im = None
            
            # Plot each column for the current row
            for j, (col_label, df) in enumerate(cols):
                ax = axes[i, j]
                
                # Create a copy of the dataframe to safely apply the multiplier for plotting
                plot_df = df.copy()
                if mean_col in plot_df.columns:
                    plot_df[mean_col] = plot_df[mean_col] * multiplier
                
                # Print maximum values for this specific subplot (now using the multiplied data)
                if mean_col in plot_df.columns and not plot_df[mean_col].dropna().empty:
                    max_idx = plot_df[mean_col].idxmax()
                    max_row = plot_df.loc[max_idx]
                    print(f"Max {mean_col} for {row_label} ({col_label}): {max_row[mean_col]:.4f}")
                
                # Extract the unique gamma value safely
                gamma_str = ""
                if "gamma" in plot_df.columns:
                    unique_gammas = plot_df["gamma"].dropna().unique()
                    if len(unique_gammas) > 0:
                        gamma_val = unique_gammas[0]
                        # Format gamma to append to the title
                        gamma_str = r" ($\gamma=" + str(gamma_val) + r"$)"
                plotted_gammas = plot_df["gamma"].dropna().unique()
                gamma_value = float(plotted_gammas[0])
                
                # Combine original title logic with the extracted gamma
                full_title = f"{row_label}: {col_label}{gamma_str}"
                
                # Filter and pivot
                sub = plot_df.dropna(subset=[mean_col])
                
                if sub.empty or (threshold is not None and sub[mean_col].max() <= threshold):
                    ax.text(0.5, 0.5, "no data\nor below threshold", ha="center", va="center",
                            transform=ax.transAxes, alpha=0.6)
                    ax.set_title(full_title)
                    continue

                pivot_df = sub.pivot_table(
                    index="encoding_strength", 
                    columns="coupling_strength", 
                    values=mean_col,
                    aggfunc="mean" 
                )
                
                # Reindex ensures identical shape and axis alignment
                pivot_df = pivot_df.reindex(index=enc_vals, columns=cs_vals)

                # Plot heatmap with row-specific vmin and vmax
                row_im = ax.imshow(
                    pivot_df.values, 
                    cmap="viridis", 
                    aspect="auto", 
                    origin="lower", 
                    vmin=row_vmin,  
                    vmax=row_vmax,  
                    interpolation="bilinear" 
                )
                values = pivot_df.values
                for y_index, encoding_strength in enumerate(enc_vals):
                    for x_index, coupling_strength in enumerate(cs_vals):
                        value = values[y_index, x_index]
                        if np.isnan(value):
                            continue
                        source_rows.append({
                            "model": row_label,
                            "encoding": col_label,
                            "gamma": gamma_value,
                            "encoding_strength": float(encoding_strength),
                            "coupling_strength": float(coupling_strength),
                            "C_mix": float(value),
                        })

                # Apply the filtered log ticks 
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)

                # Set the updated title
                ax.set_title(full_title, fontsize=10)

                # Only label axes on the outer edges
                if i == len(models_data) - 1:
                    ax.set_xlabel(r"Coupling strength $J$")
                if j == 0:
                    ax.set_ylabel(r"Encoding strength $\epsilon$")

            # 4. Attach a colorbar to the current row
            if row_im is not None:
                # ax=axes[i, :] tells matplotlib to attach the colorbar to the entire row
                fig.colorbar(row_im, ax=axes[i, :], label=r"$C_\mathrm{mix}$", shrink=0.8, aspect=20)

        # fig.suptitle(f"Comparison: {metric_label}", fontsize=14, fontweight="bold")
        plt.savefig(out_dir / f"mixing_capacity_comparison_heatmap_{metric_label}.pdf")
        plt.close(fig)

    write_source_data("Figure3", source_rows, out_dir=source_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
