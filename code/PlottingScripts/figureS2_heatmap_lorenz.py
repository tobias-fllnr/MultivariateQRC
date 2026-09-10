r"""Figure S2 — one-step-ahead Lorenz-63 NRMSE across encoding and coupling strength.

Reads the six Lorenz-63 parameter scans of `experiments/parameter_scans/`: 0906114150,
0906114151, 0906114152 (CV-QRC, one-to-one / fill / dense) and 0906114151, 0906114152,
0906114153 (DV-QRC, likewise). Each sits at the gamma the holdout hyperparameter protocol
selects at n = 6, d = 3, target all_1, rounded to the nearest integer -- CV 7, 18, 13 and
DV 10, 5, 3 for local / clustered / global -- which is the operating point Figure 4 reports
and the one Figure 3 and Figure 5(a) use.

Restricted to d == 3, the fully three-dimensional encoding, and to element [0] of the
length-10 first_moment_nrmse_test_average_mean array. That array is indexed by prediction
horizon — utils/prediction.py:49-59 fills entry i-1 from targets = np.roll(data, -i) for i
in 1..10 — so [0] is the one-step-ahead test NRMSE, itself already averaged over the
predicted components (utils/prediction.py:91).

Each row of panels carries its own logarithmic colour scale over the positive values of
that row. The panel titles state the gamma of the d == 3 slice, because this script's own
preprocessing loop reassigns cols[idx] to the filtered frame before the title is built.
The parameter scans it reads carry d == 3 alone, so that filter is a no-op rather than a
slice, and the titles state the single gamma each grid was run at.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figureS2_heatmap_lorenz.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

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

    df2_one_to_one_gaussian = load_averaged("lorenz", "gaussian", "one_to_one")
    df2_fill_gaussian = load_averaged("lorenz", "gaussian", "fill")
    df2_dense_gaussian = load_averaged("lorenz", "gaussian", "dense")
    df2_one_to_one_spin = load_averaged("lorenz", "tfim", "one_to_one")
    df2_fill_spin = load_averaged("lorenz", "tfim", "fill")
    df2_dense_spin = load_averaged("lorenz", "tfim", "dense")

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
        # (data_column, metric_label)
        ("first_moment_nrmse_test_average_mean", "NRMSE"),
    ]

    threshold = None  # set to a float to filter weak grids

    for mean_col, metric_label in metrics:
        
        # --- Pre-processing: Extract the [0] element from the array ---
        for row_label, cols in models_data:
            for idx, (col_label, df) in enumerate(cols):
                
                # 1. Filter for d == 3
                if 'd' in df.columns:
                    df = df[df['d'] == 3].copy()
                    # Overwrite the tuple in the list with the filtered DataFrame
                    cols[idx] = (col_label, df) 
                    
                # 2. Extract the first element
                if mean_col in df.columns:
                    # Safely extract the first element if it's an array/list, otherwise keep as is
                    df['extracted_val'] = df[mean_col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
        # --------------------------------------------------------------
        
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
            figsize=(7.3, 4.3),  
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True 
        )

        print(f"\n--- Processing Metric: {metric_label} ---")

        # 3. Iterate through the models_data structure row by row
        for i, (row_label, cols) in enumerate(models_data):
            
            # Calculate row-specific minimums and maximums for the color scale
            row_vmin = float('inf')
            row_vmax = float('-inf')
            
            for col_label, df in cols:
                if 'extracted_val' in df.columns:
                    valid_data = df['extracted_val'].dropna()
                    # Filter out zero or negative values since LogNorm requires positive floats
                    valid_data = valid_data[valid_data > 0]
                    if not valid_data.empty:
                        row_vmin = min(row_vmin, valid_data.min())
                        row_vmax = max(row_vmax, valid_data.max())
                        
            # Fallback if a row is completely empty or has no positive values
            if row_vmin == float('inf') or row_vmin <= 0:
                row_vmin, row_vmax = 1e-5, 1  
                
            row_im = None
            
            # Plot each column for the current row
            for j, (col_label, df) in enumerate(cols):
                ax = axes[i, j]
                
                # Print maximum values for this specific subplot
                if 'extracted_val' in df.columns and not df['extracted_val'].dropna().empty:
                    min_idx = df['extracted_val'].idxmin()
                    min_row = df.loc[min_idx]
                    print(f"Min {metric_label} for {row_label} ({col_label}): {min_row['extracted_val']:.4f}")
                
                # Extract the unique gamma value safely
                gamma_str = ""
                if "gamma" in df.columns:
                    unique_gammas = df["gamma"].dropna().unique()
                    if len(unique_gammas) > 0:
                        gamma_val = unique_gammas[0]
                        # Format gamma to append to the title
                        gamma_str = r" ($\gamma=" + str(gamma_val) + r"$)"
                plotted_gammas = df["gamma"].dropna().unique()
                gamma_value = float(plotted_gammas[0])
                
                # Combine original title logic with the extracted gamma
                full_title = f"{row_label}: {col_label}{gamma_str}"
                
                if 'extracted_val' not in df.columns:
                    sub = pd.DataFrame() # Empty frame to trigger the "no data" condition
                else:
                    sub = df.dropna(subset=['extracted_val'])
                
                if sub.empty or (threshold is not None and sub['extracted_val'].max() <= threshold):
                    ax.text(0.5, 0.5, "no data\nor below threshold", ha="center", va="center",
                            transform=ax.transAxes, alpha=0.6)
                    ax.set_title(full_title)
                    continue

                pivot_df = sub.pivot_table(
                    index="encoding_strength", 
                    columns="coupling_strength", 
                    values='extracted_val',
                    aggfunc="mean" 
                )
                
                # Reindex ensures identical shape and axis alignment
                pivot_df = pivot_df.reindex(index=enc_vals, columns=cs_vals)

                # Plot heatmap using LogNorm
                row_im = ax.imshow(
                    pivot_df.values, 
                    cmap="viridis_r", 
                    aspect="auto", 
                    origin="lower", 
                    norm=LogNorm(vmin=row_vmin, vmax=row_vmax), # <-- Updated for log scale
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
                            "NRMSE": float(value),
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
                # Updated the label to output 'NRMSE'
                fig.colorbar(row_im, ax=axes[i, :], label="NRMSE", shrink=0.8, aspect=20)

        # Saved file name updated for NRMSE
        plt.savefig(out_dir / "nrmse_comparison_heatmap.pdf")
        plt.close(fig)

    write_source_data("FigureS2", source_rows, out_dir=source_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
