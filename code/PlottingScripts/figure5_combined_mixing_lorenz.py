r"""Figure 5 — mixing capacity and prediction error against coupling strength, with
quantum resource measures on twin axes.

Reads all twelve parameter scans of `experiments/parameter_scans/`. Subpanel (a) uses the
mixing-capacity six (0905130003, 0905130004, 0905130005 for CV-QRC; 0905130004, 0905130005,
0905130006 for DV-QRC); subpanel (b) uses the Lorenz-63 six (0906114150, 0906114151,
0906114152; 0906114151, 0906114152, 0906114153).

**Both subpanels sit at the tuned operating point**, each gamma being the holdout
protocol's selection rounded to the nearest integer: (a) at n = 6, d = 2, CV 6, 5, 3 and
DV 7, 4, 3; (b) at n = 6, d = 3, target all_1, CV 7, 18, 13 and DV 10, 5, 3, for local /
clustered / global. The two rows of the figure therefore describe reservoirs at matching
operating points. Panel titles state each panel's own gamma.

Fixed slices:
  (a)  encoding_strength == 0.1. The mixing capacity is multiplied by 18 for DV-QRC and
       21 for CV-QRC, the observable count for each platform.
  (b)  encoding_strength == 1.0 and d == 3, and element [0] of the length-10
       first_moment_nrmse_test_average arrays. That axis is the prediction horizon
       (utils/prediction.py:49-59 fills entry i-1 from targets = np.roll(data, -i) for i in
       1..10), so [0] is the one-step-ahead test NRMSE, already averaged over components
       (utils/prediction.py:91).

The twin axis carries negativity for DV-QRC and squeezing for CV-QRC. Bands are +/- one
standard deviation across seeds.

Subpanel (b) reads gamma AFTER the d == 3 slice, so each panel title states the gamma of
the data that panel actually plots. The Lorenz parameter scans it reads carry d == 3 alone,
which makes that slice a no-op and the order immaterial; on a grid carrying more than one
d, reading gamma from the unfiltered frame would print the gamma of whichever d happened to
sit first in the result frame — a real optimised gamma, but one belonging to a different d
than the curve beneath it. figureS2_heatmap_lorenz.py reads gamma after its own filter for
the same reason.

The two encoding strengths this script slices on are the ones the manuscript's caption
for this figure states, so they are the manuscript's own numbers.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figure5_combined_mixing_lorenz.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    PLOTS_DIR,
    SOURCE_DATA_DIR,
    apply_paper_style,
    load_averaged,
    write_source_data,
)


def load_correlations() -> dict[tuple[str, str, str], dict[str, float]]:
    """The twelve correlation rows, keyed by (subpanel, model, encoding).

    Read from the deposit's SourceData/, never from the --source-data output directory:
    a caller that renders into a temporary directory rather than into the deposit's
    figures/ passes that directory as --source-data, and this file is an input, not an
    output.
    """
    path = SOURCE_DATA_DIR / "Figure5_correlations.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run "
            f"python PlottingScripts/extract_correlations.py from the deposit's code/ "
            f"directory first."
        )
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["subpanel"], row["model"], row["encoding"]): {
            "rho_sweep": float(row["rho_sweep"]),
            "rho_fixed": float(row["rho_fixed"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }
        for row in rows
    }


def panel_title(subpanel: str, row_label: str, col_label: str, gamma_str: str,
                correlations: dict) -> str:
    """Panel title with the two correlation coefficients on a second, lighter line.

    Hard-indexes `correlations`, same as the source-data block that looks up the same
    dict a few lines later: a missing row must fail loudly here too, not degrade to a
    silent single-line title while the source-data lookup still raises.
    """
    head = f"{row_label}: {col_label}{gamma_str}"
    stats = correlations[(subpanel, row_label, col_label)]
    return (
        head
        + "\n"
        + r"{\footnotesize\textcolor{gray}{$\rho_\mathrm{sweep}=%.2f$, $\rho_\mathrm{fixed}=%.2f$}}"
        % (stats["rho_sweep"], stats["rho_fixed"])
    )


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()

    # xcolor for the gray correlation line in panel_title(); scoped to this render
    # via rc_context so it is restored on exit, rather than left as a global
    # mutation that could pollute a figure rendered after this one in the same
    # process.
    with plt.rc_context({"text.latex.preamble": r"\usepackage{xcolor}"}):
        correlations = load_correlations()

        df_mc_oto_g = load_averaged("mixing", "gaussian", "one_to_one")
        df_mc_fill_g = load_averaged("mixing", "gaussian", "fill")
        df_mc_dense_g = load_averaged("mixing", "gaussian", "dense")
        df_mc_oto_s = load_averaged("mixing", "tfim", "one_to_one")
        df_mc_fill_s = load_averaged("mixing", "tfim", "fill")
        df_mc_dense_s = load_averaged("mixing", "tfim", "dense")

        df_l_oto_g = load_averaged("lorenz", "gaussian", "one_to_one")
        df_l_fill_g = load_averaged("lorenz", "gaussian", "fill")
        df_l_dense_g = load_averaged("lorenz", "gaussian", "dense")
        df_l_oto_s = load_averaged("lorenz", "tfim", "one_to_one")
        df_l_fill_s = load_averaged("lorenz", "tfim", "fill")
        df_l_dense_s = load_averaged("lorenz", "tfim", "dense")

        source_rows: list[dict] = []

        # ── Data structures ──
        models_data_mc = [
            ("DV-QRC", [
                ("Local", df_mc_oto_s),
                ("Clustered", df_mc_fill_s),
                ("Global", df_mc_dense_s)
            ]),
            ("CV-QRC", [
                ("Local", df_mc_oto_g),
                ("Clustered", df_mc_fill_g),
                ("Global", df_mc_dense_g)
            ])
        ]

        models_data_l = [
            ("DV-QRC", [
                ("Local", df_l_oto_s),
                ("Clustered", df_l_fill_s),
                ("Global", df_l_dense_s)
            ]),
            ("CV-QRC", [
                ("Local", df_l_oto_g),
                ("Clustered", df_l_fill_g),
                ("Global", df_l_dense_g)
            ])
        ]

        # ── Colors ──
        viridis_cmap = plt.get_cmap("viridis")
        prim_color = viridis_cmap(0.4)
        neg_color = viridis_cmap(0.15)
        sqz_color = viridis_cmap(0.65)

        # ── 5-row grid: rows 0-1 = panel (a), row 2 = spacer, rows 3-4 = panel (b) ──
        fig = plt.figure(figsize=(7.5, 7.6))
        gs = gridspec.GridSpec(
            5, 3, figure=fig,
            height_ratios=[1, 1, 0.2, 1, 1],  # the 0.2 is the spacer row between (a) and (b)
            hspace=0.55,   # spacing between rows within each panel — decrease to bring closer
            wspace=0.18,
        )

        # Create all 12 axes, skipping gridspec row 2 (the spacer)
        axes = np.empty((4, 3), dtype=object)
        for row in range(4):
            gs_row = row if row < 2 else row + 1  # skip gridspec row 2
            for col in range(3):
                axes[row, col] = fig.add_subplot(gs[gs_row, col])

        # ── Manual x-axis sharing: all 12 subplots share x ──
        for row in range(4):
            for col in range(3):
                if not (row == 0 and col == 0):
                    axes[row, col].sharex(axes[0, 0])

        # ── Manual y-axis sharing per row-pair ──
        # Rows 0-1 (mixing capacity) share primary y within each row
        for col in range(1, 3):
            axes[0, col].sharey(axes[0, 0])
            axes[1, col].sharey(axes[1, 0])
        # Rows 2-3 (Lorenz) share primary y within each row
        for col in range(1, 3):
            axes[2, col].sharey(axes[2, 0])
            axes[3, col].sharey(axes[3, 0])

        # =====================================================================
        # (a) Mixing Capacity – rows 0 & 1
        # =====================================================================
        target_encoding_mc = 0.1
        mix_mean_col = "first_moment_2_mean"
        mix_std_col = "first_moment_2_std"

        row_axes_data_mc = []

        for i, (row_label, cols) in enumerate(models_data_mc):
            if "DV" in row_label:
                sec_mean_col = "negativity_mean"
                sec_std_col = "negativity_std"
                sec_ylabel = r"Negativity $\mathcal{N}$"
                sec_color = neg_color
                mix_multiplier = 18.0
            else:
                sec_mean_col = "squeezing_mean"
                sec_std_col = "squeezing_std"
                sec_ylabel = r"Squeezing $\mathcal{S}$"
                sec_color = sqz_color
                mix_multiplier = 21.0

            row_y1_min, row_y1_max = float('inf'), float('-inf')
            row_y2_min, row_y2_max = float('inf'), float('-inf')
            row_ax2s = []

            for j, (col_label, df) in enumerate(cols):
                ax = axes[i, j]  # rows 0 and 1

                gamma_str = ""
                if "gamma" in df.columns:
                    ug = df["gamma"].dropna().unique()
                    if len(ug) > 0:
                        gamma_str = r" ($\gamma=" + str(ug[0]) + r"$)"
                ax.set_title(panel_title("a", row_label, col_label, gamma_str, correlations),
                             fontsize=9)

                mask = (df["encoding_strength"].notna()) & (abs(df["encoding_strength"] - target_encoding_mc) < 1e-5)
                sub_df = df[mask].copy()

                ax2 = ax.twinx()
                row_ax2s.append(ax2)

                if sub_df.empty:
                    ax.text(0.5, 0.5, f"No data for\n$\\epsilon={target_encoding_mc}$",
                            ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                else:
                    sub_df = sub_df.sort_values(by="coupling_strength")
                    x = sub_df["coupling_strength"].values

                    if mix_mean_col in sub_df.columns:
                        y1_mean = sub_df[mix_mean_col].values * mix_multiplier
                        y1_std = sub_df[mix_std_col].values * mix_multiplier if mix_std_col in sub_df.columns else np.zeros_like(y1_mean)
                        ax.plot(x, y1_mean, color=prim_color, label=r"$C_\mathrm{mix}$", marker="o", markersize=3)
                        ax.fill_between(x, y1_mean - y1_std, y1_mean + y1_std, color=prim_color, alpha=0.3)
                        row_y1_min = min(row_y1_min, (y1_mean - y1_std).min())
                        row_y1_max = max(row_y1_max, (y1_mean + y1_std).max())

                    ax.set_xscale("log")
                    ax.grid(True, which="both", ls="--", alpha=0.4)
                    ax.tick_params(axis='y', labelcolor=prim_color)

                    if sec_mean_col in sub_df.columns:
                        y2_mean = sub_df[sec_mean_col].values
                        y2_std = sub_df[sec_std_col].values if sec_std_col in sub_df.columns else np.zeros_like(y2_mean)
                        ax2.plot(x, y2_mean, color=sec_color, label=f"{sec_ylabel}", marker="s", markersize=3)
                        lower_bound_y2 = y2_mean - y2_std
                        ax2.fill_between(x, lower_bound_y2, y2_mean + y2_std, color=sec_color, alpha=0.3)
                        row_y2_min = min(row_y2_min, lower_bound_y2.min())
                        row_y2_max = max(row_y2_max, (y2_mean + y2_std).max())
                        secondary_name = "negativity" if "DV" in row_label else "squeezing"
                        gammas = sub_df["gamma"].to_numpy()
                        stats = correlations[("a", row_label, col_label)]
                        for index, x_value in enumerate(x):
                            source_rows.append({
                                "subpanel": "a",
                                "model": row_label,
                                "encoding": col_label,
                                "gamma": float(gammas[index]),
                                "coupling_strength": float(x_value),
                                "primary_quantity": "C_mix",
                                "primary_mean": float(y1_mean[index]),
                                "primary_sd": float(y1_std[index]),
                                "secondary_quantity": secondary_name,
                                "secondary_mean": float(y2_mean[index]),
                                "secondary_sd": float(y2_std[index]),
                                "rho_sweep": stats["rho_sweep"],
                                "rho_fixed": stats["rho_fixed"],
                                "rho_fixed_ci_low": stats["ci_low"],
                                "rho_fixed_ci_high": stats["ci_high"],
                            })

                    ax2.tick_params(axis='y', labelcolor=sec_color)

                    if j == 0:
                        lines_1, labels_1 = ax.get_legend_handles_labels()
                        lines_2, labels_2 = ax2.get_legend_handles_labels()
                        if lines_1 or lines_2:
                            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)

                if i == 1:
                    ax.set_xlabel(r"Coupling strength $J$")
                if j == 0:
                    ax.set_ylabel(r"$C_\mathrm{mix}$", color=prim_color)
                if j == 2:
                    ax2.set_ylabel(sec_ylabel, color=sec_color)
                else:
                    ax2.set_yticklabels([])

            row_axes_data_mc.append({
                'row_idx': i, 'ax2s': row_ax2s,
                'y1_min': row_y1_min, 'y1_max': row_y1_max,
                'y2_min': row_y2_min, 'y2_max': row_y2_max
            })

        for row_data in row_axes_data_mc:
            i = row_data['row_idx']
            r1_min, r1_max = row_data['y1_min'], row_data['y1_max']
            if r1_min != float('inf'):
                y1_pad = (r1_max - r1_min) * 0.05
                if y1_pad == 0: y1_pad = 0.1
                for ax in axes[i, :]:
                    ax.set_ylim(r1_min - y1_pad, r1_max + y1_pad)
            r2_min, r2_max = row_data['y2_min'], row_data['y2_max']
            if r2_min != float('inf'):
                y2_pad = (r2_max - r2_min) * 0.05
                if y2_pad == 0: y2_pad = 0.1
                for ax2 in row_data['ax2s']:
                    ax2.set_ylim(r2_min - y2_pad, r2_max + y2_pad)

        # =====================================================================
        # (b) Lorenz63 NRMSE – rows 2 & 3
        # =====================================================================
        target_encoding_l = 1.0
        nrmse_mean_col = "first_moment_nrmse_test_average_mean"
        nrmse_std_col = "first_moment_nrmse_test_average_std"

        row_axes_data_l = []

        for i, (row_label, cols) in enumerate(models_data_l):
            grid_row = i + 2  # axes rows 2 & 3 (mapped to gridspec rows 3 & 4)

            if "DV" in row_label:
                sec_mean_col = "negativity_mean"
                sec_std_col = "negativity_std"
                sec_ylabel = r"Negativity $\mathcal{N}$"
                sec_color = neg_color
            else:
                sec_mean_col = "squeezing_mean"
                sec_std_col = "squeezing_std"
                sec_ylabel = r"Squeezing $\mathcal{S}$"
                sec_color = sqz_color

            row_y1_min, row_y1_max = float('inf'), float('-inf')
            row_y2_min, row_y2_max = float('inf'), float('-inf')
            row_ax2s = []

            for j, (col_label, df_original) in enumerate(cols):
                ax = axes[grid_row, j]
                ax2 = ax.twinx()
                row_ax2s.append(ax2)

                if grid_row == 3:
                    ax.set_xlabel(r"Coupling strength $J$")
                if j == 0:
                    ax.set_ylabel("NRMSE", color=prim_color)
                if j == 2:
                    ax2.set_ylabel(sec_ylabel, color=sec_color)
                else:
                    ax2.set_yticklabels([])

                df = df_original.copy()
                gamma_str = ""
                has_data = False

                if not df.empty:
                    if 'd' in df.columns:
                        df = df[df['d'] == 3]
                    if "gamma" in df.columns:
                        ug = df["gamma"].dropna().unique()
                        if len(ug) > 0:
                            gamma_str = r" ($\gamma=" + str(ug[0]) + r"$)"
                    if nrmse_mean_col in df.columns:
                        df['nrmse_val'] = df[nrmse_mean_col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
                    if nrmse_std_col in df.columns:
                        df['nrmse_err'] = df[nrmse_std_col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
                    if "encoding_strength" in df.columns:
                        mask = (df["encoding_strength"].notna()) & (abs(df["encoding_strength"] - target_encoding_l) < 1e-5)
                        sub_df = df[mask].copy()
                        if not sub_df.empty and 'nrmse_val' in sub_df.columns and not sub_df['nrmse_val'].dropna().empty:
                            has_data = True

                ax.set_title(panel_title("b", row_label, col_label, gamma_str, correlations),
                             fontsize=9)

                if not has_data:
                    ax.text(0.5, 0.5, f"No data for\n$\\epsilon={target_encoding_l}$",
                            ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                else:
                    sub_df = sub_df.sort_values(by="coupling_strength")
                    x = sub_df["coupling_strength"].values

                    y1_mean = sub_df['nrmse_val'].values
                    y1_std = sub_df['nrmse_err'].values if 'nrmse_err' in sub_df.columns else np.zeros_like(y1_mean)
                    ax.plot(x, y1_mean, color=prim_color, label="NRMSE", marker="o", markersize=3)
                    lower_bound_y1 = np.clip(y1_mean - y1_std, a_min=1e-8, a_max=None)
                    ax.fill_between(x, lower_bound_y1, y1_mean + y1_std, color=prim_color, alpha=0.3)
                    row_y1_min = min(row_y1_min, y1_mean.min())
                    row_y1_max = max(row_y1_max, (y1_mean + y1_std).max())

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.grid(True, which="both", ls="--", alpha=0.4)
                    ax.tick_params(axis='y', labelcolor=prim_color)
                    ax.yaxis.set_minor_formatter(NullFormatter())

                    if sec_mean_col in sub_df.columns:
                        y2_mean = sub_df[sec_mean_col].values
                        y2_std = sub_df[sec_std_col].values if sec_std_col in sub_df.columns else np.zeros_like(y2_mean)
                        ax2.plot(x, y2_mean, color=sec_color, label=f"{sec_ylabel}", marker="s", markersize=3)
                        lower_bound_y2 = y2_mean - y2_std
                        ax2.fill_between(x, lower_bound_y2, y2_mean + y2_std, color=sec_color, alpha=0.3)
                        row_y2_min = min(row_y2_min, lower_bound_y2.min())
                        row_y2_max = max(row_y2_max, (y2_mean + y2_std).max())
                        secondary_name = "negativity" if "DV" in row_label else "squeezing"
                        gammas = sub_df["gamma"].to_numpy()
                        stats = correlations[("b", row_label, col_label)]
                        for index, x_value in enumerate(x):
                            source_rows.append({
                                "subpanel": "b",
                                "model": row_label,
                                "encoding": col_label,
                                "gamma": float(gammas[index]),
                                "coupling_strength": float(x_value),
                                "primary_quantity": "NRMSE_one_step",
                                "primary_mean": float(y1_mean[index]),
                                "primary_sd": float(y1_std[index]),
                                "secondary_quantity": secondary_name,
                                "secondary_mean": float(y2_mean[index]),
                                "secondary_sd": float(y2_std[index]),
                                "rho_sweep": stats["rho_sweep"],
                                "rho_fixed": stats["rho_fixed"],
                                "rho_fixed_ci_low": stats["ci_low"],
                                "rho_fixed_ci_high": stats["ci_high"],
                            })

                    ax2.tick_params(axis='y', labelcolor=sec_color)

                    if j == 0:
                        lines_1, labels_1 = ax.get_legend_handles_labels()
                        lines_2, labels_2 = ax2.get_legend_handles_labels()
                        if lines_1 or lines_2:
                            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)

            row_axes_data_l.append({
                'grid_row': grid_row, 'ax2s': row_ax2s,
                'y1_min': row_y1_min, 'y1_max': row_y1_max,
                'y2_min': row_y2_min, 'y2_max': row_y2_max
            })

        for row_data in row_axes_data_l:
            gr = row_data['grid_row']
            r1_min, r1_max = row_data['y1_min'], row_data['y1_max']
            if r1_min != float('inf'):
                pad1_min = r1_min * 0.7
                pad1_max = r1_max * 1.3
                for ax in axes[gr, :]:
                    if ax.get_yscale() == 'log':
                        ax.set_ylim(pad1_min, pad1_max)
            r2_min, r2_max = row_data['y2_min'], row_data['y2_max']
            if r2_min != float('inf'):
                y2_pad = (r2_max - r2_min) * 0.05
                if y2_pad == 0: y2_pad = 0.1
                for ax2 in row_data['ax2s']:
                    ax2.set_ylim(r2_min - y2_pad, r2_max + y2_pad)

        # ── Hide tick labels AFTER all plotting is done ──
        # Hide x-tick labels on non-bottom rows of each panel
        for row in [0, 2]:
            for col in range(3):
                axes[row, col].tick_params(axis='x', labelbottom=False)

        # Hide primary y-tick labels on columns 1 and 2 (only show on leftmost column)
        for row in range(4):
            for col in range(1, 3):
                axes[row, col].tick_params(axis='y', labelleft=False)

        # ── Panel labels (a) and (b) ──
        axes[0, 0].text(-0.25, 1.25, r"\textbf{(a)}", fontsize=12, transform=axes[0, 0].transAxes, va="top", ha="left")
        axes[2, 0].text(-0.25, 1.25, r"\textbf{(b)}", fontsize=12, transform=axes[2, 0].transAxes, va="top", ha="left")

        plt.savefig(out_dir / "combined_1d_mixing_lorenz.pdf", bbox_inches='tight')
        plt.close(fig)

    write_source_data("Figure5", source_rows, out_dir=source_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
