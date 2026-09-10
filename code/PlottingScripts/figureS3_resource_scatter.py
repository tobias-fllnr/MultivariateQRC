r"""Supplementary Figure S3 -- quantum resource against performance, per realization.

Twelve panels in Figure 5's arrangement: rows 0-1 are subpanel (a)'s DV and CV mixing
capacity, rows 2-3 subpanel (b)'s Lorenz-63 prediction, one column per encoding. Each
point is one of the twenty random realizations at one of the nineteen coupling strengths,
coloured by coupling.

The figure shows in one image what the two coefficients in Figure 5's titles say
numerically. The colours trace an arc as the coupling rises, which is the large
sweep-level correlation; within a single colour, where the coupling is fixed, the cloud
has no orientation, which is the small fixed-coupling one.

Performance is signed so that larger is better, exactly as in Figure 5: the mixing
capacity scaled by the readout dimension in (a), and minus the one-step-ahead NRMSE in
(b).

Input is SourceData/FigureS3.csv, written by extract_correlations.py. This script never
reads Results_run/.

Usage, from the deposit's code/ directory:
    python PlottingScripts/figureS3_resource_scatter.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    PLOTS_DIR,
    SOURCE_DATA_DIR,
    apply_paper_style,
    write_source_data,
)

ROWS = (("a", "DV-QRC"), ("a", "CV-QRC"), ("b", "DV-QRC"), ("b", "CV-QRC"))
COLUMNS = ("Local", "Clustered", "Global")

RESOURCE_LABEL = {
    "DV-QRC": r"Negativity $\mathcal{N}$",
    "CV-QRC": r"Squeezing $\mathcal{S}$",
}
PERFORMANCE_LABEL = {"a": r"$C_\mathrm{mix}$", "b": r"$-$NRMSE"}


def load_points() -> list[dict]:
    """Reads SourceData/FigureS3.csv, always from the deposit's own SourceData/.

    This CSV is an input to this script, not an output: --source-data below names where
    the *re-emitted* copy goes -- a caller may point that at a temporary directory rather
    than at the deposit -- but the data plotted always comes from the deposit's own
    SourceData/, exactly as figure5_combined_mixing_lorenz.load_correlations() does for
    Figure5_correlations.csv.
    """
    path = SOURCE_DATA_DIR / "FigureS3.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run "
            f"python PlottingScripts/extract_correlations.py from the deposit's code/ "
            f"directory first."
        )
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()
    points = load_points()

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in points:
        grouped.setdefault(
            (row["subpanel"], row["model"], row["encoding"]), []
        ).append(row)

    couplings = np.array([float(row["coupling_strength"]) for row in points])
    norm = LogNorm(vmin=couplings.min(), vmax=couplings.max())
    cmap = plt.get_cmap("viridis")

    # Two separate grids, one per subpanel, rather than one grid with a spacer row.
    # A spacer row cannot be tuned independently of the gap between the rows *within* a
    # panel: matplotlib's hspace is a single fraction of the row height applied to every
    # gap, so the panel separation is always twice hspace plus the spacer, and shrinking
    # the figure to close it also closes the within-panel gaps until the x-axis labels
    # collide with the titles beneath them. Giving each subpanel its own grid separates
    # the two: hspace sets the within-panel gap, and the top/bottom edges below set the
    # gap between the panels. Keeping the figure short matters because a tall figure has
    # to be set narrower in the SI to leave its caption room on the page.
    fig = plt.figure(figsize=(7.5, 7.4))
    panel_grids = (
        # The gap between the panels, 0.16 of the figure height, is about 1.6 times the
        # gap between the two rows inside a panel, so the grouping reads correctly
        # without the panels drifting apart.
        gridspec.GridSpec(2, 3, figure=fig, top=0.955, bottom=0.600,
                          hspace=0.75, wspace=0.30),
        gridspec.GridSpec(2, 3, figure=fig, top=0.440, bottom=0.085,
                          hspace=0.75, wspace=0.30),
    )
    axes = np.empty((4, 3), dtype=object)
    for row_index in range(4):
        panel_grid = panel_grids[row_index // 2]
        for column_index in range(3):
            axes[row_index, column_index] = fig.add_subplot(
                panel_grid[row_index % 2, column_index]
            )

    scatter = None
    source_rows = []
    for row_index, (subpanel, model) in enumerate(ROWS):
        for column_index, encoding in enumerate(COLUMNS):
            ax = axes[row_index, column_index]
            rows = grouped[(subpanel, model, encoding)]
            resource = np.array([float(r["resource"]) for r in rows])
            performance = np.array([float(r["performance"]) for r in rows])
            coupling = np.array([float(r["coupling_strength"]) for r in rows])
            scatter = ax.scatter(
                resource, performance, c=coupling, cmap=cmap, norm=norm,
                s=4, linewidths=0, alpha=0.5,
            )
            ax.set_title(f"{model}: {encoding}", fontsize=9)
            ax.grid(True, ls="--", alpha=0.4)
            ax.tick_params(labelsize=7)
            if model == "DV-QRC":
                # Negativity spans ~1e-10 to ~0.24 across the sweep. A plain log or
                # symlog with a small linthresh would spread the ~1e-7-1e-8 noise
                # floor at weak coupling (a genuinely separable state, not signal)
                # across several decades, manufacturing structure that is not
                # there. linthresh=1e-3 sits above that floor, so every
                # noise-floor point collapses into the linear region near zero
                # instead, while the real structure between moderate and strong
                # coupling -- compressed to a sliver on a linear axis -- becomes
                # visible. Squeezing (CV-QRC) has no such noise floor and stays
                # linear.
                ax.set_xscale("symlog", linthresh=1e-3)
            if column_index == 0:
                ax.set_ylabel(PERFORMANCE_LABEL[subpanel], fontsize=9)
            # Every row gets its own x-label, not just the bottom row of each
            # subpanel: DV rows plot negativity and CV rows plot squeezing --
            # different quantities, not a shared axis the way Figure 5's twin-axis
            # idiom this was copied from treats them.
            ax.set_xlabel(RESOURCE_LABEL[model], fontsize=9)
            for r in sorted(rows, key=lambda r: (float(r["coupling_strength"]), int(r["seed"]))):
                source_rows.append(
                    {
                        "subpanel": r["subpanel"],
                        "model": r["model"],
                        "encoding": r["encoding"],
                        "coupling_strength": r["coupling_strength"],
                        "seed": r["seed"],
                        "resource": r["resource"],
                        "performance": r["performance"],
                    }
                )

    axes[0, 0].text(-0.32, 1.28, r"\textbf{(a)}", fontsize=12,
                    transform=axes[0, 0].transAxes, va="top", ha="left")
    axes[2, 0].text(-0.32, 1.28, r"\textbf{(b)}", fontsize=12,
                    transform=axes[2, 0].transAxes, va="top", ha="left")

    colourbar = fig.colorbar(
        scatter, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02
    )
    colourbar.set_label(r"Coupling strength $J$", fontsize=9)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "resource_performance_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    write_source_data("FigureS3", source_rows, out_dir=source_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
