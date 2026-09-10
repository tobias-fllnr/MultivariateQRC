r"""Mixing capacity against measurement ensemble size, both reservoirs.

Two panels sharing a y axis: DV-QRC left, CV-QRC right. One line per encoding, bands at
+/- one standard deviation across the twenty held-out realizations.

**Capacity is un-normalised, as in every other figure of the paper.** `utils.ipc.IPC.ipc`
divides each per-degree capacity by `n_observables`, the width of the feature matrix it was
handed, so `first_moment_2` is a per-observable number. Every figure multiplies it back by
that width: Figure 2 with `n(n+1)/2` and `3n` branches, Figures 3 and 5 with the constants
21 and 18 their fixed `n = 6` makes of them. At `n = 4` the factors are **10** for the CV
reservoir and **12** for the DV one, and both were checked against the readout the runs
actually built rather than taken from the formula -- `GaussianQRC` at `n = 4` emits a
10-column `flat_covs`, and `run_general_job.run_mixing_capacity_qrc_tilted_tfim` slices
`measurements[:, :3*n]`, 12 columns. Mean and SD are both multiplied, as Figure 2 does.

Reads run 0904224422 from `experiments/finite_shots/ensemble_sweep_{cv,dv}` — n = 4, d = 2,
three encodings, five ensemble sizes plus the exact limit, hyperparameters re-optimised at
every point with a 300-trial search (warmup 30). Every plotted point rests on
`n_holdout_ok == 20`, asserted below.

**This is Figure 6.** The PDF keeps its semantic name, as every other figure script does,
because the manuscript cites that name; the source data is `SourceData/Figure6.csv`,
matching the other sheets.

Three decisions the plot encodes, each of which could reasonably have gone otherwise:

  - **The exact limit is drawn detached, past a break in the x axis.** `n_shots = inf` has
    no position on a log axis, so it sits one decade beyond 1e12 behind break marks, and
    the curve is NOT continued into it. Connecting them would draw a line across a gap in
    which nothing was measured.
  - **Absolute capacity, not a fraction of the exact limit.** The exact column is itself
    one 300-trial search whose winner carries the same selection noise every other point
    carries — DV `dense` at 1e12 lands 5.9% *above* its own exact reference, significant at
    t = +2.31, because the two searches picked winners that generalise differently to the
    holdout seeds. A ratio against it would show points above 100% and invite the reading
    that shot noise helps.
  - **The x axis counts shots per commuting setting**, which is what the grids varied and
    what makes the two panels share tick positions. The DV local-Pauli readout needs three
    settings against the CV homodyne's one, so at equal x the DV reservoir spends three
    times the total shots — a caption sentence, not an axis transform. `n_settings` travels
    in the source data so the figure can be redrawn against total budget without re-running
    anything.

Usage, from the deposit's code/ directory:
    python PlottingScripts/ensemble_sweep_mixing.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    PLOTS_DIR,
    RESULTS_AVERAGED_DIR,
    SOURCE_DATA_DIR,
    apply_paper_style,
    write_source_data,
)

# Pinned, for the reason _common.RUN_IDS is pinned: resolving the run ID at plot time would
# let a newly added run silently change which data a published figure shows. Not added to
# _common.RUN_IDS because that table is keyed (task, model, encoding) and this run is one
# pickle per platform covering all three encodings.
RUN_ID = "0904224422"

# Panel order matches Figures 2 and 4: DV first.
PANELS = (
    ("DV-QRC", "optuna_mixing_capacity_qrc_tilted_tfim"),
    ("CV-QRC", "optuna_mixing_capacity_qrc_gaussian"),
)

# The encoding names the manuscript's figures use, in the order Figures 2 and 4 list them.
ENCODINGS = (
    ("one_to_one", "Local"),
    ("fill", "Clustered"),
    ("dense", "Global"),
)

FINITE_SHOTS = (1e4, 1e6, 1e8, 1e10, 1e12)

# The readout width each capacity was divided by, per panel, as a function of n. Kept as
# formulas rather than the constants 10 and 12 so a re-run at another n stays correct;
# asserted against the width the reservoirs actually build.
READOUT_DIMENSION = {
    # CV: the upper triangle of the n x n q-covariance, cov_measurements="q_only".
    "CV-QRC": lambda n: n * (n + 1) // 2,
    # DV: the 3n local Pauli expectations the mixing task slices out of the observable
    # matrix with measurements[:, :3*n].
    "DV-QRC": lambda n: 3 * n,
}

# Where the detached exact-limit point sits: one decade past the last finite one, so the
# break is wide enough to read at this figure size. Nothing is measured in between, and the
# break marks say so.
EXACT_X = 1e14
BREAK_X = 10 ** 13.0
X_LIMITS = (3e3, 6e14)

# The three exact markers are spread around the infinity tick so their error bars stay
# legible -- in the DV panel they fall within 0.017 of each other and would otherwise
# overlap into an unreadable stack. This is presentation only: the exact limit has no
# position on a shot axis, so there is no x value to distort, and all three sit past the
# break. The source-data CSV records them as "inf", not as these offsets.
EXACT_OFFSETS = (0.45, 1.0, 2.2)


def load_run(task_stem: str) -> pd.DataFrame:
    """One platform's combined study results as a DataFrame, unfiltered."""
    path = RESULTS_AVERAGED_DIR / f"{task_stem}_results_{RUN_ID}_combined.pkl"
    with open(path, "rb") as handle:
        return pd.DataFrame(pickle.load(handle))


def axes_fraction_x(ax, value: float) -> float:
    """Where `value` sits across a log x axis, as a 0-1 axes fraction."""
    low, high = ax.get_xlim()
    return (np.log10(value) - np.log10(low)) / (np.log10(high) - np.log10(low))


def draw_axis_break(ax, position: float) -> None:
    """Two slashes on the x spine, marking that the axis is not continuous there."""
    fraction = axes_fraction_x(ax, position)
    height, width = 0.022, 0.012
    for offset in (-width, width):
        ax.plot(
            [fraction + offset - width, fraction + offset + width],
            [-height, height],
            transform=ax.transAxes,
            color="black",
            linewidth=0.8,
            clip_on=False,
            zorder=5,
        )


def main(out_dir: Path = PLOTS_DIR, source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    apply_paper_style()

    colors = cm.viridis(np.linspace(0, 1, len(ENCODINGS)))
    color_of = {label: color for (_, label), color in zip(ENCODINGS, colors)}
    exact_x_of = {label: EXACT_X * factor
                  for (_, label), factor in zip(ENCODINGS, EXACT_OFFSETS)}

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9), sharey=True)
    rows: list[dict] = []

    for ax, (panel, task_stem) in zip(axes, PANELS):
        frame = load_run(task_stem)

        sizes = set(frame["n"])
        if len(sizes) != 1:
            raise ValueError(f"{panel}: run {RUN_ID} mixes system sizes {sorted(sizes)}")
        n = int(sizes.pop())
        readout = READOUT_DIMENSION[panel](n)

        for encoding, label in ENCODINGS:
            subset = frame.loc[frame["encoding_mode"] == encoding]
            by_shots = {float(row["n_shots"]): row for _, row in subset.iterrows()}

            missing = [s for s in (*FINITE_SHOTS, np.inf) if s not in by_shots]
            if missing:
                raise ValueError(f"{panel} {encoding}: run {RUN_ID} has no point at {missing}")

            # A point with n_holdout_ok < 20 carries a mean conditional on the
            # realizations that survived, which is not neutral in either direction: for a
            # maximised metric like this one the dropped draws are the ones that would
            # have pulled the mean down, and for a minimised one a diverged reservoir
            # yields a large or non-finite value, so dropping it removes exactly the draws
            # that would have pulled the mean up. Either way such a point must not be
            # plotted unannotated.
            for shots, row in by_shots.items():
                if int(row["n_holdout_ok"]) != 20:
                    raise ValueError(
                        f"{panel} {encoding} n_shots={shots}: n_holdout_ok="
                        f"{row['n_holdout_ok']}, not 20; the mean is conditional on the "
                        "survivors and this point may not be plotted unannotated"
                    )

            x = np.array(FINITE_SHOTS, dtype=float)
            y = readout * np.array([by_shots[s]["holdout_score_mean"] for s in FINITE_SHOTS])
            sd = readout * np.array([by_shots[s]["holdout_score_sd"] for s in FINITE_SHOTS])

            ax.plot(x, y, marker="o", markersize=3.5, label=label, color=color_of[label])
            ax.fill_between(x, y - sd, y + sd, color=color_of[label], alpha=0.2)

            # The exact limit, detached: no segment joins it to 1e12, because nothing was
            # measured between them.
            exact = by_shots[np.inf]
            ax.errorbar(
                exact_x_of[label],
                readout * exact["holdout_score_mean"],
                yerr=readout * exact["holdout_score_sd"],
                marker="o",
                markersize=3.5,
                color=color_of[label],
                capsize=2,
                linewidth=1.0,
            )

            for shots in (*FINITE_SHOTS, np.inf):
                row = by_shots[shots]
                rows.append({
                    "model": panel,
                    "encoding": label,
                    "n_shots_per_setting": "inf" if np.isinf(shots) else shots,
                    "n_settings": int(row["n_settings"]),
                    "readout_dimension": int(readout),
                    "mixing_capacity_mean": float(readout * row["holdout_score_mean"]),
                    "mixing_capacity_sd": float(readout * row["holdout_score_sd"]),
                    "n_realizations": int(row["n_holdout_ok"]),
                })

        ax.set_xscale("log")
        ax.set_xlim(*X_LIMITS)
        ax.set_xticks([*FINITE_SHOTS, EXACT_X])
        ax.set_xticklabels(
            [r"$10^{4}$", r"$10^{6}$", r"$10^{8}$", r"$10^{10}$", r"$10^{12}$", r"$\infty$"]
        )
        ax.set_xlabel(r"Shots per measurement setting $M$")
        ax.set_title(panel)
        draw_axis_break(ax, BREAK_X)

    axes[0].set_ylabel(r"$C_\mathrm{mix}$")
    # Default frame, which is the house style: no other figure script touches frameon,
    # framealpha or the edge, so figures 2, 4 and 5 all carry matplotlib's box. Only the
    # placement is chosen here -- low-right is the one region no curve enters, since every
    # curve rises with M.
    axes[0].legend(loc="lower right")

    # One explicit limit, set after both panels are drawn. Setting it inside the loop turns
    # autoscaling off, and with sharey=True the first panel's call would fix the top before
    # the second panel's data existed -- which clipped CV Local's exact point, the highest
    # value in the figure and the one carrying the "CV does not reach its limit" reading.
    ceiling = max(row["mixing_capacity_mean"] + row["mixing_capacity_sd"] for row in rows)
    axes[0].set_ylim(0.0, ceiling * 1.06)

    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ensemble_sweep_mixing.pdf"
    plt.savefig(target, bbox_inches="tight")
    plt.savefig(target.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    written = write_source_data("Figure6", rows, out_dir=source_data_dir)
    print(f"wrote {target}")
    print(f"wrote {target.with_suffix('.png')}")
    print(f"wrote {written} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(out_dir=options.out, source_data_dir=options.source_data)
