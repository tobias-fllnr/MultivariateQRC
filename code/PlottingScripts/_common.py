"""Data loading, styling and source-data output shared by the figure scripts.

Three things every figure script needs live here, once each: the usetex rcParams, the table
of run IDs each figure resolves through, and the pickle load that returns one averaged
result set as a DataFrame. Everything figure-specific — every filter, slice and transform —
stays in the figure script that needs it.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The deposit root, two levels up from code/PlottingScripts/.
DEPOSIT = Path(__file__).resolve().parents[2]
PLOTS_DIR = DEPOSIT / "figures"
SOURCE_DATA_DIR = DEPOSIT / "data" / "SourceData"
RESULTS_AVERAGED_DIR = DEPOSIT / "data" / "Results_averaged"
RESULTS_RUN_DIR = DEPOSIT / "data" / "Results_run"

TASKS = ("mixing", "lorenz", "optuna_mixing", "optuna_lorenz")
# "esn" is a model for the two optuna tasks: it is the classical baseline that appears as
# a third row in Figures 2 and S1 and as a third panel in Figure 4. There is no ESN
# counterpart for the *general* grids -- the `mixing` and `lorenz` tasks -- which never ran
# it, so Figures 3, 5 and S2 have no classical row. That is not an omission: those figures
# plot negativity and squeezing beside the performance metric, and a classical reservoir
# has neither. A (task, model, encoding) triple that has no pin in RUN_IDS raises from
# averaged_path rather than resolving to a plausible wrong file.
MODELS = ("gaussian", "tfim", "esn")
ENCODINGS = ("one_to_one", "fill", "dense")

# The run that produced each result set a figure plots, pinned. Resolving the run ID at
# plot time -- from a ledger, or from whatever happens to sit in the results directory --
# would let a newly added run silently change which data a published figure shows, so every
# pin is written by hand instead.
#
# The two optuna keys resolve into experiments/hyperparameter_studies/ and the two general
# keys into experiments/parameter_scans/.
RUN_IDS: dict[tuple[str, str, str], str] = {
    # Figure 3 and Figure 5(a) sit at the gamma the holdout hyperparameter protocol
    # selects, rounded to the nearest integer, which is the operating point Figure 2
    # reports. The grids in experiments/parameter_scans/ are run at that gamma: 6, 5, 3
    # for CV local / clustered / global and 7, 4, 3 for DV.
    ("mixing", "gaussian", "one_to_one"): "0905130003",
    ("mixing", "gaussian", "fill"): "0905130004",
    ("mixing", "gaussian", "dense"): "0905130005",
    ("mixing", "tfim", "one_to_one"): "0905130004",
    ("mixing", "tfim", "fill"): "0905130005",
    ("mixing", "tfim", "dense"): "0905130006",
    # Supplementary Figure S2 and Figure 5(b) sit at the gamma the holdout hyperparameter
    # protocol selects at n = 6, d = 3, target all_1, rounded to the nearest integer -- the
    # same treatment the mixing pins above got. The grids in experiments/parameter_scans/
    # are run at that gamma: 7, 18, 13 for CV local / clustered / global and 10, 5, 3 for
    # DV. They carry `d == 3` alone, which is the only slice either figure plots.
    ("lorenz", "gaussian", "one_to_one"): "0906114150",
    ("lorenz", "gaussian", "fill"): "0906114151",
    ("lorenz", "gaussian", "dense"): "0906114152",
    ("lorenz", "tfim", "one_to_one"): "0906114151",
    ("lorenz", "tfim", "fill"): "0906114152",
    ("lorenz", "tfim", "dense"): "0906114153",
    # The mixing-capacity studies of experiments/hyperparameter_studies/. Figures 2 and S1
    # report `holdout_score_mean` over the twenty unseen EVAL_SEEDS realizations rather
    # than the search optimum over the tuning seeds, so the reported number carries no
    # selection bias from the search.
    ("optuna_mixing", "gaussian", "one_to_one"): "0903012719",
    ("optuna_mixing", "gaussian", "fill"): "0903012720",
    ("optuna_mixing", "gaussian", "dense"): "0903012721",
    ("optuna_mixing", "tfim", "one_to_one"): "0903012718",
    ("optuna_mixing", "tfim", "fill"): "0903012719",
    ("optuna_mixing", "tfim", "dense"): "0903012720",
    # The classical ESN row. Its ledger records five successive passes, so the pin matters
    # more here than anywhere else in this table: these are the fifth, which tunes
    # bias_scale as a fourth hyperparameter and searches spectral_radius to 10. A default
    # `average_runs_optuna` combine over that ledger would pool all five.
    ("optuna_mixing", "esn", "one_to_one"): "0903151603",
    ("optuna_mixing", "esn", "fill"): "0903151602",
    ("optuna_mixing", "esn", "dense"): "0903151601",
    # The Lorenz-63 studies of experiments/hyperparameter_studies/. Figure 4 reports
    # `holdout_score_mean` over the twenty unseen EVAL_SEEDS realizations, scored on a test
    # block the search never read either, so neither the hyperparameters nor the seeds were
    # selected on the data the reported number comes from.
    #
    # These grids sweep two targets, `x_1` and `all_1`. Figure 4 plots `x_1`, the
    # one-step-ahead NRMSE of the x component alone; `all_1` is carried only so its
    # optimised gamma can set the operating point of the Lorenz parameter scans the
    # `lorenz` pins above resolve to. So a Figure 4 that forgot to filter on target would
    # silently average two different metrics into one point.
    ("optuna_lorenz", "gaussian", "one_to_one"): "0903163856",
    ("optuna_lorenz", "gaussian", "fill"): "0903163855",
    ("optuna_lorenz", "gaussian", "dense"): "0903163854",
    ("optuna_lorenz", "tfim", "one_to_one"): "0903163857",
    ("optuna_lorenz", "tfim", "fill"): "0903163856",
    ("optuna_lorenz", "tfim", "dense"): "0903163855",
    # The classical baseline's Lorenz panel, the counterpart of the optuna_mixing/esn pins
    # above. Unlike those, this ledger records a single pass: the four-hyperparameter
    # search space was already settled by the mixing runs before these were run, so there
    # is no earlier pass to select against.
    ("optuna_lorenz", "esn", "one_to_one"): "0903163855",
    ("optuna_lorenz", "esn", "fill"): "0903163854",
    ("optuna_lorenz", "esn", "dense"): "0903163853",
}

_NAME_STEMS = {
    "mixing": "mixing_capacity_{model}",
    "lorenz": "lorenz63_{model}",
    "optuna_mixing": "optuna_mixing_capacity_{model}",
    "optuna_lorenz": "optuna_lorenz63_{model}",
}

_SUFFIXES = {
    "mixing": "averaged",
    "lorenz": "averaged",
    "optuna_mixing": "combined",
    "optuna_lorenz": "combined",
}

# The notebooks call this dimension "spin" in variable names and "DV-QRC" in panel labels;
# the result filenames spell it "tilted_tfim".
#
# The fragment carries the `qrc_` prefix rather than _NAME_STEMS, because the ESN is not a
# QRC and its results are named `optuna_mixing_capacity_esn`, with no such infix. Moving
# the prefix here leaves all 24 quantum paths character-for-character what they were.
_MODEL_FRAGMENTS = {
    "gaussian": "qrc_gaussian",
    "tfim": "qrc_tilted_tfim",
    "esn": "esn",
}


def averaged_path(task: str, model: str, encoding: str) -> Path:
    """The Results_averaged/ pickle holding one run's aggregated results."""
    key = (task, model, encoding)
    if key not in RUN_IDS:
        raise KeyError(
            f"no run ID pinned for {key}; tasks {TASKS}, models {MODELS}, "
            f"encodings {ENCODINGS}"
        )
    stem = _NAME_STEMS[task].format(model=_MODEL_FRAGMENTS[model])
    return RESULTS_AVERAGED_DIR / f"{stem}_results_{RUN_IDS[key]}_{_SUFFIXES[task]}.pkl"


def load_averaged(task: str, model: str, encoding: str) -> pd.DataFrame:
    """Loads one averaged result set as a DataFrame, unfiltered.

    No filtering happens here, deliberately. encoding_comparison_mixing.ipynb drops -inf
    rows from the optuna_mixing/tfim/dense frame and from that frame only — 10 of its 35
    rows. Filtering uniformly here would silently change the five other panels of Figure 2.
    Every filter, slice and transform belongs in the figure script that needs it.
    """
    with open(averaged_path(task, model, encoding), "rb") as handle:
        return pd.DataFrame(pickle.load(handle))


def assert_full_holdout(frame, label: str) -> None:
    """Refuses to plot a point whose mean rests on fewer than all twenty realizations.

    `run_optuna_job.holdout_moments` skips a failed seed rather than turning the point into
    -inf, so a frame can carry a mean conditional on the survivors. That is not neutral in
    either direction. For a *maximised* metric like the mixing capacity the dropped draws
    are the ones that would have pulled the mean down; for a *minimised* one like the
    Lorenz NRMSE a diverged reservoir yields a large or non-finite error, so dropping it
    removes exactly the draws that would have pulled the mean up. Either way such a point
    must not reach a figure unannotated.

    All 54 Lorenz studies and all 144 mixing studies came back with twenty of twenty, so
    nothing is currently annotated — this is what keeps that a fact rather than an
    assumption. Lives here rather than in one figure script because it is a property of the
    held-out protocol, not of any one figure; Figures 2, S1 and 4 all read it.
    """
    if "n_holdout_ok" not in frame.columns:
        raise KeyError(
            f"{label}: frame carries no 'n_holdout_ok'; it is not a holdout-protocol run"
        )
    short = frame.loc[frame["n_holdout_ok"] != 20]
    if not short.empty:
        # Whichever of the grid-identifying columns this task actually carries. A mixing
        # frame writes target = None throughout, so an all-null column is dropped rather
        # than printed as "target=None" on every line.
        id_columns = [
            column for column in ("n", "d", "target")
            if column in frame.columns and frame[column].notna().any()
        ]
        points = ", ".join(
            "(" + ", ".join(f"{column}={row[column]}" for column in id_columns) + ")"
            f": {int(row['n_holdout_ok'])}/20"
            for _, row in short.iterrows()
        )
        raise ValueError(
            f"{label}: {len(short)} point(s) rest on fewer than twenty holdout "
            f"realizations and would be plotted as if complete -- {points}"
        )


def run_dir(task: str, model: str, encoding: str) -> Path:
    """The Results_run/ directory holding one run's per-seed pickles.

    Only extract_correlations.py reads it: the figure scripts themselves read
    Results_averaged/ and SourceData/ alone, so a figure can be redrawn without the
    per-seed pickles this directory holds.
    """
    key = (task, model, encoding)
    if key not in RUN_IDS:
        raise KeyError(
            f"no run ID pinned for {key}; tasks {TASKS}, models {MODELS}, "
            f"encodings {ENCODINGS}"
        )
    stem = _NAME_STEMS[task].format(model=_MODEL_FRAGMENTS[model])
    return RESULTS_RUN_DIR / f"{stem}_results_{RUN_IDS[key]}"


def apply_paper_style() -> None:
    """The rcParams block every notebook set inline, verbatim.

    text.usetex means a LaTeX installation with Computer Modern is required — the same one
    latexmk needs for the manuscript.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })


def _cell(value):
    """Formats one CSV cell, keeping full float precision."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    return str(value)


def write_source_data(
    name: str, rows: list[dict], out_dir: Path = SOURCE_DATA_DIR
) -> Path:
    """Writes SourceData/<name>.csv from rows built out of the plotted arrays.

    Callers pass the very arrays they handed to ax.plot, ax.fill_between or ax.imshow, so
    the CSV cannot drift from the figure. Nothing is computed here. Column order comes from
    the first row, and every row must carry the same columns in the same order.
    """
    if not rows:
        raise ValueError(f"{name}: no source-data rows; the figure drew nothing")
    columns = list(rows[0])
    for index, row in enumerate(rows):
        if list(row) != columns:
            raise ValueError(
                f"{name}: row {index} has columns {list(row)}, expected {columns}"
            )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])
    return path
