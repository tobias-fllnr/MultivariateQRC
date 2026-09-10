r"""Correlation statistics behind Figure 5's annotations and Supplementary Figure S3.

This is the only script that reads the per-seed Results_run/. It writes two CSVs under
SourceData/, and both figure scripts read those, so a figure never depends on the raw
per-seed data.

It reads the same runs Figure 5 plots, through the same _common.RUN_IDS pins, so the
statistics and the curves cannot drift apart.

Two measures per curve; see utils/correlation.py for what each one holds constant.
Performance is signed so that larger is better: the mixing capacity as Figure 5(a) plots
it (multiplied by the observable count), and minus the one-step-ahead NRMSE for (b).

Usage, from the deposit's code/ directory:
    python PlottingScripts/extract_correlations.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PlottingScripts._common import (  # noqa: E402
    SOURCE_DATA_DIR,
    run_dir,
    write_source_data,
)
from utils.correlation import (  # noqa: E402
    PRESENCE_FRAC,
    fisher_interval,
    fisher_pool,
    peak_offset_decades,
    per_coupling_spearman,
    resource_present,
    sweep_spearman,
)

N_SEEDS = 20
N_COUPLINGS = 19

Curve = namedtuple(
    "Curve",
    "subpanel model_label encoding_label task model encoding encoding_strength multiplier",
)

# The twelve subplots of Figure 5, in its own row order. The multiplier is the observable
# count Figure 5(a) scales the mixing capacity by -- 3n = 18 for the DV readout's local
# Pauli expectation values, n(n+1)/2 = 21 for the CV q-covariances, both at n = 6. It is 1
# for panel (b), which plots the NRMSE unscaled. Spearman is invariant under positive
# scaling, so the multiplier changes the scatter's axis and none of the coefficients.
CURVES = (
    Curve("a", "DV-QRC", "Local", "mixing", "tfim", "one_to_one", 0.1, 18.0),
    Curve("a", "DV-QRC", "Clustered", "mixing", "tfim", "fill", 0.1, 18.0),
    Curve("a", "DV-QRC", "Global", "mixing", "tfim", "dense", 0.1, 18.0),
    Curve("a", "CV-QRC", "Local", "mixing", "gaussian", "one_to_one", 0.1, 21.0),
    Curve("a", "CV-QRC", "Clustered", "mixing", "gaussian", "fill", 0.1, 21.0),
    Curve("a", "CV-QRC", "Global", "mixing", "gaussian", "dense", 0.1, 21.0),
    Curve("b", "DV-QRC", "Local", "lorenz", "tfim", "one_to_one", 1.0, 1.0),
    Curve("b", "DV-QRC", "Clustered", "lorenz", "tfim", "fill", 1.0, 1.0),
    Curve("b", "DV-QRC", "Global", "lorenz", "tfim", "dense", 1.0, 1.0),
    Curve("b", "CV-QRC", "Local", "lorenz", "gaussian", "one_to_one", 1.0, 1.0),
    Curve("b", "CV-QRC", "Clustered", "lorenz", "gaussian", "fill", 1.0, 1.0),
    Curve("b", "CV-QRC", "Global", "lorenz", "gaussian", "dense", 1.0, 1.0),
)


def resource_key(model: str) -> str:
    """The quantum resource each platform is measured by."""
    return "negativity" if model == "tfim" else "squeezing"


def _performance(record: dict, curve: Curve) -> float:
    """Performance for one per-seed record, signed so that larger is better."""
    if curve.task == "mixing":
        return curve.multiplier * float(record["first_moment_2"])
    # Element 0 of the length-10 horizon axis is the one-step-ahead test NRMSE, which is
    # what Figure 5(b) plots. Negated so that larger is better, as in panel (a).
    return -float(np.asarray(record["first_moment_nrmse_test_average"])[0])


def collect(curve: Curve) -> tuple[dict[float, list[tuple[float, float, int]]], float]:
    """Reads one curve's per-seed records, returning them grouped by coupling.

    Returns (pairs_by_coupling, gamma). Each entry is a list of (resource, performance,
    seed) triples. A run directory carries exactly one gamma and one input dimension, so
    gamma is derived here and asserted unique rather than pinned a second time.
    """
    directory = run_dir(curve.task, curve.model, curve.encoding)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} is absent: this script reads the raw per-seed pickles of the "
            f"runs Figure 5 pins, and that run's directory is not in data/Results_run/."
        )
    key = resource_key(curve.model)
    grouped: dict[float, list[tuple[float, float, int]]] = {}
    gammas: set[float] = set()
    for path in sorted(directory.glob("*.pkl")):
        with open(path, "rb") as handle:
            record = pickle.load(handle)
        if abs(float(record["encoding_strength"]) - curve.encoding_strength) > 1e-12:
            continue
        gammas.add(float(record["gamma"]))
        coupling = float(record["coupling_strength"])
        grouped.setdefault(coupling, []).append(
            (
                float(np.asarray(record[key]).reshape(-1)[0]),
                _performance(record, curve),
                int(record["seed"]),
            )
        )
    gamma = _validate_grouped(
        grouped,
        gammas,
        label=f"{directory.name} at encoding strength {curve.encoding_strength}",
    )
    return grouped, gamma


def _validate_grouped(
    grouped: dict[float, list[tuple[float, float, int]]],
    gammas: set[float],
    label: str,
) -> float:
    """Validates one curve's grouped per-seed triples, returning the single gamma.

    Split out of collect() so its three raise paths -- more than one gamma at the
    curve's encoding strength, a coupling count other than N_COUPLINGS, or a coupling
    with fewer than N_SEEDS seeds -- can be exercised directly against constructed
    dicts, without needing Results_run/ on disk.
    """
    if len(gammas) != 1:
        raise ValueError(f"{label}: expected one gamma, found {sorted(gammas)}")
    if len(grouped) != N_COUPLINGS:
        raise ValueError(f"{label}: {len(grouped)} couplings, expected {N_COUPLINGS}")
    for coupling, triples in grouped.items():
        if len(triples) != N_SEEDS:
            raise ValueError(
                f"{label}: coupling {coupling} has {len(triples)} seeds, "
                f"expected {N_SEEDS}. A partial grid point must not silently produce a "
                f"correlation over fewer realizations."
            )
    return gammas.pop()


def statistics_row(curve: Curve, grouped, gamma: float) -> dict:
    """The one Figure5_correlations.csv row for a curve."""
    couplings = np.array(sorted(grouped), dtype=float)
    resource_means = np.array(
        [np.mean([t[0] for t in grouped[c]]) for c in couplings], dtype=float
    )
    performance_means = np.array(
        [np.mean([t[1] for t in grouped[c]]) for c in couplings], dtype=float
    )

    rho_sweep = sweep_spearman(resource_means, performance_means)

    present = resource_present(resource_means, frac=PRESENCE_FRAC)
    pairs = {
        float(coupling): [(t[0], t[1]) for t in grouped[float(coupling)]]
        for coupling, keep in zip(couplings, present)
        if keep
    }
    per_coupling = per_coupling_spearman(pairs)
    rho_fixed, standard_error, pooled = fisher_pool(
        list(per_coupling.values()), n_seeds=N_SEEDS
    )
    low, high = fisher_interval(rho_fixed, standard_error)

    return {
        "subpanel": curve.subpanel,
        "model": curve.model_label,
        "encoding": curve.encoding_label,
        "resource_quantity": resource_key(curve.model),
        "performance_quantity": "C_mix" if curve.task == "mixing" else "minus_NRMSE",
        "gamma": gamma,
        "rho_sweep": rho_sweep,
        "rho_fixed": rho_fixed,
        "ci_low": low,
        "ci_high": high,
        "k_couplings": pooled,
        "n_seeds": N_SEEDS,
        "peak_offset_decades": peak_offset_decades(
            couplings, resource_means, performance_means
        ),
    }


def scatter_rows(curve: Curve, grouped) -> list[dict]:
    """One FigureS3.csv row per (coupling, seed) for a curve."""
    rows = []
    for coupling in sorted(grouped):
        for resource, performance, seed in sorted(grouped[coupling], key=lambda t: t[2]):
            rows.append(
                {
                    "subpanel": curve.subpanel,
                    "model": curve.model_label,
                    "encoding": curve.encoding_label,
                    "coupling_strength": coupling,
                    "seed": seed,
                    "resource": resource,
                    "performance": performance,
                }
            )
    return rows


def main(source_data_dir: Path = SOURCE_DATA_DIR) -> None:
    statistics, scatter = [], []
    for curve in CURVES:
        grouped, gamma = collect(curve)
        statistics.append(statistics_row(curve, grouped, gamma))
        scatter.extend(scatter_rows(curve, grouped))
        row = statistics[-1]
        print(
            f"({row['subpanel']}) {row['model']:7s} {row['encoding']:10s} "
            f"rho_sweep={row['rho_sweep']:+.2f}  "
            f"rho_fixed={row['rho_fixed']:+.2f} "
            f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]  "
            f"k={row['k_couplings']:2d}  "
            f"peak offset={row['peak_offset_decades']:+.2f} dec"
        )
    write_source_data("Figure5_correlations", statistics, out_dir=source_data_dir)
    write_source_data("FigureS3", scatter, out_dir=source_data_dir)
    print(f"\nWrote {len(statistics)} statistics rows and {len(scatter)} scatter rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    options = parser.parse_args()
    main(source_data_dir=options.source_data)
