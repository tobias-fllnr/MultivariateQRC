#!/bin/bash
# Reproduces the figures of
#
#   Tobias Fellner, Jonas Merklinger and Christian Holm,
#   "Multivariate quantum reservoir computing with discrete and continuous variable systems"
#
# from the data deposited alongside this script.
#
#   ./reproduce_all.sh --tier figures    redraw the figures from the aggregated data (minutes)
#   ./reproduce_all.sh --tier average    re-aggregate the raw per-seed data, then redraw
#   ./reproduce_all.sh --tier simulate   generate the simulation jobs (59,514 of them) and stop
#
# Needs bash 4 or newer (tier `average` falls through into tier `figures`), Python 3.12
# with the pinned requirements, and a LaTeX installation -- the figure scripts typeset
# their labels with LaTeX and fail without one. See README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/code"

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
    PY=python
else
    PY=python3
fi

# The header comment above is the usage message. $0 is resolved against $ROOT rather than
# read directly, because this script has already cd'd into code/ by now and a relative $0
# would no longer point at anything.
SELF="$ROOT/$(basename "$0")"

usage() {
    sed -n '2,15p' "$SELF"
}

TIER=figures
while [ $# -gt 0 ]; do
    case "$1" in
        --tier)
            # `shift 2` on a lone --tier would fail under `set -e` and exit without a
            # word of explanation, so the missing value is caught here instead.
            if [ $# -lt 2 ]; then
                echo "--tier needs a value: figures, average or simulate" >&2
                usage >&2
                exit 2
            fi
            TIER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Each grid is aggregated over the run ID whose per-seed results this deposit carries.
# Naming it explicitly is required, not a convenience: without --run-id the averaging
# scripts look for a runs.yaml ledger, which records commits of a repository a reader of
# this deposit cannot fetch and is therefore not deposited.
average_general() {
    echo "=== $1"
    $PY average_runs_general.py --config "$1" --run-id "$2"
}

average_optuna() {
    echo "=== $1"
    $PY average_runs_optuna.py --config "$1" --run-id "$2"
}

FIGURE_SCRIPTS=(
    figure2_encoding_comparison_mixing.py
    figure3_heatmap_mixing.py
    figure4_encoding_comparison_lorenz.py
    figure5_combined_mixing_lorenz.py
    ensemble_sweep_mixing.py
    figureS2_heatmap_lorenz.py
    figureS3_resource_scatter.py
)

case "$TIER" in
  simulate)
    for grid in ../experiments/*/*/grid.yaml; do
        echo "=== $grid"
        $PY prepare_jobs.py --config "$grid"
    done
    echo
    echo "Jobs generated under code/jobs/. Each grid printed a local, an HTCondor and a"
    echo "Slurm command above. Every run allocates a fresh run ID, so nothing already in"
    echo "data/Results_run/ is overwritten."
    exit 0
    ;;
  average)
    # Parameter scans
    average_general ../experiments/parameter_scans/lorenz_gaussian_dense/grid.yaml 0906114152
    average_general ../experiments/parameter_scans/lorenz_gaussian_fill/grid.yaml 0906114151
    average_general ../experiments/parameter_scans/lorenz_gaussian_one_to_one/grid.yaml 0906114150
    average_general ../experiments/parameter_scans/lorenz_tfim_dense/grid.yaml 0906114153
    average_general ../experiments/parameter_scans/lorenz_tfim_fill/grid.yaml 0906114152
    average_general ../experiments/parameter_scans/lorenz_tfim_one_to_one/grid.yaml 0906114151
    average_general ../experiments/parameter_scans/mixing_gaussian_dense/grid.yaml 0905130005
    average_general ../experiments/parameter_scans/mixing_gaussian_fill/grid.yaml 0905130004
    average_general ../experiments/parameter_scans/mixing_gaussian_one_to_one/grid.yaml 0905130003
    average_general ../experiments/parameter_scans/mixing_tfim_dense/grid.yaml 0905130006
    average_general ../experiments/parameter_scans/mixing_tfim_fill/grid.yaml 0905130005
    average_general ../experiments/parameter_scans/mixing_tfim_one_to_one/grid.yaml 0905130004
    # Hyperparameter studies
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_esn_dense/grid.yaml 0903163853
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_esn_fill/grid.yaml 0903163854
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_esn_one_to_one/grid.yaml 0903163855
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_gaussian_dense/grid.yaml 0903163854
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_gaussian_fill/grid.yaml 0903163855
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_gaussian_one_to_one/grid.yaml 0903163856
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_tfim_dense/grid.yaml 0903163855
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_tfim_fill/grid.yaml 0903163856
    average_optuna ../experiments/hyperparameter_studies/optuna_lorenz_tfim_one_to_one/grid.yaml 0903163857
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_esn_dense/grid.yaml 0903151601
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_esn_fill/grid.yaml 0903151602
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_esn_one_to_one/grid.yaml 0903151603
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_gaussian_dense/grid.yaml 0903012721
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_gaussian_fill/grid.yaml 0903012720
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_gaussian_one_to_one/grid.yaml 0903012719
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_tfim_dense/grid.yaml 0903012720
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_tfim_fill/grid.yaml 0903012719
    average_optuna ../experiments/hyperparameter_studies/optuna_mixing_tfim_one_to_one/grid.yaml 0903012718
    # Measurement-ensemble sweeps
    average_optuna ../experiments/finite_shots/ensemble_sweep_cv/grid.yaml 0904224422
    average_optuna ../experiments/finite_shots/ensemble_sweep_dv/grid.yaml 0904224422
    # Reads the raw per-seed data and writes data/SourceData/Figure5_correlations.csv and
    # data/SourceData/FigureS3.csv. It must run before figure5_combined_mixing_lorenz.py
    # and figureS3_resource_scatter.py, which read those two files rather than recomputing
    # them -- running the figures first would draw the statistics deposited here over data
    # that has just been re-aggregated.
    echo "=== extract_correlations.py"
    $PY PlottingScripts/extract_correlations.py
    ;&
  figures)
    for script in "${FIGURE_SCRIPTS[@]}"; do
        echo "=== $script"
        $PY "PlottingScripts/$script"
    done
    # ensemble_sweep_mixing.py writes a PNG preview beside its PDF. The deposited figure
    # set is the 8 PDFs and nothing reads the preview, so it is removed
    # here rather than left behind as an extra file in figures/.
    rm -f "$ROOT"/figures/*.png
    echo
    echo "Figures written to figures/, source data to data/SourceData/."
    ;;
  *)
    echo "unknown tier: $TIER (expected figures, average or simulate)" >&2
    usage >&2
    exit 2
    ;;
esac
