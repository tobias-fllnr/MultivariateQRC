"""
Combines Optuna study results -- one JSON per grid point, each holding that point's
best parameters and its per-trial record -- into a single pickle file in Results_averaged/.

Usage:
    python average_runs_optuna.py --config ../experiments/<tree>/<slug>/grid.yaml --run-id <run id>
"""

import argparse
import os
import json
import pickle
import time
from pathlib import Path

from joblib import Parallel, delayed

from utils.gridconfig import GridConfig
from utils.provenance import read_run_ids
from utils.shotguard import check_shots_declared, check_trials_declared


# The filename prefix run_optuna_job.save_results writes. Anything else in a results
# directory is not a study result.
RESULT_PREFIX = "best_params_"


def load_single_json(file_path):
    """Loads a single JSON file and flattens 'best_params' into the top level."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'best_params' in data:
                params = data.pop('best_params')
                data.update(params)
            # None is how the payload encodes the exact limit, because json.dump renders
            # infinity as the non-standard `Infinity` token. Normalise here so every
            # downstream row carries a number.
            if data.get("n_shots", "absent") is None:
                data["n_shots"] = float("inf")
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def check_schema_not_mixed(results):
    """Refuses to combine the best-of-search schema with the holdout schema.

    A default invocation (no --run-id) combines every run ID in runs.yaml, so a re-run under
    a different protocol appends its rows to the earlier ones by default. The two schemas
    name the plotted point differently -- `best_score_mean`/`best_score_std` in the
    best-of-search rows, `holdout_score_mean` in the holdout ones -- so a plotting script's
    `dropna` on the old columns would silently drop exactly the new rows and re-plot the
    search optima under a new run ID, which is the selection bias the holdout protocol
    exists to remove.
    """
    has_best = any("best_score_mean" in row for row in results)
    has_holdout = any("holdout_score_mean" in row for row in results)
    if has_best and has_holdout:
        raise ValueError(
            "average_runs_optuna: this combine mixes the best-of-search schema "
            "('best_score_mean'/'best_score_std') with the holdout schema "
            "('holdout_score_mean'). Pass --run-id to select one run's schema alone."
        )


def combine_results_parallel(dates, name, base_dir="../data/Results_run", n_jobs=-1):
    """Loads JSON results from multiple date-stamped directories in parallel."""
    time_start = time.time()
    all_file_paths = []

    print("Scanning directories...")
    for date in dates:
        dir_path = os.path.join(base_dir, f"{name}_results_{date}")
        if os.path.exists(dir_path):
            files = [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                # Not every .json in a results directory is a result. Since job generation
                # started writing run_config.json for provenance, a bare `.json` filter
                # also swallowed that file and flattened it into the combined pickle as an
                # extra row -- one whose n, d and encoding_mode are all NaN and which drags
                # thirteen provenance columns (name, run_id, git_commit, ...) into the
                # frame. It is silent: the row survives a groupby on the grid axes as a NaN
                # group and only shows up as a row count one higher than the job count.
                # Matching the prefix run_optuna_job.save_results actually writes is what
                # keeps the next provenance file from doing the same thing.
                if fname.startswith(RESULT_PREFIX) and fname.endswith(".json")
            ]
            all_file_paths.extend(files)
        else:
            print(f"Warning: Directory not found: {dir_path}")

    print(f"Found {len(all_file_paths)} JSON files across {len(dates)} directories.")
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_single_json)(fp) for fp in all_file_paths
    )
    results = [r for r in results if r is not None]
    print(f"Loaded in {time.time() - time_start:.2f} seconds.")
    return results


def main(config_path, run_ids=None, out_dir="../data/Results_averaged",
         results_run_dir="../data/Results_run"):
    """Combines one experiment's Optuna JSON results. Returns the written pickle path."""
    config = GridConfig.from_yaml(config_path)
    if not run_ids:
        run_ids = read_run_ids(Path(config_path).parent)

    print(f"Looking for results in: {results_run_dir}")
    combined_results = combine_results_parallel(
        run_ids, config.name, base_dir=str(results_run_dir)
    )

    if not combined_results:
        print(f"No results found for {config.name} in runs {run_ids}. Exiting.")
        raise SystemExit(1)

    check_schema_not_mixed(combined_results)
    check_shots_declared(combined_results, config)
    check_trials_declared(combined_results, config)

    time_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    output_filepath = os.path.join(
        out_dir, f"{config.name}_results_{run_ids[-1]}_combined.pkl"
    )
    with open(output_filepath, "wb") as f:
        pickle.dump(combined_results, f)

    print(f"Saving completed in {time.time() - time_start:.2f} seconds.")
    print(f"Saved {len(combined_results)} combined results to: {output_filepath}")
    return output_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the experiment's grid.yaml")
    parser.add_argument(
        "--run-id", action="append", dest="run_ids",
        help="run ID to include; repeatable; defaults to every run in runs.yaml",
    )
    parser.add_argument("--out", default="../data/Results_averaged", help="output directory")
    options = parser.parse_args()
    main(options.config, run_ids=options.run_ids, out_dir=options.out)
