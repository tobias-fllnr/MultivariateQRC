"""
Aggregates per-seed results from Results_run/ into averaged results in Results_averaged/.

Usage:
    python average_runs_general.py --config ../experiments/<tree>/<slug>/grid.yaml --run-id <run id>
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from utils.gridconfig import GridConfig
from utils.provenance import read_run_ids
from utils.shotguard import check_shots_declared

PARAM_KEYS = ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma']


def aggregate_nested_dicts(dict_list):
    """Recursively averages nested dictionaries of arrays/scalars across seeds."""
    if not dict_list:
        return {}, {}
    if not isinstance(dict_list[0], dict):
        return np.mean(dict_list, axis=0), np.std(dict_list, axis=0)

    mean_dict = {}
    std_dict = {}
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    for k in all_keys:
        sub_list = [d[k] for d in dict_list if k in d]
        if sub_list:
            mean_dict[k], std_dict[k] = aggregate_nested_dicts(sub_list)
    return mean_dict, std_dict


def average_results(results, param_keys=None):
    """Groups results by hyperparameters and computes mean/std across seeds.

    `param_keys` is the group key: the parameters that identify one point of the grid, so
    that everything varying within a group is a seed. It defaults to PARAM_KEYS, the
    QRC grid's axes except `seed`: leaving the seed out is what makes it a grouping
    key rather than an identity. Callers that hold a grid should pass
    `GridConfig.group_keys()` instead, so an added axis is grouped over rather than
    silently pooled as though its values were seeds.
    """
    param_keys = list(PARAM_KEYS if param_keys is None else param_keys)

    grouped = {}
    for result in results:
        missing = [k for k in param_keys if k not in result]
        if missing:
            raise KeyError(
                f"result is missing the grid axes {missing}; the grid declares them but "
                f"the job script does not record them in its result dict. Result keys: "
                f"{sorted(result)}"
            )
        key = tuple(result[k] for k in param_keys)
        grouped.setdefault(key, []).append(result)

    averaged_results = []
    for key, group in grouped.items():
        averaged = {k: key[i] for i, k in enumerate(param_keys)}
        averaged['num_seeds'] = len(group)

        # Average regular keys (scalars/arrays)
        skip = set(param_keys) | {'seed'}
        regular_keys = [k for k in group[0] if k not in skip and 'breakdown' not in k]
        for k in regular_keys:
            values = [entry[k] for entry in group if k in entry]
            if values:
                averaged[f"{k}_mean"] = np.mean(values, axis=0)
                averaged[f"{k}_std"] = np.std(values, axis=0)

        # Average breakdown keys (nested dictionaries)
        breakdown_keys = [k for k in group[0] if 'breakdown' in k]
        for b_key in breakdown_keys:
            b_dicts = [entry[b_key] for entry in group if b_key in entry]
            if b_dicts:
                b_mean, b_std = aggregate_nested_dicts(b_dicts)
                averaged[f"{b_key}_mean"] = b_mean
                averaged[f"{b_key}_std"] = b_std

        averaged_results.append(averaged)
    return averaged_results


def load_single_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_results_parallel(dates, name, base_dir="../data/Results_run", n_jobs=-1):
    """Loads pickle results from multiple date-stamped directories in parallel."""
    time_start = time.time()
    all_file_paths = []

    print("Scanning directories...")
    for date in dates:
        dir_path = os.path.join(base_dir, f"{name}_results_{date}")
        if os.path.exists(dir_path):
            files = [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".pkl")
            ]
            all_file_paths.extend(files)
        else:
            print(f"Warning: Directory not found: {dir_path}")

    print(f"Found {len(all_file_paths)} files across {len(dates)} directories.")
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_single_pickle)(fp) for fp in all_file_paths
    )
    print(f"Loaded in {time.time() - time_start:.2f} seconds.")
    return results


def main(config_path, run_ids=None, out_dir="../data/Results_averaged",
         results_run_dir="../data/Results_run"):
    """Averages the runs of one experiment across seeds. Returns the written pickle path."""
    config = GridConfig.from_yaml(config_path)
    if not run_ids:
        run_ids = read_run_ids(Path(config_path).parent)

    print(f"Looking for results in: {results_run_dir}")
    results = load_results_parallel(run_ids, config.name, base_dir=str(results_run_dir))

    if not results:
        print(f"No results found for {config.name} in runs {run_ids}. Exiting.")
        raise SystemExit(1)

    check_shots_declared(results, config)

    time_start = time.time()
    averaged_results = average_results(results, param_keys=config.group_keys())

    os.makedirs(out_dir, exist_ok=True)
    output_filepath = os.path.join(
        out_dir, f"{config.name}_results_{run_ids[-1]}_averaged.pkl"
    )
    with open(output_filepath, "wb") as f:
        pickle.dump(averaged_results, f)

    print(f"Averaging completed in {time.time() - time_start:.2f} seconds.")
    print(f"Saved averaged results to: {output_filepath}")
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
