"""Records which grid produced which run.

Two records are written. `run_config.json` goes into the run's own results directory and
holds the resolved grid, the job count and the generation time; `runs.yaml`, written beside
the grid.yaml the run was generated from, gains one entry per run of that grid. grid.yaml is
hand-written and never rewritten by these functions; runs.yaml is machine-written and never
hand-edited.

This deposit carries no runs.yaml. The ledgers recorded commits of the repository the code
was developed in, so instead the run ID behind every deposited result is stated explicitly:
in each grid's README, and on the command line as `--run-id`. Preparing a fresh run from
this deposit writes both records again, and read_run_ids then resolves against that new
ledger.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from utils.gridconfig import GridConfig


def allocate_run_id(name: str, results_root: str | Path, now: datetime | None = None) -> str:
    """A timestamp run ID whose results directory does not exist yet.

    The ID used to be `%m%d%H%M`, and the minute resolution silently destroyed runs: the
    results directory, the args file and run_config.json are all named from `name` and the
    run ID, so two grids prepared inside one minute shared all three and the second
    overwrote the first. That is not a rare case here -- the three mixing encodings share
    the `name` `optuna_mixing_capacity_qrc_gaussian`, so preparing them back to back is
    exactly the collision.

    Seconds make an accidental collision unlikely; the loop makes it impossible. A taken
    second is stepped over rather than reused, and the ID stays a plain timestamp, which is
    what read_run_ids, the results directory name and the RUN_IDS pins all treat it as.
    """
    results_root = Path(results_root)
    stamp = now or datetime.now()
    for offset in range(3600):
        run_id = (stamp + timedelta(seconds=offset)).strftime("%m%d%H%M%S")
        if not (results_root / f"{name}_results_{run_id}").exists():
            return run_id
    raise RuntimeError(
        f"no free run ID for {name!r} within an hour of {stamp:%Y-%m-%d %H:%M:%S} under "
        f"{results_root}; something is creating results directories in a loop"
    )


def git_state() -> tuple[str, bool]:
    """Returns the current commit hash and whether the working tree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _resolved_axes(config: GridConfig) -> list[dict]:
    """The distinct values per axis, rendered exactly as the args lines render them."""
    axes = []
    for index, name in enumerate(config.axis_names):
        seen = {}
        for combination in config.combinations:
            seen[str(combination[index])] = None
        axes.append({name: list(seen)})
    return axes


def write_run_config(
    results_dir: str | Path, config: GridConfig, run_id: str, job_count: int
) -> Path:
    """Writes run_config.json into the results directory and returns its path."""
    results_dir = Path(results_dir)
    commit, dirty = git_state()
    payload = {
        "name": config.name,
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_path": str(config.source_path),
        "config_sha256": _sha256_of(config.source_path),
        "grid_fingerprint": config.fingerprint(),
        "axes": _resolved_axes(config),
        "filter": config.filter_expr,
        # None for every grid that runs the protocol's search budget.
        "n_trials": config.n_trials,
        "n_startup_trials": config.n_startup_trials,
        "job_count": job_count,
        "target_script": config.target_script,
        "python": platform.python_version(),
    }
    path = results_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_run(
    config_dir: str | Path,
    run_id: str,
    job_count: int,
    commit: str,
    submitted: str | None = None,
    verified_against_results: bool | None = None,
) -> None:
    """Appends one entry to the experiment's runs.yaml ledger, creating it if absent."""
    config_dir = Path(config_dir)
    path = config_dir / "runs.yaml"
    ledger = yaml.safe_load(path.read_text()) if path.exists() else None
    ledger = ledger or {"runs": []}

    entry = {
        "run_id": run_id,
        "submitted": submitted or datetime.now().astimezone().isoformat(timespec="minutes"),
        "commit": commit,
        "jobs": job_count,
    }
    if verified_against_results is not None:
        entry["verified_against_results"] = verified_against_results

    ledger["runs"].append(entry)
    path.write_text(yaml.safe_dump(ledger, sort_keys=False))


def read_run_ids(config_dir: str | Path) -> list[str]:
    """Returns the run IDs recorded for an experiment, in ledger order."""
    path = Path(config_dir) / "runs.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no runs.yaml in {config_dir}; pass --run-id explicitly, or prepare a run first"
        )
    ledger = yaml.safe_load(path.read_text()) or {"runs": []}
    return [str(entry["run_id"]) for entry in ledger["runs"]]
