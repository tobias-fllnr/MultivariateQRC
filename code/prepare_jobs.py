"""Generates job argument files and submission scripts for QRC experiments.

The grid comes from a YAML config; every grid behind a figure of the manuscript is
deposited as one under experiments/. Three execution modes are generated every time:
  - Local:  a bash script that runs all jobs sequentially.
  - Condor: an HTCondor submit file.
  - Slurm:  a Slurm array job script.

Usage:
    python prepare_jobs.py --config ../experiments/<tree>/<slug>/grid.yaml
    python prepare_jobs.py --config <path> --dry-run
"""

import argparse
import math
import os
from pathlib import Path

from utils.gridconfig import GridConfig
from utils.provenance import allocate_run_id, append_run, git_state, write_run_config

# How wide an optuna job forks its seed evaluations, and therefore how many cores it asks
# for. Condor measured 8.9 of 10 cores busy on a real study, so the width earns its
# allocation -- but only 34 of the pool's 63 machines have ten cores at all, and a DV n=6
# study is fourteen hours while a CV one is minutes. So the DV grids keep ten and the CV
# grids take five, which nearly every machine can match and which costs a CV job about
# eight extra minutes. Both divide the ten tuning seeds and the twenty holdout seeds
# evenly. run_optuna_job.worker_count() reads the value back out of the environment.
DV_WORKERS = 10
CV_WORKERS = 5

# Peak RSS measured on cluster nodes, one worker process, 2000 steps of a mixing study:
#   CV  n = 2..10       202-209 MB, flat in n
#   DV  n = 4 / 5 / 6   212 / 241 / 365 MB   (the n=6 figure is at gamma=100, its worst)
# About 200 MB of that is the shared interpreter, numpy and qutip image, which forked
# workers do not duplicate -- Condor measured a whole ten-worker CV job at 293 MB against
# its 8000 MB request. What does scale is the DV density matrix and its Liouvillian, by 4x
# per qubit: the private residue is ~165 MB at n=6, ~41 MB at n=5, ~12 MB at n=4.
#
# Sizing every optuna job for the DV n=6 worst case is what made that request 8000 MB, and
# the memory clause alone then rejected 21 of the pool's 108 slots. Sizing per grid instead
# lets the CV grids match nearly every machine while DV n=6 keeps real headroom.
PARENT_MB = 300
WORKER_BASE_MB = 60
DV_STATE_MB_AT_N6 = 165
MEMORY_HEADROOM = 1.5
MEMORY_FLOOR_MB = 1500


def is_dv(config: GridConfig) -> bool:
    """True for the tilted-TFIM reservoir, the only one whose state grows with n."""
    return "tilted_tfim" in config.name


def optuna_workers(config: GridConfig) -> int:
    return DV_WORKERS if is_dv(config) else CV_WORKERS


def optuna_memory_mb(config: GridConfig, workers: int | None = None) -> int:
    """The memory one optuna job of `config` needs, sized from the grid's largest n.

    One submit file covers a whole grid, so the largest n in it sets the request. Rounded
    up to 500 MB because Condor matches against whole slots anyway.
    """
    workers = optuna_workers(config) if workers is None else workers
    largest_n = max(
        combination[config.axis_names.index("n")] for combination in config.combinations
    )
    # The DV state grows 4x per qubit; the CV covariance is 2n x 2n and effectively flat.
    state_mb = DV_STATE_MB_AT_N6 * 4.0 ** (largest_n - 6) if is_dv(config) else 0.0
    needed = PARENT_MB + workers * (WORKER_BASE_MB + state_mb)
    return max(MEMORY_FLOOR_MB, int(math.ceil(needed * MEMORY_HEADROOM / 500.0) * 500))


def job_resources(config: GridConfig) -> tuple[int, int]:
    """(cpus, memory in MB) for one grid's jobs. Each backend formats the unit itself."""
    if config.name.startswith("optuna_"):
        return optuna_workers(config), optuna_memory_mb(config)
    return 1, 2048


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the experiment's grid.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the job count and the first and last args lines; write nothing",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    config = GridConfig.from_yaml(options.config)
    config.validate_axis_order()
    name = config.name
    job_count = len(config.combinations)

    request_cpus, request_memory_mb = job_resources(config)
    # The job environment. QRC_MAX_WORKERS follows the cores Condor allocated;
    # QRC_N_TRIALS appears only when the grid asks for a search budget other than the
    # protocol's, so every grid that says nothing produces exactly the environment, and
    # therefore exactly the study, it always did.
    job_env = {"QRC_MAX_WORKERS": str(request_cpus)}
    if config.n_trials is not None:
        job_env["QRC_N_TRIALS"] = str(config.n_trials)
    if config.n_startup_trials is not None:
        job_env["QRC_N_STARTUP_TRIALS"] = str(config.n_startup_trials)
    run_id = allocate_run_id(name, "../data/Results_run")
    relative_outdir = f"../data/Results_run/{name}_results_{run_id}"
    args_lines = config.args_lines(relative_outdir)

    if options.dry_run:
        print(f"--- {job_count} jobs would be generated for '{name}' ---")
        print(f"Config    : {config.source_path}")
        print(f"Axes      : {', '.join(config.axis_names)}")
        print(f"Filter    : {config.filter_expr or 'none'}")
        print(f"Resources : {request_cpus} cpus, {request_memory_mb}MB")
        print(f"Environment: {' '.join(f'{k}={v}' for k, v in job_env.items())}")
        print(f"First args: {args_lines[0]}")
        print(f"Last args : {args_lines[-1]}")
        return

    # The deposit root: prepare_jobs.py itself sits in code/, which is
    # exactly where the generated Condor initialdir and Slurm cd must land --
    # run_general_job.py, the relative jobs/ directory and the relative
    # ../data/Results_run output path are all correct only from there, not
    # from whatever directory the reader happened to invoke this from.
    project_dir = str(Path(__file__).resolve().parent)
    os.makedirs(relative_outdir, exist_ok=True)
    os.makedirs("jobs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    args_filename = f"jobs/args_{name}_{run_id}.txt"
    local_filename = f"jobs/run_local_{name}_{run_id}.sh"
    condor_filename = f"jobs/submit_{name}_{run_id}.sub"
    slurm_filename = f"jobs/submit_{name}_{run_id}.slurm"

    target_script = config.target_script
    condor_env = " ".join(f"{key}={value}" for key, value in job_env.items())
    local_exports = "\n".join(f"export {key}={value}" for key, value in job_env.items())
    slurm_exports = local_exports

    # A. Write arguments file
    with open(args_filename, "w") as f:
        f.write("\n".join(args_lines) + "\n")

    # B. Generate local run script (no cluster required)
    with open(local_filename, "w") as f:
        f.write(f"""\
#!/bin/bash
# Local execution script - runs all {job_count} jobs sequentially.
# Usage: bash {local_filename}

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    PY=python
else
    PY=python3
fi
{local_exports}
TOTAL=$(wc -l < {args_filename})
COUNT=0

while IFS= read -r ARGS; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running: $PY {target_script}.py $ARGS"
    $PY {target_script}.py $ARGS
done < {args_filename}

echo "All $TOTAL jobs completed."
""")
    os.chmod(local_filename, 0o755)

    # C. Generate HTCondor submit file
    #    Uses initialdir so jobs run from the project directory on the shared filesystem.
    with open(condor_filename, "w") as f:
        f.write(f"""\
initialdir = {project_dir}
Executable = {target_script}.sh
Arguments = $(ARGS)
Log    = logs/$(Cluster)_$(Process).log
Output = logs/$(Cluster)_$(Process).out
Error  = logs/$(Cluster)_$(Process).err
should_transfer_files = NO
request_cpus = {request_cpus}
request_memory = {request_memory_mb}MB
environment = "{condor_env}"
Queue ARGS from {args_filename}
""")

    # D. Generate Slurm array job script
    with open(slurm_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --cpus-per-task={request_cpus}
#SBATCH --mem={request_memory_mb}M
#SBATCH --time=2:00:00

cd {project_dir}
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    PY=python
else
    PY=python3
fi
{slurm_exports}

OFFSET=${{OFFSET:-0}}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + OFFSET))

ARGS=$(sed -n "${{LINE_NUM}}p" {args_filename})

$PY {target_script}.py $ARGS
""")

    # E. Record provenance
    write_run_config(relative_outdir, config, run_id, job_count)
    commit, dirty = git_state()
    append_run(
        Path(config.source_path).parent,
        run_id=run_id,
        job_count=job_count,
        commit=f"{commit}-dirty" if dirty else commit,
    )

    print(f"--- Generated {job_count} jobs (run ID: {run_id}) ---")
    print(f"Arguments file  : {args_filename}")
    print(f"Provenance      : {relative_outdir}/run_config.json")
    print(f"Ledger          : {Path(config.source_path).parent}/runs.yaml")
    print(f"")
    print(f"To run locally (no cluster):")
    print(f"  bash {local_filename}")
    print(f"")
    print(f"To submit on HTCondor:")
    print(f"  condor_submit {condor_filename}")
    print(f"")
    print(f"To submit on Slurm:")
    print(f"  sbatch --array=1-{job_count} {slurm_filename}")


if __name__ == "__main__":
    main()
