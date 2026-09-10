"""Averaging guards for the two non-axis parameters an optuna study can vary.

`n_shots` and `n_trials` are both declared in grid.yaml rather than as grid axes, so
neither reaches the result filename and neither stops two runs of the same slug from
landing in one experiment directory. Both averaging scripts then combine *every* run ID in
that directory by default, which is where these guards earn their place.

The original of the pair, and the reason for the module's name:

Both averaging scripts default to combining *every* run ID listed in an experiment
directory's runs.yaml ledger, and name their output after the last of them. This deposit
ships no ledger and names each run ID on the command line instead, but a run prepared from
it writes one. An exact run and a finite-shot run placed in the same experiment directory
would therefore not overwrite each other's pickle -- the filenames differ -- but they would
be pooled into a single combined pickle in which a plotting script could silently average
across ensemble sizes. Keeping exact and finite-shot runs in separate experiment
directories is what prevents that; this guard is what catches it if a --run-id is ever
passed by hand across trees.

Modelled on average_runs_optuna.check_schema_not_mixed, which guards the analogous hazard
for the holdout result schema.
"""

from __future__ import annotations

from utils import shots


def declared_n_shots(config) -> set[float]:
    """The ensemble sizes `config` asks for. `{inf}` when it declares no n_shots axis."""
    if "n_shots" not in config.axis_names:
        return {shots.EXACT}
    position = config.axis_names.index("n_shots")
    return {float(combination[position]) for combination in config.combinations}


def row_n_shots(row) -> float:
    """One result row's ensemble size.

    An absent key means the exact limit, which is how an exact-limit result reads -- the
    key is written only when a grid declares the axis. `None` means the same thing and is
    how the optuna payload encodes it, because json.dump renders infinity as the
    non-standard `Infinity` token.
    """
    value = row.get("n_shots")
    if value is None:
        return shots.EXACT
    return float(value)


def check_shots_declared(rows, config) -> None:
    """Refuses to combine rows whose measurement ensemble the grid does not declare."""
    expected = declared_n_shots(config)
    observed = {row_n_shots(row) for row in rows}
    stray = observed - expected

    if stray:
        raise ValueError(
            f"{config.source_path}: this combine mixes measurement ensembles the grid does "
            f"not declare. Grid declares n_shots in "
            f"{sorted(shots.label(value) for value in expected)}; the results also contain "
            f"{sorted(shots.label(value) for value in stray)}. Exact and finite-shot runs "
            "belong to different experiment directories; pass --run-id to select one run."
        )


# The protocol's search budget, and optuna's own TPE warmup. Both are what a result written
# before either was configurable must be read as.
PROTOCOL_TRIALS = 100
PROTOCOL_STARTUP = 10


def declared_n_trials(config) -> int:
    """The search budget `config` asks for. 100, the protocol default, when it says nothing."""
    return PROTOCOL_TRIALS if config.n_trials is None else int(config.n_trials)


def _search_field(row, key):
    """Reads one search-budget field from a study row.

    run_optuna_job.build_payload nests these under `protocol`, and
    average_runs_optuna.load_single_json flattens only `best_params` -- so `protocol` is
    still a dict on the row when the guard sees it. The top-level lookup is the fallback,
    for a caller that flattened the block itself. Absent either way means the protocol
    default, which is how every result written before these were configurable reads.
    """
    protocol = row.get("protocol")
    if isinstance(protocol, dict) and protocol.get(key) is not None:
        return protocol[key]
    return row.get(key)


def row_n_trials(row) -> int:
    """One study's search budget."""
    value = _search_field(row, "n_trials")
    return PROTOCOL_TRIALS if value is None else int(value)


def declared_search_budget(config) -> tuple[int, int]:
    """The (trials, warmup) pair `config` asks for."""
    startup = config.n_startup_trials
    return (
        declared_n_trials(config),
        PROTOCOL_STARTUP if startup is None else int(startup),
    )


def row_search_budget(row) -> tuple[int, int]:
    """One study's (trials, warmup) pair."""
    startup = _search_field(row, "n_startup_trials")
    return (
        row_n_trials(row),
        PROTOCOL_STARTUP if startup is None else int(startup),
    )


def check_trials_declared(rows, config) -> None:
    """Refuses to combine studies searched with a budget the grid does not declare.

    The hazard is the same shape as check_shots_declared's and it is not caught by
    check_schema_not_mixed, because a 100-trial study and a 300-trial one carry the *same*
    keys -- both are the holdout schema. Pooled, they would produce a curve whose points
    were searched with different effort, which is exactly the artefact the deeper search
    was run to remove: a point can then sit below its neighbour because it was searched
    less hard, not because its capacity is lower.

    A re-run under a different budget is a different protocol and belongs in its own
    experiment directory rather than being appended to an existing one's ledger. This guard
    is what makes that a rule rather than a convention.

    The warmup counts as part of the budget, and it has to: changing n_startup_trials
    changes every TPE suggestion from trial 10 onward, so two studies that agree on
    n_trials but not on the warmup are still different searches.
    """
    expected = declared_search_budget(config)
    observed = {row_search_budget(row) for row in rows}
    stray = sorted(observed - {expected})

    if stray:
        shape = lambda pair: f"n_trials={pair[0]} n_startup_trials={pair[1]}"
        raise ValueError(
            f"{config.source_path}: this combine mixes search budgets the grid does not "
            f"declare. Grid declares {shape(expected)}; the results also contain "
            f"{[shape(pair) for pair in stray]}. A re-run under a different budget is a "
            "different protocol and belongs in its own experiment directory; pass "
            "--run-id to select one run."
        )
