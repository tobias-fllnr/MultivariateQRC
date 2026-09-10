"""Turns a grid.yaml experiment configuration into ordered parameter combinations.

The rendering of values is load-bearing: job argument lines are produced with str() over
the numpy scalars the generators emit, and the result filenames carry exactly that
formatting. Values are never coerced, rounded or reformatted on the way through.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import yaml

GENERATORS = ("values", "logspace", "linspace", "range")

# The positional argv signatures the two job scripts read. This table is the single source
# of truth for them: run_general_job.py imports expected_axis_names and parses its own
# argv through it, rather than unpacking a fixed number of positions, so the four QRC tasks
# and the ESN tasks can read different signatures without the grid and the job script
# disagreeing about what each position means. run_optuna_job.py dispatches on the full task
# name in its own __main__ and its tasks read two different signatures.
GENERAL_AXES = (
    "n", "d", "encoding_mode", "dt",
    "encoding_strength", "coupling_strength", "gamma", "seed",
)
OPTUNA_MIXING_TASKS = (
    "optuna_mixing_capacity_qrc_tilted_tfim",
    "optuna_mixing_capacity_qrc_gaussian",
)
OPTUNA_MIXING_AXES = ("n", "d", "encoding_mode")
OPTUNA_LORENZ_TASKS = (
    "optuna_lorenz63_qrc_tilted_tfim",
    "optuna_lorenz63_qrc_gaussian",
)
OPTUNA_LORENZ_AXES = ("n", "d", "target", "encoding_mode")

# The ESN baseline reads seven axes: it has no dt, and its three tuned hyperparameters are
# not the quantum reservoirs'. Its two optuna tasks read the same argv shapes the QRC
# optuna tasks do, so they reuse OPTUNA_MIXING_AXES and OPTUNA_LORENZ_AXES.
ESN_GENERAL_TASKS = ("mixing_capacity_esn", "lorenz63_esn")
ESN_AXES = (
    "n", "d", "encoding_mode",
    "input_scaling", "spectral_radius", "leak_rate", "seed",
)
OPTUNA_ESN_MIXING_TASKS = ("optuna_mixing_capacity_esn",)
OPTUNA_ESN_LORENZ_TASKS = ("optuna_lorenz63_esn",)

# The one optional axis, and the whole reason it is declared last. Appending rather than
# inserting keeps GENERAL_AXES and both optuna tuples strict *prefixes* of the extended
# signatures, so a grid that does not declare it produces exactly the argv, exactly the
# result filenames and exactly the result schema of a grid without it. Inserting n_shots
# before `seed`, or adding it to GENERAL_AXES, would change every result filename. The cost
# is cosmetic: `seed` is no longer the last field of a filename that carries a shot count.
SHOT_AXIS = ("n_shots",)


def expected_axis_names(name: str) -> tuple[str, ...]:
    """The axis names, in argv order, that `name`'s target job script reads positionally.

    Used to catch the failure mode a wrong arity does not: an axis list with the right
    names but the wrong order, or a duplicated axis name, both corrupt a run silently
    rather than raising at the target script's own unpacking line.
    """
    if name in OPTUNA_MIXING_TASKS or name in OPTUNA_ESN_MIXING_TASKS:
        return OPTUNA_MIXING_AXES
    if name in OPTUNA_LORENZ_TASKS or name in OPTUNA_ESN_LORENZ_TASKS:
        return OPTUNA_LORENZ_AXES
    if name.startswith("optuna_"):
        raise ValueError(
            f"{name!r} is not a recognised optuna task name; expected one of "
            f"{OPTUNA_MIXING_TASKS + OPTUNA_LORENZ_TASKS + OPTUNA_ESN_MIXING_TASKS + OPTUNA_ESN_LORENZ_TASKS}"
        )
    if name in ESN_GENERAL_TASKS:
        return ESN_AXES
    return GENERAL_AXES


def optional_axis_names(name: str) -> tuple[str, ...]:
    """Axes `name`'s target script reads only if its grid declares them, in append order.

    The ESN is a classical reservoir with no measurement ensemble, so it has none -- which
    is what makes an n_shots axis on an ESN grid an error caught by validate_axis_order,
    before any job is written, rather than a parameter silently ignored at run time.
    """
    classical = (
        ESN_GENERAL_TASKS + OPTUNA_ESN_MIXING_TASKS + OPTUNA_ESN_LORENZ_TASKS
    )
    if name in classical:
        return ()
    return SHOT_AXIS


def axis_names_for(name: str, count: int) -> tuple[str, ...]:
    """The axis names, in argv order, that a `count`-axis grid maps onto for `name`.

    The required axes, plus as many of the optional ones as the count accounts for. Both
    the grid validation and run_general_job's argv parsing go through this one function,
    so the argv order a grid produces and the argv order the job script reads still cannot
    drift apart -- the property expected_axis_names was introduced for.
    """
    required = expected_axis_names(name)
    optional = optional_axis_names(name)
    extra = count - len(required)
    if not 0 <= extra <= len(optional):
        raise ValueError(
            f"{name!r} reads {len(required)} required axes {required}"
            + (f" plus up to {len(optional)} optional {optional}" if optional else
               " and takes no optional axes")
            + f"; got {count}"
        )
    return required + optional[:extra]


def expand_generator(spec: dict) -> list:
    """Expands one axis generator spec into its list of values."""
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(
            f"an axis generator needs exactly one key, one of {GENERATORS}; got {spec!r}"
        )
    kind, argument = next(iter(spec.items()))

    if kind == "values":
        if not argument:
            raise ValueError("'values' is empty; an axis needs at least one value")
        return list(argument)
    if kind == "logspace":
        start, stop, num = argument
        return list(np.logspace(start, stop, num))
    if kind == "linspace":
        start, stop, num = argument
        return list(np.linspace(start, stop, num))
    if kind == "range":
        return list(range(argument))

    raise ValueError(f"unknown generator {kind!r}; expected one of {GENERATORS}")


def parse_axes(axes: list) -> tuple[tuple[str, ...], list[list[tuple]]]:
    """Expands the ordered axes list into names and per-group value tuples.

    Returns the axis names in the order written — which is the positional argv order the
    job scripts read — and, for each axis group, the tuples of values it contributes. A
    plain axis contributes 1-tuples; a zip group contributes one tuple per position.
    """
    names: list[str] = []
    groups: list[list[tuple]] = []

    for entry in axes:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"an axes entry needs exactly one key, an axis name or 'zip'; got {entry!r}"
            )
        key, spec = next(iter(entry.items()))

        if key == "zip":
            member_names = list(spec)
            columns = [expand_generator(spec[name]) for name in member_names]
            lengths = {len(column) for column in columns}
            if len(lengths) != 1:
                raise ValueError(
                    f"zip axes {member_names} must have the same length; got "
                    f"{[len(c) for c in columns]}"
                )
            names.extend(member_names)
            groups.append([tuple(values) for values in zip(*columns)])
        else:
            names.append(key)
            groups.append([(value,) for value in expand_generator(spec)])

    return tuple(names), groups


def passes_filter(axis_names: tuple[str, ...], combination: tuple, expr: str) -> bool:
    """Evaluates a filter expression with the axis names bound.

    `expr` is trusted, author-written configuration, not untrusted input -- clearing
    builtins from the eval namespace is not a sandbox and is not meant as one. Its purpose
    is to catch a typo or a stray name in a hand-written filter early, with a clear error,
    not to contain hostile code.
    """
    namespace = dict(zip(axis_names, combination))
    try:
        return bool(eval(expr, {"__builtins__": {}}, namespace))
    except Exception as error:
        raise ValueError(f"filter {expr!r} failed on {namespace}: {error}") from error


@dataclass(frozen=True)
class GridConfig:
    """A resolved experiment grid: what to run, in the order the job scripts expect."""

    name: str
    axis_names: tuple[str, ...]
    combinations: tuple[tuple, ...]
    filter_expr: str | None
    source_path: Path
    # The search budget, for optuna grids that need one other than the protocol's 100. It is
    # deliberately NOT an axis: it does not vary within a grid, it must not enter the argv
    # (which would change every result filename), and it must not enter the sampler seed
    # (a 300-trial study is then a strict extension of the 100-trial one at the same grid
    # point -- TPE's suggestions depend only on the seed and the trials before them, so
    # trials 0..99 are identical and best-of-300 >= best-of-100 by construction).
    # prepare_jobs passes it to the job through the environment, the way it already
    # passes the worker count. None means the protocol default.
    n_trials: int | None = None
    # The TPE warmup, on the same terms and for the same reasons. None means optuna's
    # default of 10. Unlike n_trials this one does NOT leave a smaller study's trials
    # intact -- changing it changes every suggestion from trial 10 onward.
    n_startup_trials: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GridConfig":
        path = Path(path)
        document = yaml.safe_load(path.read_text()) or {}

        name = document.get("name")
        if not name:
            raise ValueError(f"{path}: 'name' is required")
        axes = document.get("axes")
        if not axes:
            raise ValueError(f"{path}: 'axes' is required and must be a non-empty list")

        axis_names, groups = parse_axes(axes)
        filter_expr = document.get("filter")
        search_settings = {}
        for key in ("n_trials", "n_startup_trials"):
            value = document.get(key)
            if value is None:
                search_settings[key] = None
                continue
            if not name.startswith("optuna_"):
                raise ValueError(
                    f"{path}: {key!r} configures a hyperparameter search and only means "
                    f"anything for an optuna task; {name!r} is run by run_general_job, "
                    "which has no search"
                )
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"{path}: {key!r} must be a positive integer; got {value!r}"
                )
            search_settings[key] = value
        n_trials = search_settings["n_trials"]
        n_startup_trials = search_settings["n_startup_trials"]
        if n_startup_trials is not None and n_trials is not None and n_startup_trials > n_trials:
            raise ValueError(
                f"{path}: 'n_startup_trials' ({n_startup_trials}) exceeds 'n_trials' "
                f"({n_trials}); the whole budget would be a random search"
            )

        combinations = []
        for grouped in product(*groups):
            combination = tuple(value for group in grouped for value in group)
            if filter_expr is None or passes_filter(axis_names, combination, filter_expr):
                combinations.append(combination)

        return cls(
            name=name,
            axis_names=axis_names,
            combinations=tuple(combinations),
            filter_expr=filter_expr,
            source_path=path,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
        )

    @property
    def target_script(self) -> str:
        return "run_optuna_job" if self.name.startswith("optuna_") else "run_general_job"

    def validate_axis_order(self) -> None:
        """Raises unless axis_names exactly matches what the target script reads
        positionally for this task name -- same names, same order, no duplicates.

        A wrong arity already fails loudly when the target script unpacks its argv. This
        catches the two failure modes that would not: a reordered axis list with the right
        arity, and a duplicated axis name (which also produces the right arity by displacing
        a different axis) -- both corrupt a run's parameters silently instead of raising.
        Exposed as a method, rather than folded into from_yaml, so a caller building a
        GridConfig for something other than job generation is not forced through it, and any
        caller that does generate jobs -- the CLI or otherwise -- can call it explicitly.

        An optional trailing axis (currently only `n_shots`) is accepted when the grid
        declares it, and rejected when it appears anywhere but last -- a grid listing
        n_shots before `seed` has the right arity and the right names, which is exactly
        the failure mode arity checking does not catch.
        """
        required = expected_axis_names(self.name)
        try:
            expected = axis_names_for(self.name, len(self.axis_names))
        except ValueError:
            # A wrong arity is an axis-order problem too, and this method reports every
            # such problem in one message shape -- which run_general_job's argv parsing
            # deliberately does not, since there an arity error is the more precise
            # complaint. Falling back to the required axes names the signature the grid
            # should have had; an optional axis cannot rescue a count that does not fit.
            expected = required
        if self.axis_names != expected:
            raise ValueError(
                f"{self.source_path}: axis order {self.axis_names} does not match what "
                f"{self.name!r} reads positionally; expected {expected}"
            )

    def group_keys(self) -> tuple[str, ...]:
        """The axes that identify one parameter point: every axis except `seed`.

        This is the key the averaging groups by. Deriving it from the grid rather than
        hard-coding it means a grid that adds an axis is averaged over that axis
        correctly, with no matching edit in the averaging script — where omitting the new
        axis would silently pool its values together as though they were seeds.
        """
        return tuple(name for name in self.axis_names if name != "seed")

    def args_lines(self, outdir: str) -> list[str]:
        """Renders one argument line per combination: axis values, then outdir, then name."""
        return [
            " ".join([*(str(value) for value in combination), outdir, self.name])
            for combination in self.combinations
        ]

    def fingerprint(self) -> str:
        """A sha256 over the resolved grid, for provenance and change detection."""
        payload = json.dumps(
            {
                "name": self.name,
                "axis_names": list(self.axis_names),
                "filter": self.filter_expr,
                "n_trials": self.n_trials,
                "n_startup_trials": self.n_startup_trials,
                "combinations": [
                    [str(value) for value in combination]
                    for combination in self.combinations
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
