import os

# Set thread limits before importing numerical libraries to prevent
# thread oversubscription when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import copy
import concurrent.futures
import hashlib
import json
import optuna
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass

from utils import shots
from utils.gridconfig import axis_names_for, optional_axis_names
from run_general_job import (
    lorenz_split,
    run_mixing_capacity_qrc_tilted_tfim,
    run_mixing_capacity_qrc_gaussian,
    run_lorenz63_qrc_tilted_tfim,
    run_lorenz63_qrc_gaussian,
    run_mixing_capacity_esn,
    run_lorenz63_esn,
)

# The search budget of the protocol, mirrored exactly for the ESN so its headline number
# is comparable to the QRC numbers. Named constants rather than literals so a test can
# shrink a study without rewriting the optimisers.
N_TRIALS = 100


def trial_count() -> int:
    """The search budget for this study.

    N_TRIALS = 100 is the protocol, and it is what Figures 2 and 4 report. A grid may ask
    for a different budget with a top-level `n_trials:` in its grid.yaml, which
    prepare_jobs passes in the environment exactly as it passes QRC_MAX_WORKERS --
    so the budget is declared with the experiment rather than edited into this module, and
    a grid that says nothing runs the protocol.

    The measurement-ensemble sweep is why this exists. Under shot noise the objective is
    0.0 over most of the search space, which starves TPE of any gradient: at n=6 two
    `one_to_one` studies returned no non-zero trial in 100, and at n=4 the DV `one_to_one`
    study at 1e8 shots found a region significantly worse than both its neighbours in M.
    Capacity cannot fall as the ensemble grows if both points sit at their optimum, so that
    is a search deficiency and not a measurement.

    Nothing about the sampler seed depends on this, deliberately -- see GridConfig.n_trials.
    """
    return max(1, int(os.environ.get("QRC_N_TRIALS", N_TRIALS)))


# optuna's own TPESampler default. Named here so the value this project relies on is
# explicit, and so passing it deliberately is indistinguishable from passing nothing --
# verified: TPESampler(seed=s) and TPESampler(seed=s, n_startup_trials=10) produce
# identical suggestion sequences. That equality is what makes this change safe for a
# Condor job of an earlier run that is evicted and restarts against this file: its
# environment was fixed in its ClassAd at submit time and carries no override, so it
# resumes on exactly the sampler it started with.
STARTUP_TRIALS = 10


def startup_trial_count() -> int:
    """How many trials TPE draws at random before it starts modelling.

    A grid may override it with a top-level `n_startup_trials:`, passed in the environment
    like the search budget. The ensemble sweep sets 30, holding the warmup at the 10%
    of budget the 100-trial protocol had rather than letting it fall to 3.3% when the total
    tripled.

    The warmup is not the lever it looks like, and the numbers are worth keeping. On a
    synthetic landscape that is exactly 0.0 over 94% of the search space -- the shape that
    starves TPE of a gradient -- a 300-trial budget gives, by warmup size:

        10 -> best 0.1997 (236 of 300 trials in the productive region)
        30 -> best 0.1998 (232)
       100 -> best 0.1998 (200)
       300 -> best 0.1560 ( 15)   pure random, 22% worse

    So 10, 30 and 100 are indistinguishable and pure random is much worse: TPE earns its
    keep as soon as it has any non-zero value at all. 30 is chosen for the proportion, not
    because the measurement demanded it.

    Note this does NOT preserve a smaller study's trials: changing the warmup changes every
    suggestion from trial 10 onward, so a 300-trial study at 30 is not an extension of a
    100-trial study at 10.
    """
    return max(1, int(os.environ.get("QRC_N_STARTUP_TRIALS", STARTUP_TRIALS)))

# The ESN's four tuned hyperparameters. input_scaling uses the same range as the QRC's
# encoding_strength. leak_rate is capped at 1, above which a leaky integrator overshoots
# rather than leaking.
#
# spectral_radius runs to 10, and the history of this bound is worth keeping, because both
# ends of it were wrong for a while.
#
# It began at 10, on the reasoning that the search should not be fenced inside the stable
# region. With a FIXED bias that was a mistake: TPE was attracted to the tail rather than
# indifferent to it, and in the first batch (runs 0903101311-0903101313, kept on disk)
# thirteen of 48 studies selected rho* > 1 with seven above 3, spending some 80% of the
# budget there and then collapsing on the holdout by as much as 0.085. It was cut to 2.0 on
# the echo state property: a reservoir without it is not an echo state network.
#
# Tuning the bias changed that argument. rho < 1 is the ESP condition linearised about zero
# bias; once b is large, tanh'(b) = sech^2(b) < 1, so what actually contracts is
# alpha * rho * <sech^2> + (1 - alpha). Measured at the six winners that pressed the 2.0
# cap, that quantity is 0.32 to 0.70, and all six are strictly contracting -- two different
# initial states converge to a separation of exactly zero within 500 steps. rho near 2 is
# therefore no ESP violation when the bias is tuned, the cap was binding rather than
# protecting (6 of 48 winners sat at 1.96-1.998), and the ceiling goes back to 10.
#
# The old failure mode is not impossible again: rho = 10 with a small B would genuinely not
# contract. So every run's winners are checked for contraction afterwards, and the two-stage
# holdout is what would expose a collapse regardless.
#
# The *lower* bound stays at 0.01, deliberately. Eighteen of the 48 studies chose
# rho* < 0.1 and the objective rises monotonically as rho falls there: a 0.1 floor would
# cost up to 0.068 capacity (dense n=6 d=4: 0.3323 -> 0.2846) and would hit the dense
# encoding, the ESN's strongest, hardest. Those low-rho optima are real -- they take 70-80%
# of their mixing capacity from zero-delay cross terms, so the reservoir is trading memory
# for instantaneous nonlinear mixing rather than failing.
# Each entry is (low, high, scale), with scale "log" or "linear".
#
# leak_rate is sampled LINEARLY, and it is the one place the ESN deliberately departs from
# the quantum tasks' sampling family. It is not a rate: it is the convex weight alpha in
# x_t = (1 - alpha) x_{t-1} + alpha tanh(...), bounded in (0, 1], whose interesting
# structure sits near 1. It was log-uniform only because it was written to mirror gamma,
# which genuinely does span [0.01, 100]. The first batch measured the cost: all 48 winners
# landed above 0.539 with a median of 0.902, while a log-uniform draw lands above 0.5 only
# 15% of the time -- so roughly 85% of every study's leak_rate budget was spent where no
# optimum has ever been found, and a 36-point uniform grid using 36% of the trial budget
# beat the 100-trial study by 40% at one_to_one n=6 d=3.
#
# leak_rate's lower bound is 0, the full range the convex weight can take. alpha = 0 is a
# frozen state and the constructor rejects it, but TPE did not draw the bound exactly in
# 1200 trials pushed hard against it, and a draw of exactly 0 records as one -inf trial
# rather than a crash.
#
# bias_scale is the fourth hyperparameter, and it has no quantum counterpart. tanh is odd,
# so a zero-bias reservoir is an exactly odd functional of its input and its degree-2
# mixing capacity is identically zero by symmetry -- the bias is the model's only source of
# the even nonlinearity that turns a mixture of two streams into their product. How much is
# wanted differs sharply by task: mixing wants 5 to 10 or more, while Lorenz-63 prediction
# rides on the linear response that a large bias saturates and prefers 0.25 to 1. The range
# is the same for both, so it is the search that resolves the tension rather than a choice
# of ours. It is a scale over more than two decades, hence log.
#
# The ceiling is 20 because 10 was measurably too low. Six paired 100-trial studies, three
# hyperparameters against four, improved the holdout at 6 of 6 by a median of 38% -- so the
# fourth parameter pays for itself several times over at an unchanged budget -- but three of
# the six winners came back at 9.86, 9.92 and 9.96 against a bound of 10, which is the bound
# shaping the answer rather than containing it. Unlike spectral_radius, no physical
# principle fixes 20; it is headroom above where the searches were pushing, and where the
# 48 winners land is itself the measurement -- at 20 none came within 10% of the bound and
# the median was 10.25, above the old ceiling.
ESN_SEARCH_SPACE = {
    "input_scaling": (0.001, 1.0, "log"),
    "spectral_radius": (0.01, 10.0, "log"),
    "leak_rate": (0.0, 1.0, "linear"),
    "bias_scale": (0.1, 20.0, "log"),
}


# The two-stage protocol. The search stage runs N_TRIALS trials, each averaged over the
# ten TUNE_SEEDS realizations. The holdout stage evaluates the selected hyperparameters
# once more, on the twenty EVAL_SEEDS realizations the search never saw and -- for Lorenz
# -- on a test block the search never saw either. The holdout is the number the figures
# report, so nothing is selected on the data its score is read from.
TUNE_SEEDS = tuple(range(10))
# Starts at 1000 so the block is disjoint from TUNE_SEEDS and from the general grids'
# seed: {range: 20}, keeping every seed block in the project distinguishable.
EVAL_SEEDS = tuple(range(1000, 1020))

# The default pool width, and what every run before per-grid resources used.
MAX_WORKERS = 10


def worker_count() -> int:
    """How many seeds to evaluate at once.

    prepare_jobs.py sizes request_cpus per grid and passes it in the environment,
    so the pool width follows the cores Condor actually allocated instead of assuming ten:
    the DV grids keep ten, the CV grids take five, whose studies are minutes long and match
    nearly every machine in the pool at the narrower width rather than the third of it that
    has ten cores free. Both widths divide the ten tuning seeds and the twenty holdout
    seeds evenly, so no wave is ragged.

    Nothing numerical depends on this. `evaluate_seeds` keys its results by seed and the
    scores are read back in TUNE_SEEDS order, so a study at five workers and the same study
    at ten produce the same trials.
    """
    return max(1, int(os.environ.get("QRC_MAX_WORKERS", MAX_WORKERS)))


# The Lorenz validation block. Carved out of the 4000-step evaluation block by
# run_general_job.lorenz_split, so the sequence length is unchanged.
LORENZ_VAL_LENGTH = 2000

# The QRC's three tuned hyperparameters and their search ranges. (low, high, scale), as
# ESN_SEARCH_SPACE. All three are log-uniform: every one is a rate or a coupling spanning
# decades, so the family is right for them, and these bounds are the ones the manuscript's
# Methods states.
QRC_SEARCH_SPACE = {
    "encoding_strength": (0.001, 1.0, "log"),
    "coupling_strength": (0.0001, 10.0, "log"),
    "gamma": (0.01, 100.0, "log"),
}


@dataclass(frozen=True)
class StudyTask:
    """One optuna task: what to run, which way is better, and what to search.

    `kind` carries the three things that differ between the two task families -- the
    optimisation direction, the objective's data block, and whether a validation block is
    requested at all -- so a task's protocol is stated once here rather than in six
    near-identical optimisers.
    """

    run_func: Callable[[dict], dict]
    model_name: str
    kind: str  # "mixing" or "lorenz"
    search_space: dict
    fixed_params: dict

    @property
    def direction(self) -> str:
        return "maximize" if self.kind == "mixing" else "minimize"

    @property
    def objective_split(self) -> str | None:
        """The block the search optimises. None for a task with no split at all."""
        return "val" if self.kind == "lorenz" else None


STUDY_TASKS = {
    "optuna_mixing_capacity_qrc_tilted_tfim": StudyTask(
        run_func=run_mixing_capacity_qrc_tilted_tfim, model_name="qrc_tilted_tfim",
        kind="mixing", search_space=QRC_SEARCH_SPACE, fixed_params={"dt": 1.0},
    ),
    "optuna_mixing_capacity_qrc_gaussian": StudyTask(
        run_func=run_mixing_capacity_qrc_gaussian, model_name="qrc_gaussian",
        kind="mixing", search_space=QRC_SEARCH_SPACE, fixed_params={"dt": 1.0},
    ),
    "optuna_lorenz63_qrc_tilted_tfim": StudyTask(
        run_func=run_lorenz63_qrc_tilted_tfim, model_name="qrc_tilted_tfim",
        kind="lorenz", search_space=QRC_SEARCH_SPACE,
        fixed_params={"dt": 1.0, "val_length": LORENZ_VAL_LENGTH},
    ),
    "optuna_lorenz63_qrc_gaussian": StudyTask(
        run_func=run_lorenz63_qrc_gaussian, model_name="qrc_gaussian",
        kind="lorenz", search_space=QRC_SEARCH_SPACE,
        fixed_params={"dt": 1.0, "val_length": LORENZ_VAL_LENGTH},
    ),
    "optuna_mixing_capacity_esn": StudyTask(
        run_func=run_mixing_capacity_esn, model_name="esn",
        kind="mixing", search_space=ESN_SEARCH_SPACE, fixed_params={},
    ),
    "optuna_lorenz63_esn": StudyTask(
        run_func=run_lorenz63_esn, model_name="esn",
        kind="lorenz", search_space=ESN_SEARCH_SPACE,
        fixed_params={"val_length": LORENZ_VAL_LENGTH},
    ),
}


def study_sampler_seed(task_name, n, d, encoding_mode, target=None,
                       n_shots=shots.EXACT) -> int:
    """A deterministic sampler seed for one grid point.

    optuna.create_study() leaves the TPE sampler unseeded, which makes a study
    irreproducible -- a problem for the DARUS deposit as much as for us. Deriving the seed
    from the grid point keeps it stable across re-runs and distinct between points.

    The shot count joins the key only when finite, so an exact-limit study's seed does not
    depend on the ensemble machinery existing at all. Two studies at different ensemble
    sizes are different searches and get different seeds.
    """
    parts = [task_name, n, d, encoding_mode, target]
    if not shots.is_exact(n_shots):
        parts.append(f"nshots{shots.label(n_shots)}")
    key = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def run_single_seed(run_func, params, seed):
    """Helper function to run a single simulation in an isolated process."""
    local_params = copy.deepcopy(params)
    local_params['seed'] = seed
    try:
        return run_func(local_params)
    except Exception as e:
        print(f"Seed {seed} failed with error: {e}")
        return None


def evaluate_seeds(run_func, parameters, seeds) -> dict:
    """Runs one parameter set once per seed, in parallel. Returns {seed: result or None}."""
    results = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(seeds), worker_count())
    ) as executor:
        futures = {
            executor.submit(run_single_seed, run_func, parameters, seed): seed
            for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    return results


def get_nrmse_element(data_dict, keyword, split="test"):
    """Retrieves an NRMSE element from a result dictionary based on a keyword like 'x_1'.

    `split` names the data block: 'val' during the search, 'test' for the holdout. The
    default is 'test', the block the reported number is read from.
    """
    try:
        prefix, num_str = keyword.split('_')
        row_idx = int(num_str) - 1
    except ValueError:
        raise ValueError(f"Invalid keyword format. Expected format like 'x_1', got: '{keyword}'")

    if prefix == 'all':
        return data_dict[f'first_moment_nrmse_{split}_average'][row_idx]
    elif prefix in ['x', 'y', 'z']:
        col_map = {'x': 0, 'y': 1, 'z': 2}
        return data_dict[f'first_moment_nrmse_{split}_list'][row_idx, col_map[prefix]]
    else:
        raise ValueError(f"Unknown prefix '{prefix}'. Must be 'x', 'y', 'z', or 'all'.")


def extract_score(task, result, target, split):
    """One run's score for `task`.

    Returns None for a missing or NaN mixing-capacity score, or when `result` itself is
    None (the run failed outright). The Lorenz path is different: it goes through
    `get_nrmse_element`'s raw dict subscript, so a genuinely missing key raises `KeyError`
    rather than returning None. That is inherited behaviour, unchanged here.
    """
    if result is None:
        return None
    if task.kind == "mixing":
        value = result.get("first_moment_2")
    else:
        if split is None:
            raise ValueError(
                f"extract_score: a '{task.kind}' task needs split='val' or split='test' "
                "to read a Lorenz NRMSE key; got split=None"
            )
        value = get_nrmse_element(result, target, split)
    if value is None or np.isnan(value):
        return None
    return float(value)


def run_study(task_name, n, d, encoding_mode, outdir, target=None,
              n_shots=shots.EXACT) -> str:
    """Searches hyperparameters, then evaluates the winner on unseen seeds and data.

    Stage one is the hyperparameter search. For Lorenz its objective is the validation
    block, not the test block, so the search does not select on the data the reported
    number is read from. Stage two runs study.best_params once more over EVAL_SEEDS --
    realizations no trial saw -- and scores them on the test block. That second number, with
    its standard deviation across those realizations, is what the figures report.

    The measurement RNG is seeded from the realization seed alone, so all N_TRIALS trials
    at a given tuning seed share one measurement stream. That is common random numbers: it
    keeps trial selection from being driven by resampled noise rather than by the
    hyperparameters. EVAL_SEEDS is disjoint from TUNE_SEEDS, so the reported holdout still
    rests on measurement noise no trial saw.
    """
    task = STUDY_TASKS[task_name]
    fixed = dict(task.fixed_params, n=n, d=d, encoding_mode=encoding_mode)
    # Injected only for the quantum tasks. The ESN is a classical reservoir with no
    # measurement ensemble, and its task functions would reject the key.
    if optional_axis_names(task_name):
        fixed['n_shots'] = n_shots
    worst = float('-inf') if task.direction == "maximize" else float('inf')

    def objective(trial):
        params = dict(fixed)
        for key, (low, high, scale) in task.search_space.items():
            params[key] = trial.suggest_float(key, low, high, log=(scale == "log"))

        results = evaluate_seeds(task.run_func, params, TUNE_SEEDS)
        scores = [
            extract_score(task, results.get(seed), target, task.objective_split)
            for seed in TUNE_SEEDS
        ]
        if any(score is None for score in scores):
            return worst

        # ddof=0: this is the search's own bookkeeping, not the reported band. The
        # reported SD comes from holdout_moments, with ddof=1.
        trial.set_user_attr("std", float(np.std(scores)))

        if task.kind == "lorenz":
            # Free: these same runs already scored the test block, so the winner's test
            # score on the tuning seeds costs nothing and separates seed selection from
            # trial selection.
            test_scores = [
                extract_score(task, results.get(seed), target, "test")
                for seed in TUNE_SEEDS
            ]
            if all(score is not None for score in test_scores):
                trial.set_user_attr("test_mean", float(np.mean(test_scores)))
                trial.set_user_attr("test_std", float(np.std(test_scores)))

        return float(np.mean(scores))

    sampler_seed = study_sampler_seed(task_name, n, d, encoding_mode, target, n_shots)
    study = optuna.create_study(
        direction=task.direction,
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, n_startup_trials=startup_trial_count()
        ),
    )
    study.optimize(objective, n_trials=trial_count())

    holdout_split = "test" if task.kind == "lorenz" else None
    holdout_results = evaluate_seeds(
        task.run_func, dict(fixed, **study.best_params), EVAL_SEEDS
    )
    holdout_scores = [
        extract_score(task, holdout_results.get(seed), target, holdout_split)
        for seed in EVAL_SEEDS
    ]

    payload = build_payload(
        task, n, d, encoding_mode, target, sampler_seed, study,
        holdout_scores, EVAL_SEEDS, n_shots,
    )
    return save_results(
        payload, task.model_name, n, d, outdir,
        encoding_mode=encoding_mode, target=target, n_shots=n_shots,
    )


def holdout_moments(scores) -> dict:
    """Mean, sample SD and SEM over the holdout scores that survived.

    A failed seed is skipped, not fatal: one bad seed must not turn a whole grid point into
    -inf, which is how a run comes to hold unusable rows. `n_ok` records how many of the
    twenty realizations the moments rest on, so the loss is visible in the data.
    """
    finite = [score for score in scores if score is not None and np.isfinite(score)]
    if not finite:
        return {"mean": None, "sd": None, "sem": None, "n_ok": 0}
    if len(finite) == 1:
        return {"mean": float(finite[0]), "sd": None, "sem": None, "n_ok": 1}

    sd = float(np.std(finite, ddof=1))
    return {
        "mean": float(np.mean(finite)),
        "sd": sd,
        "sem": sd / float(np.sqrt(len(finite))),
        "n_ok": len(finite),
    }


def signed_optimism(search_mean, holdout_mean, direction):
    """How much better the search looked than the holdout. Positive means inflated."""
    if search_mean is None or holdout_mean is None:
        return None
    if not (np.isfinite(search_mean) and np.isfinite(holdout_mean)):
        return None
    gap = float(search_mean) - float(holdout_mean)
    return gap if direction == "maximize" else -gap


def trial_records(study) -> list:
    """One JSON-serialisable record per trial, in trial order.

    The whole history is kept, not just the winner, so the gap between the best trial and
    the rest of the search can be quantified rather than asserted.
    """
    return [
        {
            "number": trial.number,
            "params": trial.params,
            "mean": trial.value,
            "std": trial.user_attrs.get("std"),
            "test_mean": trial.user_attrs.get("test_mean"),
            "test_std": trial.user_attrs.get("test_std"),
            "state": trial.state.name,
        }
        for trial in study.trials
    ]


def build_payload(task, n, d, encoding_mode, target, sampler_seed, study,
                  holdout_scores, holdout_seeds, n_shots=shots.EXACT) -> dict:
    """Assembles one study's result.

    Deliberately writes no `best_score_mean` / `best_score_std`: those keys named the
    best-of-search value under an earlier protocol, so a consumer still reading them fails
    loudly rather than silently plotting the search optimum in place of the holdout.
    """
    best = study.best_trial
    moments = holdout_moments(holdout_scores)
    scores = [
        None if score is None or not np.isfinite(score) else float(score)
        for score in holdout_scores
    ]
    failed = [seed for seed, score in zip(holdout_seeds, scores, strict=True) if score is None]

    if task.kind == "lorenz":
        # task.fixed_params, not the module constant: the recorded split must be the one
        # the run actually used, not a value that could drift away from it.
        washout, train_length, val_length, test_length, _ = lorenz_split(
            task.fixed_params
        )
        split = {
            "washout": washout, "train_length": train_length,
            "val_length": val_length, "test_length": test_length,
        }
    else:
        split = None

    if task.model_name == "esn":
        # A classical reservoir has no measurement ensemble; writing n_shots: null for it
        # would suggest it merely happens to be in the exact limit.
        ensemble = {}
    else:
        ensemble = {
            # None, not float('inf'): json.dump renders infinity as the non-standard
            # `Infinity` token, which a non-Python reader of the deposit cannot parse.
            # average_runs_optuna maps None back to the exact limit on read.
            "n_shots": None if shots.is_exact(n_shots) else int(n_shots),
            # One homodyne setting for the CV readout, three non-commuting Pauli settings
            # for the DV one -- so at equal n_shots the DV study costs three times the
            # measurement time.
            "n_settings": 1 if task.model_name == "qrc_gaussian" else 3,
        }

    return {
        "best_params": best.params,
        "sampler_seed": sampler_seed,
        "n": n,
        "d": d,
        "encoding_mode": encoding_mode,
        "target": target,
        **ensemble,
        "search_score_mean": best.value,
        "search_score_std": best.user_attrs.get("std"),
        "search_score_test_mean": best.user_attrs.get("test_mean"),
        "search_score_test_std": best.user_attrs.get("test_std"),
        "holdout_score_mean": moments["mean"],
        "holdout_score_sd": moments["sd"],
        "holdout_score_sem": moments["sem"],
        "holdout_scores": scores,
        "holdout_seeds": list(holdout_seeds),
        "n_holdout_ok": moments["n_ok"],
        "holdout_failed_seeds": failed,
        "search_optimism": signed_optimism(best.value, moments["mean"], task.direction),
        "trials": trial_records(study),
        "protocol": {
            "n_trials": trial_count(),
            "n_startup_trials": startup_trial_count(),
            "tune_seeds": list(TUNE_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "direction": task.direction,
            "objective_split": task.objective_split,
            "split": split,
        },
    }


def save_results(payload, model_name, n, d, outdir, encoding_mode=None, target=None,
                 n_shots=shots.EXACT) -> str:
    """Writes one study's payload and returns the path.

    A re-run lands beside its predecessor under a new run ID rather than needing a new
    naming scheme. The shot field is appended only when the ensemble is finite, so an
    exact-limit study's filename does not depend on the ensemble machinery existing.
    """
    os.makedirs(outdir, exist_ok=True)
    suffix = "" if shots.is_exact(n_shots) else f"_nshots{shots.label(n_shots)}"
    json_path = os.path.join(
        outdir,
        f"best_params_{model_name}_n{n}_d{d}_em{encoding_mode}_t{target}{suffix}.json",
    )
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)
    return json_path


if __name__ == "__main__":
    args_full = sys.argv[1:]
    name = args_full[-1]
    args = args_full[:-1]

    task = STUDY_TASKS.get(name)
    if task is None:
        print(f"Unknown task: {name}")
        sys.exit(1)

    # The argv order is the grid's axis order, from utils.gridconfig.axis_names_for:
    # (n, d, encoding_mode) for a mixing task, (n, d, target, encoding_mode) for a Lorenz
    # one, each optionally followed by n_shots, then the output directory.
    outdir = args[-1]
    values = args[:-1]
    axes = axis_names_for(name, len(values))
    parameters = dict(zip(axes, values, strict=True))

    run_study(
        name,
        int(parameters["n"]),
        int(parameters["d"]),
        str(parameters["encoding_mode"]),
        str(outdir),
        target=str(parameters["target"]) if "target" in parameters else None,
        n_shots=float(parameters.get("n_shots", shots.EXACT)),
    )
