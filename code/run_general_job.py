import pickle
import sys
import os
import numpy as np
from utils.data import generate_random_sequence
from utils.prediction import Prediction
from utils.qrc_spin import SpinQRC
from utils.ipc import IPC
from utils.qrc_gaussian import GaussianQRC
from utils.lorenz63 import Lorenz63Generator
from utils.gridconfig import axis_names_for, expected_axis_names
from utils import shots
from utils.esn import ESN


# The Lorenz-63 sequence layout, shared by all three Lorenz task functions. An optional
# validation block is carved out of the evaluation block rather than added to the sequence,
# so the trajectory, the training block and the NRMSE denominator do not depend on it.
LORENZ_WASHOUT = 1000
LORENZ_TRAIN_LENGTH = 6000
LORENZ_EVAL_LENGTH = 4000


def lorenz_split(parameters: dict) -> tuple[int, int, int, int, int]:
    """One run's Lorenz layout: (washout, train, val, test, total).

    `val_length` is an optional parameter, not a grid axis, and defaults to 0 -- the plain
    1000/6000/4000 two-way layout, keys included. `total` is the same for every value of it.
    """
    val_length = int(parameters.get('val_length', 0))
    if not 0 <= val_length < LORENZ_EVAL_LENGTH:
        raise ValueError(
            f"val_length must be in [0, {LORENZ_EVAL_LENGTH}), got {val_length}"
        )
    return (
        LORENZ_WASHOUT,
        LORENZ_TRAIN_LENGTH,
        val_length,
        LORENZ_EVAL_LENGTH - val_length,
        LORENZ_WASHOUT + LORENZ_TRAIN_LENGTH + LORENZ_EVAL_LENGTH,
    )


def shot_keys(parameters: dict, reservoir) -> dict:
    """The measurement-ensemble keys a result carries, if any.

    Empty unless the grid declared `n_shots`. That condition covers `n_settings` too, and
    must: a grid run in the exact limit does not declare the axis, so an unconditional
    `n_settings` would add a key its per-seed pickles do not have, and re-averaging such a
    grid would emit an `n_settings_mean`/`_std` pair its averaged pickle lacks. The same
    pattern `val_length` uses -- an optional parameter whose keys exist only when it is set.
    """
    if 'n_shots' not in parameters:
        return {}
    return {'n_shots': parameters['n_shots'], 'n_settings': reservoir.n_settings}


def run_mixing_capacity_qrc_tilted_tfim(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = int(parameters.get('washout', 1000))
    train_length = int(parameters.get('train_length', 10000))
    total_length = washout + train_length
    data = generate_random_sequence(length=total_length, dimension=d, seed=seed)
    reservoir = SpinQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, model="TiltedTFIM", encoding_method=encoding_mode, observables="local_and_twoqubit", seed=seed, n_shots=parameters.get('n_shots', shots.EXACT))
    measurements, negativity = reservoir.run(data, return_negativity=True, return_coherence=False)
    measurements_local = measurements[:, :3*n]
    max_degree = min(d, 2)
    ipc_local = IPC(values=measurements_local, targets=data, washout=washout, train_length=train_length)
    mixing_capacity_breakdown_local, capacity_mixing_local = ipc_local.ipc(max_delay=50, max_degree=max_degree, return_ipc=False, return_capacity_mixing=True)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in capacity_mixing_local.items()})
    result.update({f"first_moment_breakdown_{k}": v for k, v in mixing_capacity_breakdown_local.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result.update(shot_keys(parameters, reservoir))
    return result


def run_mixing_capacity_qrc_gaussian(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = int(parameters.get('washout', 1000))
    train_length = int(parameters.get('train_length', 10000))
    total_length = washout + train_length
    data = generate_random_sequence(length=total_length, dimension=d, seed=seed)
    reservoir = GaussianQRC(
        n=n,
        encoding_strength=encoding_strength,
        coupling_strength=coupling_strength,
        gamma=gamma,
        dt=dt,
        cov_measurements="q_only",
        encoding_mode=encoding_mode,
        seed=seed,
        return_fourth_moments=False,
        n_shots=parameters.get('n_shots', shots.EXACT),
    )
    means, flat_covs, squeezing = reservoir.run(
        data, return_negativity=False, return_purity=False, return_squeezing=True
    )
    max_degree = min(d, 2)
    ipc_covs = IPC(values=flat_covs, targets=data, washout=washout, train_length=train_length)
    mixing_capacity_breakdown_covs, capacity_mixing_covs = ipc_covs.ipc(max_delay=50, max_degree=max_degree, return_ipc=False, return_capacity_mixing=True)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in capacity_mixing_covs.items()})
    result.update({f"first_moment_breakdown_{k}": v for k, v in mixing_capacity_breakdown_covs.items()})
    result['squeezing'] = np.mean(squeezing[washout:washout+train_length])
    result.update(shot_keys(parameters, reservoir))
    return result


def run_lorenz63_qrc_tilted_tfim(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout, train_length, val_length, test_length, total_length = lorenz_split(parameters)
    generator = Lorenz63Generator(length=total_length, dt_int=0.005, dt_data=0.092, t_init_cutoff=20.0, seed=42)
    data = generator.generate()
    min_max_normalized_data = normalize_data_min_max(data.copy())
    data_in = min_max_normalized_data[:, :d]
    reservoir = SpinQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, model="TiltedTFIM", encoding_method=encoding_mode, observables="local_and_twoqubit", seed=seed, n_shots=parameters.get('n_shots', shots.EXACT))
    measurements, negativity = reservoir.run(data_in, return_negativity=True, return_coherence=False)
    measurements_local = measurements[:, :3*n]
    prediction_local = Prediction(observations=measurements_local, data=data, washout=washout, train_length=train_length, test_length=test_length, val_length=val_length, model="linear")
    pred_results_local = prediction_local.prediction_multi_step(max_steps=10)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in pred_results_local.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result.update(shot_keys(parameters, reservoir))
    return result


def run_lorenz63_qrc_gaussian(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout, train_length, val_length, test_length, total_length = lorenz_split(parameters)
    generator = Lorenz63Generator(length=total_length, dt_int=0.005, dt_data=0.092, t_init_cutoff=20.0, seed=42)
    data = generator.generate()
    min_max_normalized_data = normalize_data_min_max(data.copy())
    data_in = min_max_normalized_data[:, :d]
    reservoir = GaussianQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, encoding_mode=encoding_mode, cov_measurements="q_only", return_fourth_moments=False, seed=seed, n_shots=parameters.get('n_shots', shots.EXACT))
    means, flat_covs, squeezing = reservoir.run(
        data_in, return_negativity=False, return_purity=False, return_squeezing=True
    )
    prediction_covs = Prediction(observations=flat_covs, data=data, washout=washout, train_length=train_length, test_length=test_length, val_length=val_length, model="linear")
    pred_results_covs = prediction_covs.prediction_multi_step(max_steps=10)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in pred_results_covs.items()})
    result['squeezing'] = np.mean(squeezing[washout:washout+train_length])
    result.update(shot_keys(parameters, reservoir))
    return result


def run_mixing_capacity_esn(parameters: dict) -> dict:
    n, d, encoding_mode, input_scaling, spectral_radius, leak_rate, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'input_scaling', 'spectral_radius', 'leak_rate', 'seed']
    )
    # A tuned hyperparameter, not a grid axis -- the same treatment val_length gets. It
    # defaults to a U(-1, 1) bias, so a grid that does not set it produces exactly the
    # argv, the filename and the numbers the plain ESN_AXES signature produces.
    bias_scale = float(parameters.get('bias_scale', 1.0))

    washout = 1000
    train_length = 10000
    total_length = washout + train_length
    data = generate_random_sequence(length=total_length, dimension=d, seed=seed)
    reservoir = ESN(n=n, d=d, encoding_mode=encoding_mode, input_scaling=input_scaling, spectral_radius=spectral_radius, leak_rate=leak_rate, seed=seed, bias_scale=bias_scale)
    states = reservoir.run(data)
    max_degree = min(d, 2)
    ipc_states = IPC(values=states, targets=data, washout=washout, train_length=train_length)
    mixing_capacity_breakdown, capacity_mixing = ipc_states.ipc(max_delay=50, max_degree=max_degree, return_ipc=False, return_capacity_mixing=True)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'input_scaling', 'spectral_radius', 'leak_rate', 'seed']}
    result['bias_scale'] = bias_scale
    result.update({f"first_moment_{k}": v for k, v in capacity_mixing.items()})
    result.update({f"first_moment_breakdown_{k}": v for k, v in mixing_capacity_breakdown.items()})
    result['n_nodes'] = reservoir.n_nodes
    return result


def run_lorenz63_esn(parameters: dict) -> dict:
    n, d, encoding_mode, input_scaling, spectral_radius, leak_rate, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'input_scaling', 'spectral_radius', 'leak_rate', 'seed']
    )
    # A tuned hyperparameter, not a grid axis -- the same treatment val_length gets. It
    # defaults to a U(-1, 1) bias, so a grid that does not set it produces exactly the
    # argv, the filename and the numbers the plain ESN_AXES signature produces.
    bias_scale = float(parameters.get('bias_scale', 1.0))

    washout, train_length, val_length, test_length, total_length = lorenz_split(parameters)
    generator = Lorenz63Generator(length=total_length, dt_int=0.005, dt_data=0.092, t_init_cutoff=20.0, seed=42)
    data = generator.generate()
    min_max_normalized_data = normalize_data_min_max(data.copy())
    data_in = min_max_normalized_data[:, :d]
    reservoir = ESN(n=n, d=d, encoding_mode=encoding_mode, input_scaling=input_scaling, spectral_radius=spectral_radius, leak_rate=leak_rate, seed=seed, bias_scale=bias_scale)
    states = reservoir.run(data_in)
    prediction_states = Prediction(observations=states, data=data, washout=washout, train_length=train_length, test_length=test_length, val_length=val_length, model="linear")
    pred_results = prediction_states.prediction_multi_step(max_steps=10)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'input_scaling', 'spectral_radius', 'leak_rate', 'seed']}
    result['bias_scale'] = bias_scale
    result.update({f"first_moment_{k}": v for k, v in pred_results.items()})
    result['n_nodes'] = reservoir.n_nodes
    return result


def normalize_data_min_max(data: np.ndarray) -> np.ndarray:
    for dim_i in range(data.shape[1]):
        col = data[:, dim_i]
        d_min, d_max = col.min(), col.max()
        if d_max > d_min:
            data[:, dim_i] = (col - d_min) / (d_max - d_min)
    return data


# How each axis's argv string is parsed. Keyed by axis name rather than by position, so a
# task with its own signature needs no new parsing code -- only an entry here for any axis
# it introduces.
AXIS_CASTS = {
    "n": int,
    "d": int,
    "encoding_mode": str,
    "dt": float,
    "encoding_strength": float,
    "coupling_strength": float,
    "gamma": float,
    # A float, not an int, because `.inf` -- the infinite-shot limit -- must be a legal
    # value on this axis.
    "n_shots": float,
    "input_scaling": float,
    "spectral_radius": float,
    "leak_rate": float,
    "seed": int,
}

# Filename prefixes for the axes whose prefix is not simply their own name. Every other
# axis contributes "<name><value>".
FILENAME_ABBREV = {
    "encoding_mode": "em",
    "encoding_strength": "es",
    "coupling_strength": "cs",
    "input_scaling": "is",
    "spectral_radius": "sr",
    "leak_rate": "lr",
    # Required, not cosmetic: the default prefix is the axis name, and "n_shots1000000"
    # contains an underscore, which every filename parser in the project treats as a field
    # separator.
    "n_shots": "nshots",
}


def parse_job_args(argv: list[str]) -> tuple[str, str, dict, tuple[str, ...], list[str]]:
    """Splits a job's argument list into task name, output directory and parameters.

    `argv` is sys.argv[1:]: the axis values in the order the grid declares them, then the
    output directory, then the task name. The axis names come from
    utils.gridconfig.axis_names_for, the same table prepare_jobs.py validates grids
    against, so the argv order a grid produces and the argv order this script reads cannot
    drift apart. That function also rejects a wrong arity, naming the axes the task reads,
    which is why no separate length check is needed here.

    Returns (name, outdir, params, axes, raw_values). raw_values is kept unparsed because
    the result filename is built from the argv strings, not from the cast values.
    """
    name = argv[-1]
    outdir = argv[-2]
    raw_values = list(argv[:-2])
    axes = axis_names_for(name, len(raw_values))

    params = {axis: AXIS_CASTS[axis](raw) for axis, raw in zip(axes, raw_values)}
    return name, outdir, params, axes, raw_values


def result_filename(axes: tuple[str, ...], raw_values: list[str]) -> str:
    """The per-seed result filename for one parameter point.

    Built from the raw argv strings so the rendering the grid produced is preserved
    exactly -- "dt1.0" stays "dt1.0", "cs1e-05" stays "cs1e-05". Derived from the task's
    signature rather than hard-coded, so a task whose signature gains an axis cannot write
    two distinct grid points to the same path.
    """
    if len(axes) != len(raw_values):
        raise ValueError(
            f"{len(axes)} axes but {len(raw_values)} raw values: {axes} vs {raw_values}"
        )
    parts = [
        f"{FILENAME_ABBREV.get(axis, axis)}{raw}"
        for axis, raw in zip(axes, raw_values)
    ]
    return "result_" + "_".join(parts) + ".pkl"


# Maps task names to their run functions
TASK_FUNCTIONS = {
    "mixing_capacity_qrc_tilted_tfim": run_mixing_capacity_qrc_tilted_tfim,
    "mixing_capacity_qrc_gaussian": run_mixing_capacity_qrc_gaussian,
    "lorenz63_qrc_tilted_tfim": run_lorenz63_qrc_tilted_tfim,
    "lorenz63_qrc_gaussian": run_lorenz63_qrc_gaussian,
    "mixing_capacity_esn": run_mixing_capacity_esn,
    "lorenz63_esn": run_lorenz63_esn,
}


if __name__ == "__main__":
    name, outdir, params, axes, raw_values = parse_job_args(sys.argv[1:])

    run_func = TASK_FUNCTIONS.get(name)
    if run_func is None:
        print(f"Unknown task: {name}")
        sys.exit(1)

    result = run_func(params)

    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, result_filename(axes, raw_values))
    with open(filename, 'wb') as f:
        pickle.dump(result, f)