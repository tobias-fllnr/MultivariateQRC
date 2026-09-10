"""Finite measurement ensembles for the two quantum readouts.

Every expectation value a real quantum reservoir reports is estimated from a finite number
of shots. This module holds the two samplers that turn an exact readout into an M-shot
estimate of it -- one per reservoir family -- plus the small amount of shared bookkeeping
they need.

Both samplers are the *exact* sampling distribution of the readout they serve, not an
additive-noise surrogate. That is possible because of the measurement protocol assumed
throughout: the reservoir is re-prepared and re-driven for every measurement, so
measurement never perturbs the trajectory the next time step evolves from. Mujal et al.,
npj Quantum Inf. 9, 16 (2023) call this the restarting protocol and note it is "usually
implicit in QRC proposals so far"; Hahto & Nokkala, New J. Phys. 27, 094510 (2025) rely on
their platform's measurement back-action averaging out. Two consequences follow, and they
are what make the model this simple: the state trajectory is identical to the noiseless
one, and the readout noise is independent across time steps.

`n_shots` counts shots **per commuting measurement setting**. The CV homodyne readout is
one setting; the DV local Pauli readout is three (X, Y, Z), which is the factor 3 in
Mujal et al.'s Eqs. (G1)-(G3). At equal `n_shots`, the DV readout therefore costs three
times the experimental time -- callers record `n_settings` so a figure can be replotted
against total budget without re-running anything.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.stats import wishart

# The infinite-shot limit: exact expectation values, and the default everywhere. Chosen as
# a float rather than a sentinel object so a grid can write it as YAML `.inf` and the axis
# can stay a plain float axis.
EXACT = float("inf")

# Separates the measurement stream from the reservoirs' own disorder. The reservoir classes
# seed their coupling matrices, input weights and fill noise with the legacy global
# np.random.seed(seed), which drives a different bit generator entirely; this constant makes
# the separation legible rather than incidental, so a run's disorder realization is provably
# the same whether or not shots are finite.
SHOT_STREAM = 0x53484F54  # "SHOT"


def is_exact(n_shots) -> bool:
    """True when `n_shots` asks for exact expectation values.

    Positive infinity specifically, not "non-finite": `not np.isfinite(x)` would classify
    NaN as the exact limit and let validate_n_shots wave it through.
    """
    return bool(np.isposinf(n_shots))


def label(n_shots) -> str:
    """How a shot count is rendered in a filename or a hash key."""
    return "inf" if is_exact(n_shots) else str(int(n_shots))


def validate_n_shots(n_shots, *, minimum: int, what: str) -> None:
    """Raises unless `n_shots` is the exact limit or a finite integral count >= `minimum`.

    `minimum` is 1 for a readout whose estimator works for any ensemble, and the number of
    modes for the CV covariance readout, whose sample covariance is singular below that.
    `what` names what the minimum counts, so the error says which constraint was violated.
    """
    if is_exact(n_shots):
        return
    value = float(n_shots)
    if not np.isfinite(value):
        raise ValueError(
            f"n_shots must be a finite count or float('inf'), got {n_shots!r}"
        )
    if value != int(value):
        raise ValueError(f"n_shots must be integral, got {n_shots!r}")
    if int(value) < minimum:
        raise ValueError(
            f"n_shots={int(value)} is below the minimum of {minimum} set by {what}; "
            "a smaller ensemble cannot produce this readout"
        )


def shot_rng(seed: int) -> np.random.Generator:
    """The measurement RNG for one run, seeded from that run's realization seed."""
    return np.random.default_rng([SHOT_STREAM, int(seed)])


def sample_q_covariance(sigma_q: np.ndarray, n_shots, rng) -> np.ndarray:
    """One M-shot homodyne estimate of the q-quadrature covariance matrix.

    Sigma_tilde = (1/M) W(sigma_q, M) -- Hahto & Nokkala Eq. (4). This is exact rather than
    approximate for three reasons, all of which hold for our Gaussian reservoir:

    - A homodyne detector reading the q quadrature of every mode reads commuting
      observables, so one shot yields one joint sample q ~ N(mu_q, sigma_q).
    - The Gaussian state's first moments are identically zero at every step (the mean
      propagates as mu <- M mu from a zero start), so no mean is estimated and the M-shot
      MLE is the uncentred estimator -- Wishart with M degrees of freedom, not M - 1.
    - The readout is exactly the q-block, so the sampled matrix is the readout itself.

    The draw is scaled after the fact rather than by passing scale=sigma_q/M, which would
    lose precision on small variances.
    """
    if is_exact(n_shots):
        return sigma_q

    modes = sigma_q.shape[0]
    validate_n_shots(n_shots, minimum=modes, what="the number of modes")
    degrees = int(n_shots)

    drawn = wishart.rvs(df=degrees, scale=sigma_q, random_state=rng)
    # scipy returns a bare scalar for a 1x1 scale matrix.
    return np.asarray(drawn, dtype=float).reshape(modes, modes) / degrees


# The three commuting measurement settings of the local Pauli readout, in the column order
# the reservoir's observable list uses: index 3 * qubit + setting.
PAULI_SETTINGS = ("x", "y", "z")


def pauli_basis_rotations(n_qubits: int) -> list:
    """One unitary per setting, mapping that setting's readout onto the computational basis.

    U_B satisfies <sigma_B^i>_rho = <sigma_z^i>_{U_B rho U_B^dagger}, so the outcome
    distribution of setting B is the diagonal of the rotated state. sigma_x = H sigma_z H
    gives U_X = (x)H, and sigma_y = (SH) sigma_z (SH)^dagger gives U_Y = (x)H S^dagger.
    The z setting needs no rotation, and returning None for it skips two matrix products
    per time step.

    The correctness of these rotations is not taken on trust: a test asserts that
    contracting the resulting distributions with `outcome_signs` reproduces
    qt.expect(observables, rho) for random states, which pins the rotations and qutip's
    tensor ordering together.
    """
    hadamard = qt.Qobj(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0))
    s_dagger = qt.Qobj(np.array([[1.0, 0.0], [0.0, -1.0j]]))
    singles = [hadamard, hadamard * s_dagger, None]
    return [
        None if single is None else qt.tensor([single] * n_qubits)
        for single in singles
    ]


def outcome_signs(n_qubits: int) -> np.ndarray:
    """The +-1 eigenvalue of each qubit in each computational-basis outcome pattern.

    Shape (2**n, n). Row `s` is one measurement outcome; column `i` is qubit `i`'s
    eigenvalue in it, +1 for |0> and -1 for |1>. Qubit 0 is the most significant bit,
    which is qutip's tensor ordering.
    """
    patterns = np.arange(2 ** n_qubits)
    return np.array([
        1 - 2 * ((patterns >> (n_qubits - 1 - qubit)) & 1)
        for qubit in range(n_qubits)
    ]).T


def as_distribution(probabilities: np.ndarray) -> np.ndarray:
    """Clips at zero and renormalises one outcome distribution.

    mesolve can leave a diagonal entry of the density matrix negative at the 1e-16 level,
    and np.random.Generator.multinomial rejects such a vector outright. The resulting bias
    is O(1e-16), many orders below the shot noise it sits inside -- it is a solver
    artefact being absorbed, not a modelling choice.

    Applied both where the distribution is built and where it is sampled: the sampler is
    the function that hands the array to numpy, so it owns that precondition rather than
    trusting every caller to have cleaned the vector first.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    return clipped / clipped.sum()


def basis_probabilities(rho, rotations) -> np.ndarray:
    """The outcome distribution of each measurement setting, shape (len(rotations), 2**n)."""
    distributions = []
    for rotation in rotations:
        rotated = rho if rotation is None else rotation * rho * rotation.dag()
        distributions.append(as_distribution(np.real(np.diag(rotated.full()))))
    return np.array(distributions)


def sample_local_pauli(probabilities: np.ndarray, signs: np.ndarray, n_shots, rng) -> np.ndarray:
    """One M-shot-per-setting estimate of the local Pauli readout.

    Returns 3 * n estimates indexed `3 * qubit + setting`, the reservoir's own column
    order. Each setting is sampled as a single multinomial draw over its 2**n outcome
    patterns, which is the exact projective-measurement statistics: the marginal of each
    observable is the binomial model of Mujal et al. Eqs. (B9)-(B10), with variance
    (1 - <O>^2) / M, and the joint draw additionally carries the within-setting
    cross-qubit noise correlation that a per-observable binomial drops. That correlation
    matters because the readout is a linear combination of these features.

    2**n outcomes is cheap at the reservoir sizes here (64 for n = 6) and negligible next
    to one mesolve step.
    """
    n_settings = probabilities.shape[0]
    n_qubits = signs.shape[1]

    if is_exact(n_shots):
        means = probabilities @ signs
    else:
        validate_n_shots(n_shots, minimum=1, what="one shot per setting")
        draws = int(n_shots)
        counts = np.array([
            rng.multinomial(draws, as_distribution(probabilities[setting]))
            for setting in range(n_settings)
        ])
        means = (counts @ signs) / draws

    estimates = np.empty(n_settings * n_qubits)
    for setting in range(n_settings):
        for qubit in range(n_qubits):
            estimates[n_settings * qubit + setting] = means[setting, qubit]
    return estimates
