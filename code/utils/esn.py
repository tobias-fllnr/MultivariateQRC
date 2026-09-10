"""A classical echo state network matched to the DV reservoir's readout dimension.

The reservoir carries N = 3n nodes, grouped into n triplets, one triplet per site of the
n-qubit DV reservoir. N is therefore exactly the DV reservoir's readout dimension, since
run_general_job.py reads its 3n local Pauli expectation values.

The input drive is computed over the n *sites* and broadcast to each site's triplet, so the
encoding map is numerically SpinQRC._generate_field_strengths with its constant field
factored out into the bias. Driving sites rather than nodes is what keeps the driven
fraction of the reservoir at d/n, as in the DV reservoir, instead of d/3n.

Three of the four free hyperparameters map one-to-one onto the quantum reservoirs':
input_scaling onto encoding_strength, spectral_radius onto coupling_strength (both scale a
dense random coupling matrix), and leak_rate onto gamma (both set how fast the state
forgets).

The fourth, bias_scale, has no quantum counterpart and is tuned because it has to be. tanh
is an odd function, so a reservoir with zero bias is an exactly odd functional of its input
and cannot represent any even-degree target at all -- its degree-2 mixing capacity is
identically zero by symmetry, not merely small. The bias is the model's only source of the
even nonlinearity that turns a mixture of two streams into their product, and how much of
it is wanted differs sharply by task: mixing capacity peaks near bias_scale = 8 while
Lorenz-63 prediction, which rides on the linear response that a large bias saturates,
prefers 0.25 to 1. No single fixed value serves both, so it is searched rather than fixed.
The default of 1.0 reproduces the original U(-1, 1) bias exactly.
"""

import numpy as np


class ESN:
    """A leaky-integrator echo state network with 3n nodes over n input sites."""

    VALID_ENCODING_MODES = ("one_to_one", "fill", "dense")
    NODES_PER_SITE = 3

    def __init__(self, n: int, d: int, encoding_mode: str, input_scaling: float,
                 spectral_radius: float, leak_rate: float, seed: int = 42,
                 bias_scale: float = 1.0):
        """Initializes the reservoir and draws its fixed random weights.

        Args:
            n: Number of input sites. The reservoir has 3n nodes.
            d: Input dimension.
            encoding_mode: One of 'one_to_one', 'fill', 'dense'.
            input_scaling: Scales the encoded input. Mirrors encoding_strength.
            spectral_radius: Scales the unit-spectral-radius recurrent matrix. Mirrors
                coupling_strength.
            leak_rate: Leaking rate in (0, 1]. Mirrors gamma.
            seed: Random seed for reproducibility.
            bias_scale: Half-width of the node bias, drawn b_i ~ U(-bias_scale,
                bias_scale). Defaults to 1.0, the plain U(-1, 1) bias.
        """
        if n <= 0:
            raise ValueError("number of sites 'n' must be positive")
        if d <= 0:
            raise ValueError("input dimension 'd' must be positive")
        if encoding_mode not in self.VALID_ENCODING_MODES:
            raise ValueError(
                f"encoding_mode must be one of {self.VALID_ENCODING_MODES}; "
                f"got {encoding_mode!r}"
            )
        if encoding_mode in ("one_to_one", "fill") and d > n:
            raise ValueError(
                f"input dimension ({d}) cannot exceed the number of sites ({n}) in "
                f"{encoding_mode} mode"
            )
        if input_scaling < 0:
            raise ValueError("input_scaling must be non-negative")
        if spectral_radius < 0:
            raise ValueError("spectral_radius must be non-negative")
        if not 0 < leak_rate <= 1:
            raise ValueError(f"leak_rate must lie in (0, 1]; got {leak_rate}")
        if bias_scale <= 0:
            raise ValueError(f"bias_scale must be positive; got {bias_scale}")

        self.n = n
        self.d = d
        self.encoding_mode = encoding_mode
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.seed = seed
        self.bias_scale = bias_scale
        self.n_nodes = self.NODES_PER_SITE * n

        # One stream per component, so the recurrent matrix's size cannot shift the
        # encoding draws that would follow it under a single sequential generator.
        encoding_rng = np.random.default_rng([seed, 0])
        reservoir_rng = np.random.default_rng([seed, 1])
        bias_rng = np.random.default_rng([seed, 2])

        self.W_in = self._build_input_matrix(encoding_rng)
        self.noise_fill = (
            encoding_rng.uniform(0.0, 1.0, n) if encoding_mode == "fill" else None
        )
        self.W_hat = self._build_reservoir_matrix(reservoir_rng)
        self.bias = bias_rng.uniform(-bias_scale, bias_scale, self.n_nodes)

    def run(self, input_sequence: np.ndarray,
            initial_state: np.ndarray | None = None) -> np.ndarray:
        """Drives the reservoir with an input sequence and returns the state history.

        Args:
            input_sequence: Array of shape (steps, d), values in [0, 1].
            initial_state: Optional starting state of shape (3n,). Defaults to zeros,
                mirroring the quantum reservoirs' fixed initial state.

        Returns:
            Array of shape (steps, 3n): the node states after each input step.
        """
        input_sequence = np.asarray(input_sequence, dtype=float)
        if input_sequence.ndim == 1:
            input_sequence = input_sequence[:, np.newaxis]
        if input_sequence.shape[1] != self.d:
            raise ValueError(
                f"input dimension mismatch: reservoir built for d={self.d}, got "
                f"{input_sequence.shape[1]}"
            )

        state = (
            np.zeros(self.n_nodes) if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        if state.shape != (self.n_nodes,):
            raise ValueError(
                f"initial_state must have shape ({self.n_nodes},); got {state.shape}"
            )

        states = np.zeros((input_sequence.shape[0], self.n_nodes))
        for t in range(input_sequence.shape[0]):
            drive = np.repeat(self.site_drive(input_sequence[t]), self.NODES_PER_SITE)
            pre_activation = (
                self.spectral_radius * (self.W_hat @ state) + drive + self.bias
            )
            state = (
                (1.0 - self.leak_rate) * state
                + self.leak_rate * np.tanh(pre_activation)
            )
            states[t] = state
        return states

    def site_drive(self, input_step: np.ndarray) -> np.ndarray:
        """The per-site input drive for one time step, as an array of shape (n,).

        Equals SpinQRC._generate_field_strengths(input_step) minus its constant_field, for
        the same encoding mode, input_scaling, Win and noise_fill.
        """
        drive = np.zeros(self.n)
        centred = 2.0 * (np.asarray(input_step, dtype=float) - 0.5)

        if self.encoding_mode == "one_to_one":
            drive[:self.d] = self.input_scaling * centred
        elif self.encoding_mode == "fill":
            sites_per_dim = self.n // self.d
            for i, value in enumerate(centred):
                start = i * sites_per_dim
                end = start + sites_per_dim
                drive[start:end] = (
                    self.input_scaling * value * self.noise_fill[start:end]
                )
        else:
            drive[:] = self.input_scaling * (self.W_in @ centred)
        return drive

    def _build_input_matrix(self, rng) -> np.ndarray | None:
        """The dense-encoding weight matrix, row-normalised so the drive stays bounded.

        Returns None for the modes that need no matrix. Row-normalising by the sum of
        absolute values, and replacing a zero row sum with 1.0, is what SpinQRC does.
        """
        if self.encoding_mode != "dense":
            return None
        W_in = rng.uniform(-1.0, 1.0, (self.n, self.d))
        row_sums = np.sum(np.abs(W_in), axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return W_in / row_sums

    def _build_reservoir_matrix(self, rng) -> np.ndarray:
        """A dense random recurrent matrix rescaled to unit spectral radius.

        Rescaling here means spectral_radius is the spectral radius of the recurrent term
        rather than an arbitrary scale. The guard leaves a matrix whose spectral radius
        underflows as drawn; at 3n >= 6 with dense U(-1, 1) entries this does not occur,
        but it makes a division by zero impossible.
        """
        W = rng.uniform(-1.0, 1.0, (self.n_nodes, self.n_nodes))
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 1e-12:
            W = W / radius
        return W
