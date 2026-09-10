import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error

class Prediction:
    def __init__(self, observations: np.ndarray, data: np.ndarray, washout: int = 1000, train_length: int = 6000, test_length: int = 4000, model: str = "linear", ridge_alpha: float = 1.0, val_length: int = 0):
        """Initializes the Prediction class.
        Args:
            observations (np.ndarray): The reservoir states or features.
            data (np.ndarray): The target data to predict.
            washout (int): Number of initial samples to discard.
            train_length (int): Length of the training dataset.
            test_length (int): Length of the testing dataset.
            model (str): Type of regression model ("linear" or "ridge").
            ridge_alpha (float): Regularization strength for Ridge regression.
            val_length (int): Length of an optional validation block, sitting between the
                training and testing blocks. Zero -- the default -- is the plain
                two-way split, keys included. Declared last so that no positional
                caller shifts.
        """
        self.observations = observations
        self.data = data
        self.washout = washout
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.model = model
        self.ridge_alpha = ridge_alpha

        if len(self.observations) != len(self.data):
            raise ValueError(f"observations and data must have the same length. Got {len(self.observations)} and {len(self.data)}")

        if self.val_length < 0:
            raise ValueError(f"val_length must not be negative. Got {self.val_length}")

        if self.train_length <= 0 or self.test_length <= 0:
            raise ValueError(f"train_length and test_length must be positive. Got {self.train_length} and {self.test_length}")

        total = self.washout + self.train_length + self.val_length + self.test_length
        if len(self.observations) != total:
            raise ValueError(f"observations length must equal washout + train_length + val_length + test_length. Got {len(self.observations)}, expected {total}")

    @property
    def splits(self) -> tuple[str, ...]:
        """The blocks this instance reports on. 'val' appears only when it is non-empty."""
        return ("train", "val", "test") if self.val_length > 0 else ("train", "test")

    def prediction_multi_step(self, max_steps: int=10) -> dict:
        """Performs multi-step ahead prediction.
        Args:
            max_steps (int): Maximum number of steps to predict ahead.
        Returns:
            dict: A dictionary containing RMSE results for each prediction step, per block.
        """
        results_dict = {}
        for split in self.splits:
            results_dict[f"rmse_{split}_average"] = np.zeros(max_steps)
            results_dict[f"rmse_{split}_list"] = np.zeros((max_steps, self.data.shape[1]))
            results_dict[f"nrmse_{split}_average"] = np.zeros(max_steps)
            results_dict[f"nrmse_{split}_list"] = np.zeros((max_steps, self.data.shape[1]))

        end = self.washout + self.train_length + self.val_length + self.test_length
        for i in range(1, max_steps+1):
            targets = np.roll(self.data, -i, axis=0)
            targets = targets[self.washout:end-i, :]
            observations = self.observations[self.washout:end-i, :]
            for split, metrics in self._trainer(observations, targets).items():
                results_dict[f"rmse_{split}_average"][i-1] = metrics["rmse_average"]
                results_dict[f"rmse_{split}_list"][i-1, :] = metrics["rmse_list"]
                results_dict[f"nrmse_{split}_average"][i-1] = metrics["nrmse_average"]
                results_dict[f"nrmse_{split}_list"][i-1, :] = metrics["nrmse_list"]
        return results_dict

    def _trainer(self, observations, targets) -> dict:
        """Fits the readout on the training block and scores every block.

        The blocks are contiguous and in order, so a validation block takes its samples from
        the front of what was the test block; the fit sees the same training rows either
        way. std_targets is computed over the whole sliced range, which does not depend on
        where the blocks are cut, so the NRMSE denominator is the same either way too.
        """
        blocks = {"train": slice(0, self.train_length)}
        if self.val_length > 0:
            blocks["val"] = slice(self.train_length, self.train_length + self.val_length)
        blocks["test"] = slice(self.train_length + self.val_length, None)

        if self.model == "linear":
            model = LinearRegression()
        elif self.model == "ridge":
            model = Ridge(alpha=self.ridge_alpha)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        model.fit(observations[blocks["train"]], targets[blocks["train"]])

        std_targets = np.std(targets, axis=0)

        metrics = {}
        for split, block in blocks.items():
            y_true = targets[block]
            y_pred = model.predict(observations[block])
            rmse_list = root_mean_squared_error(y_true, y_pred, multioutput='raw_values')
            nrmse_list = rmse_list / std_targets
            metrics[split] = {
                "rmse_average": root_mean_squared_error(y_true, y_pred),
                "rmse_list": rmse_list,
                "nrmse_average": np.mean(nrmse_list),
                "nrmse_list": nrmse_list,
            }
        return metrics
