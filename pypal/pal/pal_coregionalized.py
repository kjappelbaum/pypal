# -*- coding: utf-8 -*-
"""PAL for coregionalized GPR models"""

import numpy as np

from ..models.gpr import predict_coregionalized, set_xy_coregionalized
from .pal_base import PALBase
from .schedules import linear
from .validate_inputs import validate_coregionalized_gpy


class PALCoregionalized(PALBase):
    """PAL class for a coregionalized GPR model"""

    def __init__(self, *args, **kwargs):
        """Construct the PALCoregionalized instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (list): Machine learning models
            ndim (int): Number of objectives
            epsilon (Union[list, float], optional): Epsilon hyperparameter.
                Defaults to 0.01.
            delta (float, optional): Delta hyperparameter. Defaults to 0.05.
            beta_scale (float, optional): Scaling parameter for beta.
                If not equal to 1, the theoretical guarantees do not necessarily hold.
                Also note that the parametrization depends on the kernel type.
                Defaults to 1/9.
            goals (List[str], optional): If a list, provide "min" for every objective
                that shall be minimized and "max" for every objective
                that shall be maximized. Defaults to None, which means
                that the code maximizes all objectives.
            restarts (int): Number of random restarts that are used for hyperparameter
                optimization. Defaults to 20.
            parallel (bool): If true, model hyperparameters are optimized in parallel,
                using the GPy implementation. Defaults to False.
        """
        self.restarts = kwargs.pop("restarts", 20)
        self.parallel = kwargs.pop("parallel", False)
        assert isinstance(
            self.parallel, bool
        ), "the parallel keyword must be of type bool"
        assert isinstance(
            self.restarts, int
        ), "the restarts keyword must be of type int"
        super().__init__(*args, **kwargs)
        validate_coregionalized_gpy(self.models)

    def _set_data(self):
        self.models[0] = set_xy_coregionalized(
            self.models[0],
            self.design_space[self.sampled_indices],
            self.y[self.sampled_indices],
            self.sampled[self.sampled_indices],
        )

    def _train(self):
        pass

    def _predict(self):
        means, stds = [], []
        for i in range(self.ndim):
            mean, std = predict_coregionalized(self.models[0], self.design_space, i)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        self.models[0].optimize_restarts(self.restarts, parallel=self.parallel)

    def _should_optimize_hyperparameters(self) -> bool:
        return linear(self.iteration, 10)
