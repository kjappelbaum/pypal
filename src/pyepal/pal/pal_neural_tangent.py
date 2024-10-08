# -*- coding: utf-8 -*-
# Copyright 2022 PyePAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Use PAL with the neural tangent library.
This allows to perform
1. Exact Bayesian inference (NNGP)
2. Inference using gradient descent with MSE loss (NTK)

Note that the neural tangent code usually assumes mean-zero Gaussians


Reducing predict_fn_kwargs['diag_reg'] typically improves the interpolation
quality
"""

from typing import Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .pal_base import PALBase
from .validate_inputs import validate_nt_models
from ..models.nt import NTModel

__all__ = ["PALNT", "NTModel"]


# We move those functions out of the class so that we can parallelize them
def _set_one_infinite_width_model(  # pylint:disable=too-many-arguments
    i: int,
    models: Sequence[NTModel],
    design_space: np.ndarray,
    objectives: np.ndarray,
    sampled: np.ndarray,
    predict_fn_kwargs: dict = None,
) -> Tuple[callable, StandardScaler]:
    from jax.config import config  # pylint:disable=import-outside-toplevel

    config.update("jax_enable_x64", True)
    import neural_tangents as nt  # pylint:disable=import-outside-toplevel

    if predict_fn_kwargs is None:
        predict_fn_kwargs = {"diag_reg": 1e-3}
    model = models[i]
    kernel_fn = model.kernel_fn
    scaler = StandardScaler()
    y = scaler.fit_transform(  # pylint:disable=invalid-name
        objectives[sampled[:, i], i].reshape(-1, 1)
    )
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn,
        design_space[sampled[:, i]],
        y,
        **predict_fn_kwargs,
    )

    return predict_fn, scaler


def _predict_one_infinite_width_model(
    i: int, models: Sequence[NTModel], design_space: np.ndarray, kernel: str
):
    predict_fn = models[i].predict_fn
    mean, covariance = predict_fn(  # type: ignore
        x_test=design_space,
        get=kernel,
        compute_cov=True,
    )

    return mean.flatten(), np.sqrt(np.diag(covariance))


class PALNT(PALBase):
    """ε-PAL with neural tangents models
    (in the current implementation, one model per objective)"""

    def __init__(self, *args, **kwargs):
        """Construct the PALNT instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (Sequence[NTModel]): You need to provide a sequence of
                 NTModel (`pyepal.models.nt.NTModel`).
                The elements of this dataclass are the `apply_fn`, `init_fn`,
                `kernel_fn` and `predict_fn` (for latter you can typically
                provide `None`)
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
            coef_var_threshold (float, optional): Use only points with
                a coefficient of variation below this threshold
                in the classification step. Defaults to 3.
            kernel (str, optional): The kernel type you want to use ('nngp' or 'ntk').
                NNGP corresponds to a Neural Network Gaussian Process, frist established
                by Neal in 1994.  NTK refers to the neural tangent kernel,
                i.e., the linear approximation of an infinite width neural network
                (LeCun initialized) trained with gradient descent (Jacot et al., 2018).
                Defaults to 'nngp'.
        """

        self.kernel = kwargs.pop("kernel", "nngp")
        self.design_space_scaler = StandardScaler()
        super().__init__(*args, **kwargs)
        self.models = validate_nt_models(self.models, self.ndim)

    def _set_data(self):
        self.design_space = self.design_space_scaler.fit_transform(self.design_space)
        for i, _ in enumerate(self.models):

            predict_fn, scaler = _set_one_infinite_width_model(
                i,
                self.models,
                self.design_space,
                self.y,
                self.sampled,
            )
            self.models[i].predict_fn = predict_fn
            self.models[i].scaler = scaler
            self.y[:, i] = scaler.transform(self.y[:, i].reshape(-1, 1)).flatten()

    def _train(self):
        pass

    def _predict(self):
        means, stds = [], []
        for i, _ in enumerate(self.models):
            mean, std = _predict_one_infinite_width_model(
                i,
                self.models,
                self.design_space,
                self.kernel,
            )
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self._means = np.hstack(means)
        self.std = np.hstack(stds)
