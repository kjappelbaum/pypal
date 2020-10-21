# -*- coding: utf-8 -*-
"""Implements a PAL class for GBDT models which can predict uncertainity intervals
when used with quantile loss.
For an example of GBDT with quantile loss see
Jablonka, Kevin Maik; Moosavi, Seyed Mohamad; Asgari, Mehrdad; Ireland, Christopher;
Patiny, Luc; Smit, Berend (2020): A Data-Driven Perspective on
the Colours of Metal-Organic Frameworks. ChemRxiv. Preprint.
https://doi.org/10.26434/chemrxiv.13033217.v1

For general information about quantile regression
see https://en.wikipedia.org/wiki/Quantile_regression

Note that the scaling of the hyperrectangles has been derived
for GPR models (with RBF kernels).
"""
import concurrent.futures
from functools import partial

import numpy as np

from .pal_base import PALBase
from .validate_inputs import validate_njobs


def _train_model_picklable(i, models, design_space, objectives, sampled):
    model = models[i]
    model.fit(
        design_space[sampled[:, i]],
        objectives[sampled[:, i], i].reshape(-1, 1),
    )
    return model


class PALGBDT(PALBase):
    """PAL class for a list of LightGBM GBDT models"""

    def __init__(self, *args, **kwargs):
        """Construct the PALGBDT instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (List[Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]]:
                Machine learning models. You need to provide a list of tuples.
                One tuple per objective and every tuple contains three
                LGBMRegressors. The first one for the lower uncertainty limits,
                the middle one for the median and the last one for the upper limit.
                To create appropriate models, you need to use the quantile loss.
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
            coef_var_treshold (float, optional): Use only points with
                a coefficient of variation below this threshold
                in the classification step. Defaults to 3.
            n_jobs (int): Number of parallel processes that are used to fit
                the GPR models. Defaults to 1.
        """
        n_jobs = kwargs.pop("n_jobs", 1)
        validate_njobs(n_jobs)
        self.n_jobs = n_jobs
        super().__init__(*args, **kwargs)

        # ToDo: self.models = validate_sklearn_gpr_models(self.models, self.ndim)

    def _set_data(self):
        pass

    def _train(self):
        train_single_partial = partial(
            _train_model_picklable,
            models=self.models,
            design_space=self.design_space,
            objectives=self.y,
            sampled=self.sampled,
        )
        models = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_jobs
        ) as executor:
            for model in executor.map(train_single_partial, range(self.ndim)):
                models.append(model)
        self.models = models

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            mean, std = model.predict(self.design_space, return_std=True)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        # ToDo: potentially use hyperopt
        pass
