# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyepal.pal.pal_ensemble import PALEnsemble


def test_pal_ensemble_init(make_random_dataset):
    from pyepal.models.gpr import build_model
    from pyepal.pal.pal_gpy import PALGPy

    X, y = make_random_dataset
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # with pytest.raises(ValueError):
    #     # Shouldn't work if there are no kwargs

    #     ensemble = PALEnsemble.from_class_and_kwarg_lists(PALGPy, [])
    m0 = build_model(X, y, 0)  # pylint:disable=invalid-name
    m1 = build_model(X, y, 1)  # pylint:disable=invalid-name
    m2 = build_model(X, y, 2)  # pylint:disable=invalid-name

    palgpy_instance = PALGPy(
        X,
        models=[m0, m1, m2],
        ndim=3,
        delta=0.01,
        pooling_method="fro",
        restarts=3,
    )
    palgpy_instance_2 = PALGPy(
        X,
        models=[m0, m1, m2],
        ndim=3,
        delta=0.01,
        pooling_method="mean",
        restarts=3,
    )
    palgpy_instance.cross_val_points = 0
    palgpy_instance_2.cross_val_points = 0
    pal_ensemble = PALEnsemble([palgpy_instance, palgpy_instance_2])
    pal_ensemble.update_train_set(sample_idx, y[sample_idx])
    sample, _ = pal_ensemble.run_one_step(1)
    assert len(sample) == 1

    sample, _ = pal_ensemble.run_one_step(20)
    assert len(sample) == 20
