# -*- coding: utf-8 -*-
"""Testing the PALGPy class"""
import pytest

from PyPAL.models.gpr import build_model
from PyPAL.pal.pal_gpy import PALGPy


def test_pal_gpy(make_random_dataset):
    """Test basic functionality of the PALGpy class"""
    with pytest.raises(TypeError):
        palgpy_instance = PALGPy()

    X, y = make_random_dataset  # pylint:disable=invalid-name

    with pytest.raises(ValueError):
        palgpy_instance = PALGPy(X, ["m", "m", "m"], 3)

    m0 = build_model(X, y, 0)  # pylint:disable=invalid-name
    m1 = build_model(X, y, 1)  # pylint:disable=invalid-name
    m2 = build_model(X, y, 2)  # pylint:disable=invalid-name

    palgpy_instance = PALGPy(X, [m0, m1, m2], 3)
    assert palgpy_instance.restarts == 20
    assert not palgpy_instance.parallel