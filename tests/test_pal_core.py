# -*- coding: utf-8 -*-
# pylint:disable=unused-import
"""Testing the PAL module"""
import numpy as np

from PyPAL.pal.core import (
    _get_max_wt,
    _get_uncertainity_region,
    _get_uncertainity_regions,
    _union,
    _union_one_dim,
)


def test__get_uncertainity_region():
    """make sure that the uncertainity windows is computed in a reasonable way"""
    mu = 1  # pylint:disable=invalid-name

    low0, high0 = _get_uncertainity_region(mu, 0, 1)
    assert low0 == mu
    assert high0 == mu

    low1, high1 = _get_uncertainity_region(mu, 1, 0)
    assert low1 == mu
    assert high1 == mu

    low2, high2 = _get_uncertainity_region(mu, 0, 0)
    assert low2 == mu
    assert high2 == mu

    low3, high3 = _get_uncertainity_region(mu, 1, 1)
    assert low3 == 0
    assert high3 == 2


def test__get_uncertainity_regions():
    """The test uncertainity regions for three dimensions"""
    mu = 1  # pylint:disable=invalid-name
    lows, highs = _get_uncertainity_regions([mu, mu, mu], [0, 1, 2], 1)
    assert lows[0] == mu
    assert highs[0] == mu
    assert lows[1] == 0
    assert highs[1] == 2
    assert lows[2] == -1
    assert highs[1] == 3

    lows, highs = _get_uncertainity_regions([mu - 1, mu, mu], [0, 1, 2], 1)
    assert lows[0] == mu - 1
    assert highs[0] == mu - 1
    assert lows[1] == 0
    assert highs[1] == 2
    assert lows[2] == -1
    assert highs[1] == 3


def test__union_one_dim():
    """Make sure that the intersection of the uncertainity regions works"""
    zeros = np.array([0, 0, 0])
    zero_one_one = np.array([0, 1, 1])
    # Case 1: Everything is zero, we should also return zero
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )

    assert (low == zeros).all()
    assert (up == zeros).all()

    # Case 2: This should also work if this is the case for only one material
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]
    )
    assert (low == zero_one_one).all()
    assert (up == zero_one_one).all()

    # Case 3: Uncertainity regions do not intersect
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]
    )
    assert (low == [2, 2, 2]).all()
    assert (up == [3, 3, 3]).all()

    # Case 4: We have an intersection
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [3, 3, 3]
    )
    assert (low == np.array([0.5, 0.5, 0.5])).all()
    assert (up == np.array([1, 1, 1])).all()
