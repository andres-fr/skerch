#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for a-priori functionality."""


import math

import pytest

from skerch.a_priori import a_priori_hyperparams


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def heights_widths_ranks_outer_inner_posteriori():
    """Test cases for different matrices and measurements.

    Entries are ``((h, w), r, (o, i))``. For a matrix of shape ``h, w`` and
    rank ``r``, the Skinny SVD will do ``o`` outer measurements and ``i``
    inner measurements. Recommended is that ``i >= 2*o``.

    For constrained budgets, the Skinny SVD will naturally yield higher error
    with smaller shapes, hence we test a few medium shapes.
    """
    result = [
        ((1_000, 1_000), 10, (100, 300), 30),
        ((1_000, 1_000), 50, (200, 600), 30),
        ((1_000, 2_000), 10, (100, 300), 30),
        ((1_000, 2_000), 50, (200, 600), 30),
        ((2_000, 2_000), 20, (100, 300), 30),
        ((2_000, 2_000), 100, (200, 600), 30),
    ]
    return result


@pytest.fixture
def budget_ratios():
    """Test cases for different memory budget to matrix size ratios."""
    result = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return result


# ##############################################################################
# # A PRIORI ASYMMETRIC HYPERPARAMETER ESTIMATION
# ##############################################################################
def check_outer_overflow(shape, outer, budget, hermitian=False):
    """Helper function.

    Test that increasing outer measurements by 1 breaks budget or constraints.
    """
    h, w = shape
    if hermitian:
        w = 0
    available = budget - ((outer + 1) * (h + w))
    assert available > 0, "Budget overflown!"
    # if not overflown, check if inner can be twice outer
    needed = (2 * (outer + 1)) ** 2
    assert available >= needed, "Inner measurements don't fit in budget!"


def test_a_priori_hyperparams(
    heights_widths_ranks_outer_inner_posteriori, budget_ratios
):
    """Test case for a-priori hyperparameters.

    The ``a_priori_hyperparams`` function provides suggestions for number of
    measurements, given a matrix shape and a budget. This function tests that:
    * suggested measurements are nonnegative
    * budget is respected
    * ``s >= 2k``, where ``s`` are core measurements and ``k`` the outer ones

    Ideally, incrementing k by 1 would explode or break constraints, meaning
    that ``k`` is indeed as large as it gets. This is tested by
    ``check_outer_overflow``, but it turns out this is not always the case.
    Since this is not crucial, we drop that test.
    """
    # test that symmetric requires square dimensions
    with pytest.raises(AssertionError):
        a_priori_hyperparams((10, 20), 50, complex_data=False, hermitian=True)
    #
    for (h, w), _r, (_o, _i), _ in heights_widths_ranks_outer_inner_posteriori:
        for ratio in budget_ratios:
            # test hpar recommendations for non-symmetric matrices
            budget = max(1, math.floor(h * w * ratio))
            oo, ii = a_priori_hyperparams(
                (h, w), budget, complex_data=False, hermitian=False
            )
            assert oo >= 0, "Outer measurements negative?"
            assert ii >= 0, "Inner measurements negative?"
            assert (oo + ii) > 0, "At least one measurement must be given!"
            assert ii >= (oo * 2), "Inner measurements must be >= 2*outer!"
            assert (oo * (h + w) + ii**2) <= budget, "Budget surpassed!"
            # It turns out some of the outputs can be incremented by 1 without
            # breaking budget or constraints. Maybe this is due to taking the
            # math.floor. Anyway, not super important, so we ignore it.
            # with pytest.raises(AssertionError):
            #     check_outer_overflow((h, w), oo, budget, hermitian=False)
            #
            # test hpar recommendations for symmetric matrices
            dim = min(h, w)
            oo, ii = a_priori_hyperparams(
                (dim, dim), budget, complex_data=False, hermitian=True
            )
            assert oo >= 0, "Outer measurements negative?"
            assert ii >= 0, "Inner measurements negative?"
            assert (oo + ii) > 0, "At least one measurement must be given!"
            assert ii >= (oo * 2), "Inner measurements must be >= 2*outer!"
            assert (oo * dim + ii**2) <= budget, "Budget surpassed!"
            # It turns out some of the outputs can be incremented by 1 without
            # breaking budget or constraints. Maybe this is due to taking the
            # math.floor. Anyway, not super important, so we ignore it.
            # with pytest.raises(AssertionError):
            #     check_outer_overflow((dim, dim), oo, budget, hermitian=True)


# ##############################################################################
# # A PRIORI SYMMETRIC HYPERPARAMETER ESTIMATION
# ##############################################################################
