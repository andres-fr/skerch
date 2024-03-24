#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""A-priori utilities for sketched decompositions.

Given a limited amount of memory, we would like to know how many inner and
outer measurements we should perform to get the best results.

In `[TYUC2019, 5.4.2] <https://arxiv.org/abs/1902.08651>`_, it is argued that
the number of outer measurements should be maximized within memory constraints,
subject to the number of inner measurements being twice.
This yields a closed-form optimum for general matrices, presented in the paper
and implemented here.
"""


import math


# ##############################################################################
# # A PRIORI HYPERPARAMETERS
# ##############################################################################
def a_priori_hyperparams(
    matrix_shape, memory_budget, complex_data=False, hermitian=False
):
    """A-priori hyperparameter selection for sketched measurements.

    Given the shape of a generic linear operator that we wish to decompose, and
    a memory budget, return the number of inner and outer measurements required,
    such that the budget is not exceeded, ``inner >= 2 * outer``, and outer is
    maximized (see `[TYUC2019, 5.4.2] <https://arxiv.org/abs/1902.08651>`_).

    :param int memory_budget: Memory available in number of matrix entries.
      E.g. if the matrix is in float32, a budget of ``N`` means ``4N`` bytes.
    :param hermitian: If true, the matrix is assumed to be hermitian (and
      square), so measurements only need to be done on one side. This means
      that, for the same budget, we can do more measurements.
    :returns: The pair ``(k, s)``, where the first integer is the optimal
      number of outer sketch measurements, and the second one is the
      corresponding number of core measurements. Optimality is defined as:
      distribute the budget such that k is maximized, but ``s >= 2k``
      (see 5.4.2. in paper).
    """
    m, n = matrix_shape
    if hermitian:
        assert m == n, "Hermitian matrix must be also square!"
        n = 0  # if matrix is hermitian ,we only count m, not n
    alpha = 0 if complex_data else 1
    mn4a = m + n + 4 * alpha
    budget_root = 16 * (memory_budget - alpha**2)
    #
    outer_dim = math.floor((1 / 8) * (math.sqrt(mn4a**2 + budget_root) - mn4a))
    core_dim = math.floor(math.sqrt(memory_budget - outer_dim * (m + n)))
    #
    return outer_dim, core_dim
