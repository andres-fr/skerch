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





TODO:

* finish a priori and __main__ stuff
* documentation, changelog, issues
* release!

* merging tracepp and diagpp: lop is being called more than once in GH!

* add trace operators, normal and hermitian (as advertised)
* a-priori/posteriori/truncation stuff
* Integration tests/docs (add utests where needed):
  - comparing all recoveries for general and herm quasi-lowrank on complex128, using all types of noise -> boxplot
  - scale up: good recovery of very large composite linop, quick.
  - priori and posteriori...
* add remaining todos as GH issues and release!
  - sketchlord(h) facilities: leave them for paper




LATER TODO:
* xtrace
* HDF5 measurement/wrapper API
* a-priori/posteriori/truncation stuff
* out-of-core wrappers for QR, SVD, LSTSQ
* sketchlord and sketchlordh.
* sketched permutations
* batched HDF5

Triang correctness:
* triang: stairs should include bits of main diag


CHANGELOG:
* Better test coverage -> less bugs
* Clearer docs
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByBlock
  - Support for parallelization of matrix-matrix products
  - New measurement noise linops: Rademacher, Gaussian, Phase, SSRFT
* Sketching API:
  - Modular measurement API supporting multiprocessing and HDF5
  - Modular recovery methods (singlepass, Nystrom, oversampled)
* Algorithm API:
  - Algorithms: XDiag/DiagPP, XTrace/TracePP, SSVD, Triangular, Norms
  - Efficient support for Hermitian versions
  - Dispatcher for modularized use of noise sources and recovery types
  - Matrix-free a-posteriori error verification
"""


import math


# ##############################################################################
# # A PRIORI HYPERPARAMETERS
# ##############################################################################
def general_lowrank_hyperparams(
    matrix_shape, memory_budget, complex_data=False, hermitian=False
):
    """Hyperparameters for low-rank sketches with general spectrum.

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
        raise NotImplementedError("Hermitian a-priori not yet supported!")
        # check that m == n ...
        # n = 0  # if matrix is hermitian ,we only count m, not n
    alpha = 0 if complex_data else 1
    mn4a = m + n + 4 * alpha
    budget_root = 16 * (memory_budget - alpha**2)
    #
    outer_dim = math.floor(
        (1 / 8) * (math.sqrt(mn4a**2 + budget_root) - mn4a)
    )
    core_dim = math.floor(math.sqrt(memory_budget - outer_dim * (m + n)))
    #
    return outer_dim, core_dim
