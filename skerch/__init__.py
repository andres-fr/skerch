#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""``skerch``: Sketched matrix operations for PyTorch.

``skerch`` is a PyTorch-based library to perform a variety of sketched
approximations and operations on matrices and matrix-free linear operators.

The following is a list of publications referenced in the documentation:

* `[TYUC2017] <https://arxiv.org/abs/1609.00048>`_: Joel A. Tropp, Alp
  Yurtsever, Madeleine Udell, and Volkan Cevher. 2019. *“Practical Sketching
  Algorithms for Low-Rank Matrix Approximation”*. SIAM.
* `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_: Joel A. Tropp, Alp
  Yurtsever, Madeleine Udell, and Volkan Cevher. 2019. *“Streaming Low-rank
  Matrix Approximation with an Application to Scientific Simulation”*. SIAM.
* `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston and Yuji
  Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds and
  an improved algorithm”*.  CoRR abs/2201.10684.
* `[ETW2024] <https://arxiv.org/abs/2301.07825>`_: Ethan N. Epperly, Joel A.
  Tropp, Robert J. Webber. 2024. *“XTrace: Making the Most of Every Sample in
  Stochastic Trace Estimation”*. SIAM.
* `[FSMH] <https://arxiv.org/abs/2504.14701>`_: Andres Fernandez, Frank
  Schneider, Maren Mahsereci, Philipp Hennig. 2025. *“Connecting Parameter
  Magnitudes and Hessian Eigenspaces at Scale using Sketched Methods”*. TMLR.
"""


# ##############################################################################
# # GLOBALS
# ##############################################################################

# format strings to store measurements in HDF5 files.
LO_FMT = "leftouter_{}.h5"
RO_FMT = "rightouter_{}.h5"
INNER_FMT = "inner_{}.h5"
