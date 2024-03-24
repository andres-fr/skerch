#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""``skerch``: Sketched matrix decompositions for PyTorch.

This implementation is based on the following publications, referenced
throughout the documentation:

* `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_: Joel A. Tropp, Alp
  Yurtsever, Madeleine Udell, and Volkan Cevher. 2019. *“Streaming Low-rank
  Matrix Approximation with an Application to Scientific Simulation”*. SIAM
  Journal on Scientific Computing 41 (4): A2430–63.
"""


# ##############################################################################
# # GLOBALS
# ##############################################################################

# format strings to store measurements in HDF5 files.
LO_FMT = "leftouter_{}.h5"
RO_FMT = "rightouter_{}.h5"
INNER_FMT = "inner_{}.h5"
