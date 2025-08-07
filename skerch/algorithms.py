#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

TODO:
* add all in-core algorithms
* a-priori/posteriori/truncation stuff
* HDF5 measurement/wrapper API
* out-of-core wrappers for QR, SVD, LSTSQ
* Integration tests (add utests where needed):
  - comparing all recoveries for general and herm quasi-lowrank on complex128, using all types of noise -> boxplot
  - scale up: good recovery of very large composite linop, quick.
* future: diag, triang, lord



CHANGELOG:
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByVector
  - New measurement linops: Rademacher, Gaussian, Phase, SSRFT
* Sketching API:
  - Modular measurement API supporting multiprocessing and HDF5
  - Modular recovery methods (singlepass, Nystrom, oversampled) for
    general and symmetric cases
  - Algorithms: XDiag/DiagPP, SSVD/SEIGH, Sketchlord, Triangular
* A-posteriori error verification
* A-priori hyperparameter selection
"""


import torch
from .recovery import singlepass, nystrom, oversampled
from .recovery import singlepass_h, nystrom_h, oversampled_h
from .measurements import lop_measurement, perform_measurements
from .measurements import (
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SsrftNoiseLinOp,
)


# ##############################################################################
# # IN-CORE SSVD/SEIGH
# ##############################################################################
def ssvd(
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type=SsrftNoiseLinOp,
    inner_dims=None,
):
    """ """
    recovery_fn = singlepass if inner_dims is None else oversampled
    if inner_dims is not None and inner_dims < outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled recovery!"
        )

    breakpoint()
