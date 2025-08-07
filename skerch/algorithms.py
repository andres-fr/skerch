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
from .utils import COMPLEX_DTYPES


# ##############################################################################
# # HELPERS
# ##############################################################################
def recovery_dispatcher(recovery_type, hermitian=False):
    """ """
    inner_dims = None
    if recovery_type == "singlepass":
        recovery_fn = singlepass_h if hermitian else singlepass
    elif recovery_type == "nystrom":
        recovery_fn = nystrom_h if hermitian else nystrom
    elif "oversampled" in recovery_type:
        recovery_fn = oversampled_ if hermitian else oversampled
        inner_dims = int(recovery_type.split("_")[-1])
        if inner_dims < outer_dims:
            raise ValueError(
                "Inner dims must be larger than outer for oversampled!"
            )
    else:
        supported = "singlepass, nystrom, oversampled_12345"
        raise ValueError(
            f"Unknown recovery type! {recovery_type}! "
            "Supported: {supported}"
        )
    #
    return recovery_fn, inner_dims


def noise_dispatcher(noise_type, hw, seed, dtype, register=False):
    """ """
    if noise_type == "rademacher":
        mop = RademacherNoiseLinOp(
            hw, seed, dtype, by_row=False, register=register
        )
    elif noise_type == "gaussian":
        mop = GaussianNoiseLinOp(
            hw, seed, dtype, by_row=False, register=register
        )
    elif noise_type == "ssrft":
        mop = SsrftNoiseLinOp(hw, seed, norm="ortho")
    elif noise_type == "phase":
        if dtype not in COMPLEX_DTYPES:
            raise ValueError(
                "Phase noise expects complex dtype! Use Rademacher instead"
            )
        mop = PhaseNoiseLinOp(
            hw, seed, dtype, by_row=False, register=register, conj=False
        )
    else:
        supported = "rademacher, gaussian, ssrft, phase"
        raise ValueError(
            f"Unknown recovery type! {recovery_type} " "Supported: {supported}"
        )
    #
    return mop


# ##############################################################################
# # IN-CORE SSVD/SEIGH
# ##############################################################################
def ssvd(
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    recovery_type="singlepass",
):
    """ """
    h, w = lop.shape
    recovery_fn, inner_dims = recovery_dispatcher(recovery_type, False)
    if (outer_dims > max(h, w)) or (
        inner_dims is not None and (inner_dims > max(h, w))
    ):
        raise ValueError("More measurements than rows/columns not supported!")
    #
    ro_seed = seed
    lo_seed = ro_seed + outer_dims + 1
    ro_mop = noise_dispatcher(
        noise_type, (w, outer_dims), ro_seed, lop_dtype, True
    )
    lo_mop = noise_dispatcher(
        noise_type, (h, outer_dims), lo_seed, lop_dtype, True
    )
    #
    if inner_dims is not None:
        ri_seed = lo_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = noise_dispatcher(
            noise_type, (w, inner_dims), ri_seed, lop_dtype, True
        )
        li_mop = noise_dispatcher(
            noise_type, (h, inner_dims), li_seed, lop_dtype, True
        )
    #
    breakpoint()

    # mop = noise_dispatcher(
    #     noise_type,
    # )
    # meas_fn = partial(
    #     lop_measurement,
    #     lop=lop,
    #     meas_lop=mop,
    #     device=device,
    #     dtype=dtype,
    # )
    # _, z1 = perform_measurements(
    #     meas_fn,
    #     range(mop.shape[1]),
    #     adjoint=False,
    #     parallel_mode=parall,
    #     compact=True,
    #     max_mp_workers=max_mp_workers,
    # )

    # mop = noise_dispatcher(noise_type, hw, seed, dtype, register=False)
    breakpoint()
