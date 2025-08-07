#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
LATER TODO:
* add all in-core algorithms
* formal tests for algorithms and dispatchers
* HDF5 measurement/wrapper API
* a-priori/posteriori/truncation stuff
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


from functools import partial
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
from .linops import CompositeLinOp, TransposedLinOp
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
        recovery_fn = oversampled_h if hermitian else oversampled
        inner_dims = int(recovery_type.split("_")[-1])
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
    max_mp_workers=None,
    lstsq_rcond=1e-6,
):
    """ """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    recovery_fn, inner_dims = recovery_dispatcher(recovery_type, False)
    if (outer_dims > max(h, w)) or (
        inner_dims is not None and (inner_dims > max(h, w))
    ):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims < outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    # instantiate outer measurement linops
    ro_seed = seed
    lo_seed = ro_seed + outer_dims + 1
    ro_mop = noise_dispatcher(
        noise_type, (w, outer_dims), ro_seed, lop_dtype, register
    )
    lo_mop = noise_dispatcher(
        noise_type, (h, outer_dims), lo_seed, lop_dtype, register
    )
    # instantiate inner measurement linops
    if inner_dims is not None:
        ri_seed = lo_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = noise_dispatcher(
            noise_type, (w, inner_dims), ri_seed, lop_dtype, register
        )
        li_mop = noise_dispatcher(
            noise_type, (h, inner_dims), li_seed, lop_dtype, register
        )
    # perform outer measurements
    _, ro_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=ro_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=False,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    _, lo_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=lo_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=True,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    # solve sketches
    if inner_dims is None:
        U, S, Vh = recovery_fn(
            ro_sketch, lo_sketch, ro_mop, rcond=lstsq_rcond, as_svd=True
        )
    # if oversampled, perform inner measurements before solving
    else:
        lop_mop = CompositeLinOp([("lop", lop), ("ri", ri_mop)])
        _, inner_sketch = perform_measurements(
            partial(
                lop_measurement,
                lop=lop_mop,
                meas_lop=li_mop,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(inner_dims),
            adjoint=True,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        U, S, Vh = recovery_fn(
            ro_sketch,
            lo_sketch,
            inner_sketch,
            TransposedLinOp(li_mop),
            ri_mop,
            rcond=lstsq_rcond,
            as_svd=True,
        )
    #
    return U, S, Vh


def seigh(
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    recovery_type="singlepass",
    max_mp_workers=None,
    lstsq_rcond=1e-6,
    by_mag=True,
):
    """ """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise ValueError("SEIGH expects square operators!")
    dims = h
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    recovery_fn, inner_dims = recovery_dispatcher(recovery_type, True)
    if (outer_dims > dims) or (inner_dims is not None and (inner_dims > dims)):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims < outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    # instantiate right-combined measurement linop
    combined_dims = outer_dims if inner_dims is None else inner_dims
    r_seed = seed
    r_mop = noise_dispatcher(
        noise_type, (dims, combined_dims), r_seed, lop_dtype, register
    )
    # instantiate outer measurement linop and perform outer measurements
    ro_seed = seed
    ro_mop = noise_dispatcher(
        noise_type, (dims, outer_dims), ro_seed, lop_dtype, register
    )
    _, ro_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=ro_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=False,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    if inner_dims is None:
        # if no oversampling, solve sketch and return
        ews, evs = recovery_fn(
            ro_sketch,
            ro_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    else:
        # if oversampled, perform inner measurements before solving
        ri_seed = ro_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = noise_dispatcher(
            noise_type, (dims, inner_dims), ri_seed, lop_dtype, register
        )
        li_mop = noise_dispatcher(
            noise_type, (dims, inner_dims), li_seed, lop_dtype, register
        )
        #
        lop_mop = CompositeLinOp([("lop", lop), ("ri", ri_mop)])
        _, inner_sketch = perform_measurements(
            partial(
                lop_measurement,
                lop=lop_mop,
                meas_lop=li_mop,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(inner_dims),
            adjoint=True,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        ews, evs = recovery_fn(
            ro_sketch,
            inner_sketch,
            TransposedLinOp(li_mop),
            ri_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    #
    return ews, evs
