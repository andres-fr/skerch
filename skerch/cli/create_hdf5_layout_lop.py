#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""CLI plugin to create a HDF5 layout used to hold sketched measurements.

See also:

* :class:`skerch.hdf5.DistributedHDF5Tensor`
* :func:`skerch.hdf5.create_hdf5_layout_lop`
"""


import os

from .. import INNER_FMT, LO_FMT, RO_FMT
from ..hdf5 import create_hdf5_layout_lop


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(
    dirpath,
    lop_shape,
    lop_dtype,
    partsize,
    lo_meas=None,
    ro_meas=None,
    inner_meas=None,
):
    """Entry point for this CLI script. See module docstring."""
    if os.listdir(dirpath):
        raise RuntimeError("Directory must be empty!")
    lo = lo_meas is not None
    (
        (lo_pth, lo_subpaths, lo_begs_ends),
        (ro_pth, ro_subpaths, ro_begs_ends),
        (in_pth, in_subpaths, in_begs_ends),
    ) = create_hdf5_layout_lop(
        dirpath,
        lop_shape,
        lop_dtype,
        partsize,
        lo_meas=lo_meas,
        ro_meas=ro_meas,
        inner_meas=inner_meas,
        lo_fmt=LO_FMT,
        ro_fmt=RO_FMT,
        inner_fmt=INNER_FMT,
    )
    #
    printvars = (
        "dirpath",
        "lop_shape",
        "lop_dtype",
        "partsize",
        "lo_meas",
        "ro_meas",
        "inner_meas",
    )
    localvars = locals()
    print({k: localvars[k] for k in printvars})
