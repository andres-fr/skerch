#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""CLI plugin to create a HDF5 layout to hold outer and inner measurements."""


import os

import torch

from ..distributed_decompositions import create_hdf5_layout


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(
    dirpath, shape, lop_dtype, outer, inner, lo_fmt, ro_fmt, inner_fmt, with_ro
):
    """See :func:`.distributed_decompositions.create_hdf5_layout`."""
    h, w = shape
    if (h != w) and (not with_ro):
        raise AssertionError("Non-square matrix can't be symmetric!")
    assert not os.listdir(dirpath), "Directory must be empty!"
    #
    lop_dtype = getattr(torch, lop_dtype)
    (
        (lo_path, lo_subpaths),
        (ro_path, ro_subpaths),
        (inner_path, inner_subpaths),
    ) = create_hdf5_layout(
        dirpath,
        shape,
        lop_dtype,
        outer,
        inner,
        lo_fmt=lo_fmt,
        ro_fmt=ro_fmt,
        inner_fmt=inner_fmt,
        with_ro=with_ro,
    )
    print(dirpath)
    print(shape)
    print(lop_dtype)
    print(outer)
    print(inner)
    print(lo_fmt)
    print(ro_fmt)
    print(inner_fmt)
    print(with_ro)
    #
    print(lo_path)
    print(ro_path)
    print(inner_path)
    # print(lo_subpaths)
    # print(ro_subpaths)
    # print(inner_subpaths)
