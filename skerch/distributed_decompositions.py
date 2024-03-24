#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Sketched decompositions on distributed systems.

For in-core decompositions, see :mod:`.distributed_decompositions`.
"""


import os

import numpy as np
import scipy
import torch

from .distributed_measurements import DistributedHDF5
from .linops import DiagonalLinOp

#
from .utils import torch_dtype_as_str


# ##############################################################################
# # CREATE HDF5 LAYOUTS
# ##############################################################################
def create_hdf5_layout(
    dirpath,
    lop_shape,
    lop_dtype,
    num_outer_measurements,
    num_inner_measurements,
    lo_fmt="leftouter_{}.h5",
    ro_fmt="rightouter_{}.h5",
    inner_fmt="inner_{}.h5",
    with_ro=True,
):
    """Creation of persistent HDF5 files to store sketches of a linear operator.

    :param str dirpath: Where to store the HDF5 files.
    :param lop_shape: Shape of linear operator to sketch from, in the form
      ``(height, width)``.
    :param lop_dtype: Torch dtype of the operator, e.g. ``torch.float32``. The
      HDF5 arrays will be of same type.
    :param int num_outer_measurements: Left outer measurement layout contains
      ``(width, outer)`` entries, and right outer layout ``(height, outer)``.
    :param int num_inner_measurements: Inner measurement layout contains
      ``(inner, inner)`` entries.
    :lo_fmt: Format string for the left-outer HDF5 filenames.
    :ro_fmt: Format string for the right-outer HDF5 filenames.
    :inner_fmt: Format string for the inner HDF5 filenames.
    :param with_ro: If false, no right outer layout will be created (useful
      when working with symmetric matrices where only one side is needed).
    """
    h, w = lop_shape
    lo_pth, lo_subpths = DistributedHDF5.create(
        os.path.join(dirpath, lo_fmt),
        num_outer_measurements,
        (w,),
        torch_dtype_as_str(lop_dtype),
        # this needs to be transposed later, but we keep it tall to allow
        # for QR without having to transpose (which may load to RAM)
        filedim_last=True,
    )
    #
    if with_ro:
        ro_pth, ro_subpths = DistributedHDF5.create(
            os.path.join(dirpath, ro_fmt),
            num_outer_measurements,
            (h,),
            torch_dtype_as_str(lop_dtype),
            filedim_last=True,
        )
    else:
        ro_pth, ro_subpths = None, None
    #
    c_pth, c_subpths = DistributedHDF5.create(
        os.path.join(dirpath, inner_fmt),
        num_inner_measurements,
        (num_inner_measurements,),
        torch_dtype_as_str(lop_dtype),
        filedim_last=True,
    )
    return ((lo_pth, lo_subpths), (ro_pth, ro_subpths), (c_pth, c_subpths))


# ##############################################################################
# # QR DECOMPOSITIONS
# ##############################################################################
def orthogonalize(matrix, overwrite=False):
    """Orthogonalization of given matrix.

    :param matrix: matrix to orthogonalize, needs to be compatible with either
      ``scipy.linalg.qr`` or ``torch.linalg.qr``.
    :param overwrite: If true, ``matrix[:] = Q`` will also be performed.
    :returns: Orthogonal matrix ``Q`` such that ``matrix = Q @ R`` as per the
      QR decomposition.
    """
    h, w = matrix.shape
    assert h >= w, "Only non-fat matrices supported!"
    #
    if isinstance(matrix, torch.Tensor):
        Q = torch.linalg.qr(matrix, mode="reduced")[0]
    else:
        Q = scipy.linalg.qr(
            matrix,
            mode="economic",
            pivoting=True,
        )[0]
    #
    if overwrite:
        matrix[:] = Q
    return Q


# ##############################################################################
# # CORE SOLVERS
# ##############################################################################
def core_pinv_solve(Q, randlop, target):
    """Pseudo-inverse step as part of solving the core sketch matrix.

    The core matrix can be expressed in terms of ``pinv(randlop @ Q) @ target``
    operations (see `[TYUC2019, 2.7] <https://arxiv.org/abs/1902.08651>`_).
    Given the ``target`` operator, this expression can be computed
    directly via least squares without having to explicitly compute the ``pinv``
    pseudo-inverse, which is typically faster and more stable.

    This function is used as a sub-routine in :func:`solve_core_ssvd` and
    :func:`solve_core_seigh`.

    :param Q: Matrix of shape ``(h, outer_measurements)`` containing either the
      left-outer or right-outer measurements.
    :param randlop: Linear operator used to perform the inner measurements.
      If ``Q`` contains right-outer measurements, this must be the left-inner
      operator, and vice versa. Expected shape is ``(num_inner, h)``.
    :returns: matrix equivalent to ``pinv(randlop @ Q) @ target``
    """
    h, outer = Q.shape
    inner, h2 = randlop.shape
    assert h == h2, "Inconsistent Q and randlop shapes!"
    assert target.shape[0] == inner, "Inconsistent randlop and target shapes!"
    #
    with_torch = isinstance(Q, torch.Tensor)
    dtype = torch_dtype_as_str(Q.dtype) if with_torch else str(Q.dtype)
    torch_dtype = getattr(torch, dtype)
    #
    if with_torch:
        buff = torch.zeros(inner, dtype=torch_dtype).to(Q.device)
        reduced = torch.empty((inner, outer), dtype=torch_dtype).to(Q.device)
        for i in range(inner):
            buff[i - 1] = 0
            buff[i] = 1
            reduced[i, :] = (buff @ randlop) @ Q
        result = torch.linalg.lstsq(reduced, target).solution
    else:
        buff = torch.zeros(inner, dtype=torch_dtype)
        reduced = np.empty((inner, outer), dtype=dtype)
        for i in range(inner):
            buff[i - 1] = 0
            buff[i] = 1
            reduced[i, :] = (buff @ randlop).numpy() @ Q
        result = scipy.linalg.lstsq(reduced[:], target[:])[0]
    #
    return result


def solve_core_ssvd(ro_Q, lo_Q, inner_measurements, li_randlop, ri_randlop):
    """Solving the core sketch matrix for sketched SVD.

    Given the orthogonalized right- and left-outer measurements, as well as
    the inner measurements and the linear operators used to perform said
    random measurements in the context of a sketched SVD, this function solves
    the core matrix, following
    `[TYUC2019, 2.7] <https://arxiv.org/abs/1902.08651>`_.

    .. note::
      This implementation is compatible with torch tensors, numpy arrays and
      HDF5 arrays.

    :param ro_Q: Right-outer measurement matrix. Expected to be in thin
      vertical shape, i.e. ``(lop_height, num_outer_measurements)``. Also
      expected to be orthogonal, i.e. ``ro_Q.T @ ro_Q = I``.
    :param lo_Q: Left-outer measurement matrix. Expected to be in thin
      vertical shape, i.e. ``(lop_width, num_outer_measurements)``. Also
      expected to be orthogonal, i.e. ``lo_Q.T @ lo_Q = I``.
    :param inner_measurements: Inner measurement matrix of shape
      ``(num_inner, num_inner)``.
    :parma li_randlop: Torch linear operator used to perform inner measurements
      via ``li_randlop @ lop @ ri_randlop.T``. Expected shape is
      ``(inner_measurements, lop_height)``.
    :parma ri_randlop: see ``li_randlop``. Expected shape is
      ``(inner_measurements, lop_height)``.
    """
    # figure out if we are using tensors
    ro_tensor = isinstance(ro_Q, torch.Tensor)
    lo_tensor = isinstance(lo_Q, torch.Tensor)
    inner_tensor = isinstance(inner_measurements, torch.Tensor)
    assert (
        ro_tensor == lo_tensor == inner_tensor
    ), "Either all or none of the measurements should be tensors!"
    with_torch = ro_tensor
    # consistency of dtypes
    assert ro_Q.dtype == lo_Q.dtype, "Inconsistent ro_Q and lo_Q dtypes!"
    assert (
        ro_Q.dtype == inner_measurements.dtype
    ), "Inconsistent ro_Q and inner_measurement dtypes!"
    # consistency of outer measurement shapes
    h, num_outer = ro_Q.shape
    w, _ = lo_Q.shape
    assert (
        num_outer == lo_Q.shape[1]
    ), "Mismatching left and right number of outer measurements!"
    # consistency of inner measurement shapes
    num_inner = li_randlop.shape[0]
    assert (
        ri_randlop.shape[0] == num_inner
    ), "Mismatching left and right number of inner measurements!"
    assert li_randlop.shape[1] == h, "Mismatching li_randlop and lop height!"
    assert ri_randlop.shape[1] == w, "Mismatching li_randlop and lop width!"
    # solve core via pseudoinverses
    core = core_pinv_solve(ro_Q, li_randlop, inner_measurements)
    core = core_pinv_solve(lo_Q, ri_randlop, core.T).T

    # compute and return SVD decomposition of core
    if with_torch:
        core_U, core_S, core_Vt = torch.linalg.svd(core)
    else:
        core_U, core_S, core_Vt = scipy.linalg.svd(
            core, overwrite_a=True, lapack_driver="gesvd"
        )
    return core_U, core_S, core_Vt


def solve_core_seigh(Q, inner, li_randlop, ri_randlop):
    """Solving the core sketch matrix for sketched Hermitian eigendecomposition.

    Given the orthogonalized outer measurements, as well as the inner
    measurements and the linear operators used to perform said
    inner measurements in the context of a sketched EIGH, this function solves
    the core matrix, following [...].

    .. note::
      This implementation is compatible with torch tensors, numpy arrays and
      HDF5 arrays.

    :param Q: outer measurement matrix (side doesn't matter since matrix is
      assumed to be Hermitian). Expected to be in thin vertical shape, i.e.
      ``(lop_dim, num_outer_measurements)``. Also expected to be orthogonal,
      i.e. ``Q.T @ Q = I``.
    :param inner: Inner measurement matrix of shape ``(num_inner, num_inner)``.
    :parma li_randlop: Torch linear operator used to perform inner measurements
      via ``li_randlop @ lop @ ri_randlop.T``. Expected shape is
      ``(inner_measurements, lop_height)``.
    :parma ri_randlop: see ``li_randlop``. Expected shape is
      ``(inner_measurements, lop_height)``.
    """
    # figure out if we are using tensors
    o_tensor = isinstance(Q, torch.Tensor)
    i_tensor = isinstance(inner, torch.Tensor)
    assert (
        o_tensor == i_tensor
    ), "Either all or none of the measurements should be tensors!"
    with_torch = o_tensor
    # consistency of dtypes
    assert Q.dtype == inner.dtype, "Inconsistent datatypes!"
    # consistency of outer measurement shapes
    h, num_outer = Q.shape
    inner_h, inner_w = inner.shape
    assert inner_h == inner_w, "Inner operator not square?"
    assert li_randlop.shape == (inner_h, h), "Mismatching shapes!"
    assert ri_randlop.shape == (inner_h, h), "Mismatching shapes!"
    # solve asymmetric core via SVD and 2 pseudoinverses
    if with_torch:
        inner_U, inner_S, inner_V = torch.linalg.svd(inner)
    else:
        inner_U, inner_S, inner_V = scipy.linalg.svd(inner)
    l_core = core_pinv_solve(Q, li_randlop, inner_U)
    del inner_U
    r_core = core_pinv_solve(Q, ri_randlop, inner_V.T)
    del inner_V
    core = l_core @ (DiagonalLinOp(inner_S) @ r_core.T)
    del inner_S
    # compute Hermitian eigendecomposition of core and sort recovered eigdec
    # in descending magnitude, since may be non-PSD
    if with_torch:
        core_S, core_U = torch.linalg.eigh(core)  # mat = U @ diag(S) @ U.T
        _, perm = abs(core_S).sort(descending=True)
        core_S, core_U = core_S[perm], core_U[:, perm]
    else:
        core_S, core_U = scipy.linalg.eigh(core)  # mat = U @ diag(S) @ U.T
        perm = np.argsort(abs(core_S))[::-1]
        core_S, core_U = core_S[perm], core_U[:, perm]
    return core_U, core_S
