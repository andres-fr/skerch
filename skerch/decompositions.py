#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""In-core, single-process sketched decompositions.

.. note::
  Often, when the decomposition is small enough to fit into a single process
  and core, the traditional and exact algorithms may be preferred.
  For the cases that do not fit, see :mod:`.distributed_decompositions`.
"""


import torch

from .distributed_decompositions import (
    orthogonalize,
    solve_core_seigh,
    solve_core_ssvd,
)
from .distributed_measurements import innermeas_idx_torch, ssrft_idx_torch

#
from .ssrft import SSRFT


# ##############################################################################
# # TRUNCATE CORE MATRIX
# ##############################################################################
def truncate_core(k, core_U, core_S, core_Vt=None):
    """Truncation of core sketches.

    Truncates core matrices as returned by :func:`ssvd` or :func:`seigh` down to
    rank ``k``. In SEIGH, ``core_Vt`` is the same as ``core_U.T``, so it can be
    omitted.

    :param int k: Truncation rank of the output core elements.
    :param core_U: Left core basis matrix of shape ``(outer_dim, outer_dim)``.
    :param core_S: Core singular values of shape ``(outer_dim,)``.
    :param core_Vt: If given, right core matrix analogous to ``core_U``.
    :returns: Truncated ``(core_U, core_S, core_Vt)`` such that ``core_S``
      retains the first ``k`` entries, and the core matrices are truncated
      accordingly. If ``core_Vt`` is none, only the first two elements are
      returned.
    """
    core_U = core_U[:, :k]
    core_S = core_S[:k]
    if core_Vt is not None:
        core_Vt = core_Vt[:k, :]
        return core_U, core_S, core_Vt
    else:
        return core_U, core_S


# ##############################################################################
# # SKETCHED SVD
# ##############################################################################
def ssvd(
    op,
    op_device,
    op_dtype,
    outer_dim,
    inner_dim,
    seed=0b1110101001010101011,
):
    """Sketched Singular Value Decomposition (SVD).

    Sketched SVD of any given linear operator, as introduced in
    `[TYUC2019, 2] <https://arxiv.org/abs/1902.08651>`_. As with any SVD, the
    operator does not have to be square, symmetric, or PSD.

    :param op: Linear operator that implements the left- and right matmul
      operators via ``@``, as well as the ``shape=(h, w)`` attribute, but it
      doesn't need to be in explicit matrix form.
    :param int outer_dim: Number of outer measurements to be performed.
    :param int inner_dim: Number of inner measurements to be performed. It
      should be larger than ``outer_dim``, typically twice is reccommended.
    :returns: ``(Q, U, S, V.T, P.T)``, so ``Q @ U @ diag(S) @ V.T @ P.T`` is a
      low-rank approximation of ``op``. This is an SVD since ``S`` contains the
      non-negative singular values (in non-ascending order), and ``Q, U``
      contain orthogonal columns with ``Q.shape = (h, outer_dim)``, and
      ``U.shape = (outer_dim, outer_dim)``.
    """
    # convenience params
    h, w = op.shape
    lo_seed, ro_seed, li_seed, ri_seed = seed, seed + 1, seed + 2, seed + 3
    # perform random left outer measurements and their QR
    lo_t = torch.empty((outer_dim, w), dtype=op_dtype).to(op_device)
    for i in range(outer_dim):
        ssrft_idx_torch(
            i,
            outer_dim,
            op,
            op_device,
            op_dtype,
            lo_t[i],
            seed=lo_seed,
            adjoint=True,
        )
    orthogonalize(lo_t.T, overwrite=True)
    # perform random right outer measurements and their QR
    ro = torch.empty((h, outer_dim), dtype=op_dtype).to(op_device)
    for i in range(outer_dim):
        ssrft_idx_torch(
            i,
            outer_dim,
            op,
            op_device,
            op_dtype,
            ro[:, i],
            seed=ro_seed,
            adjoint=False,
        )
    orthogonalize(ro, overwrite=True)
    # perform random inner measurements
    inner_meas = torch.empty((inner_dim, inner_dim), dtype=op_dtype).to(
        op_device
    )
    for i in range(inner_dim):
        innermeas_idx_torch(
            i,
            inner_dim,
            op,
            op_device,
            op_dtype,
            inner_meas[:, i],
            li_seed,
            ri_seed,
        )
    # Solve core op to yield initial approximation
    left_inner_ssrft = SSRFT((inner_dim, h), seed=li_seed)
    right_inner_ssrft = SSRFT((inner_dim, w), seed=ri_seed)
    core_U, core_S, core_Vt = solve_core_ssvd(
        ro, lo_t.T, inner_meas, left_inner_ssrft, right_inner_ssrft
    )
    #
    return ro, core_U, core_S, core_Vt, lo_t


# ##############################################################################
# # SKETCHED EIGH
# ##############################################################################
def seigh(
    op,
    op_device,
    op_dtype,
    outer_dim,
    inner_dim,
    seed=0b1110101001010101011,
):
    """Sketched Hermitian Eigendecomposition (EIGH).

    Sketched EIGH of any given linear operator, modified from the SSVD from
    `[TYUC2019, 2] <https://arxiv.org/abs/1902.08651>`_. It leverages Hermitian
    symmetry of the operator to require half the measurements, memory and
    arithmetic. The operator must be square and Hermitian, but doesn't have to
    be PSD.

    :param op: Hermitian linear operator that implements the left- and right
      matmul operators via ``@``, as well as the ``shape=(d, d)`` attribute,
      but it doesn't need to be in explicit matrix form.
    :param int outer_dim: Number of outer measurements to be performed.
    :param int inner_dim: Number of inner measurements to be performed. It
      should be larger than ``outer_dim``, typically twice is reccommended.
    :returns: ``(Q, U, S)``, so ``Q @ U @ diag(S) @ U.T @ Q.T`` is a
      low-rank approximation of ``op``. This is an EIGH since ``S`` contains the
      eigenvalues (in descending magnitude), and ``Q, U``
      contain orthogonal columns with ``Q.shape = (h, outer_dim)``, and
      ``U.shape = (outer_dim, outer_dim)``.
    """
    assert (
        inner_dim >= outer_dim
    ), "Can't have more outer than inner measurements!"
    # convenience params and square-matrix check
    li_seed, ri_seed = seed, seed + 1
    h, w = op.shape
    assert h == w, f"Seigh expects square matrices! {(h, w)}"
    # perform inner (and recycled outer) random measurements
    outer = torch.empty((h, outer_dim), dtype=op_dtype).to(op_device)
    inner = torch.empty((inner_dim, inner_dim), dtype=op_dtype).to(op_device)
    for i in range(inner_dim):
        out_buff = innermeas_idx_torch(
            i,
            inner_dim,
            op,
            op_device,
            op_dtype,
            inner[:, i],
            li_seed,
            ri_seed,
        )
        if i < outer_dim:
            outer[:, i] = out_buff
    # orthogonalize outer measurements
    orthogonalize(outer, overwrite=True)
    # Solve core op to yield initial approximation
    li_ssrft = SSRFT((inner_dim, h), seed=li_seed)
    ri_ssrft = SSRFT((inner_dim, w), seed=ri_seed)
    core_U, core_S = solve_core_seigh(outer, inner, li_ssrft, ri_ssrft)
    #
    return outer, core_U, core_S
