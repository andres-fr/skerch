#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""A-posteriori utilities for sketched decompositions.

This module implements scalable utilities to assess the quality of a
sketched approximation, once it has been obtained.
"""

import math
import torch
from .utils import gaussian_noise, COMPLEX_DTYPES, complex_dtype_to_real
from .algorithms import SketchedAlgorithmDispatcher


################################################################################
# # A POSTERIORI ERROR ESTIMATION
# ##############################################################################
def apost_error_bounds(num_measurements, rel_err, is_complex=False):
    """Probabilistic bounds for a-posteriori sketch error estimation.


    The Frobenius error between any two linear operators can be estimated
    from sketches using :func:`apost_error`. But this estimation is random
    and subject to error.

    This function retrieves the probabilistic error bounds presented in
    `[TYUC2019, 6.4] <https://arxiv.org/abs/1902.08651>`_, which tell us
    the probability that the estimated error is wrong by a given amount.
    Interestingly, these bounds only depend on the number of measurements
    and data type, and *not* on the operator dimension.

    .. note::

      This function does not perform the error estimation itself, it merely
      informs about the probabilistic bounds associated with a given error.
      For the error estimation, see :func:`apost_error`.

    :param int num_measurements: How many measurements (assumed Gaussian iid)
      will be performed for the a-posteriori error estimation. More
      measurements means tighter error bounds (i.e. the output of our error
      estimation is more reliable).
    :param float rel_err: A float between 0 and 1, indicating the relative
      error that we want to consider. If ``x`` is the actual error we want to
      estimate, and ``y`` is our estimation, this function will return the
      probability that ``y`` is in the range ``[(1-rel_err)x, (1+rel_err)x]``.
    :param is_complex: The returned probabilities will be tighter if the
      error is measured on complex linear operators, using complex Gaussian
      noise. This boolean flag allows to specify this (assumed real-valued if
      false).
    :returns: Probabilities that, given the indicated number
      of measurements, the a-posteriori method yields values outside of
      ``actual_error * (1 +/- rel_err)``. Ideally, we want a sufficient number
      of measurements, such that the returned probabilities for small
      ``rel_err`` are themselves small (this means that the corresponding error
      estimation is tight).
    """
    if num_measurements < 1:
        raise ValueError("Num measurements must be >=1")

    if rel_err < 0 or rel_err > 1:
        raise ValueError("rel_err expected between 0 and 1")
    beta = 2 if is_complex else 1
    experr = math.exp(rel_err)
    beta_meas_half = beta * num_measurements / 2
    #
    lo_p = (experr * (1 - rel_err)) ** beta_meas_half
    hi_p = (experr / (1 + rel_err)) ** (-beta_meas_half)
    #
    result = {
        f"LOWER: P(err<={1 - rel_err}x)": lo_p,
        f"HIGHER: P(err>={1 + rel_err}x)": hi_p,
    }
    return result


def apost_error(
    lop1,
    lop2,
    device,
    dtype,
    num_meas=5,
    seed=0b1110101001010101011,
    noise_type="gaussian",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    adj_meas=None,
):
    r"""A-posteriori sketched error estimation.

    This function implements the error estimation procedure discussed in
    `[TYUC2019, 6] <https://arxiv.org/abs/1902.08651>`_.
    Given two linear operators :math:`A, \hat{A}`, it performs
    ``num_measurements`` random vector multiplications
    :math:`y_i = Av_i, \hat{y}_i = \hat{A}v_i`, and the :math:`y` outputs can
    then be used to estimate the norms
    :math:`\lVert A \rVert_F^2, \lVert \hat{A} \rVert_F^2`, as well as the
    error :math:`\lVert A - \hat{A} \rVert_F`.

    .. note::
      The test measurements performed here are assumed to be random and
      independent of how either ``lop`` was obtained. For this reason, it is
      important to pick a combination of ``noise_type`` and ``seed`` that does
      not overlap with any other noise generation procedure used before.

    :param lop1: Linear operator with a ``shape=(h, w)`` attribute and
      implementing a left (``x @ mat``) or right ``(mat @ x)`` matmul op.
      It is assumed to match the given ``device`` and ``dtype``.
    :param lop2: Linear operator to compare ``lop1`` against. It must match
      in shape, device and dtype.
    :param num_meas: Number of test measurements that will be performed on
      ``lop1`` and ``lop2``.
    :param adjoint: If true, left-matmul is used, otherwise right-matmul.
    :returns: A tuple ``((f1, f2, e), (F1, F2, E))``, where ``f1, f2`` are the
      estimated Frobenius norms squared of ``lop1, lop2`` and ``e`` is the
      estimated Frobenius error squared. The uppercase counterparts are lists
      containing all ``num_meas`` individual measurements (which are averaged
      to obtain the estimates). To get confidence bounds on this estimate,
      use :func:`apost_error_bounds`.
    """
    h, w = lop1.shape
    h2, w2 = lop2.shape
    if lop1.shape != lop2.shape:
        raise ValueError("Linear operators must be of same shape!")
    if num_meas <= 0 or num_meas > min(h, w):
        raise ValueError(
            "Measurements must be between 1 and number of rows/columns!"
        )
    if meas_blocksize is None:
        meas_blocksize = num_meas
    if adj_meas is None:
        adj_meas = h > w  # this seems to be more accurate
    is_complex = dtype in COMPLEX_DTYPES
    beta = 2 if is_complex else 1
    #
    rdtype = complex_dtype_to_real(dtype)
    frob1_all = torch.empty(num_meas, dtype=rdtype, device=device)
    frob2_all = torch.empty_like(frob1_all)
    dist_all = torch.empty_like(frob1_all)
    mop = dispatcher.mop(
        noise_type,
        (h if adj_meas else w, num_meas),
        seed,
        dtype,
        meas_blocksize,
        register=False,
    )
    #
    for block, idxs in mop.get_blocks(dtype, device):
        # we need that block @ block.H approximates I
        # assuming block is by-column!
        block *= 1000
        bnorm = block.norm(dim=1)
        block = block.T * ((len(idxs) ** 0.5) / block.norm(dim=1))
        # block has been transposed
        meas1 = block @ lop1 if adj_meas else lop1 @ block.T
        meas2 = block @ lop2 if adj_meas else lop2 @ block.T
        #
        frob1 = (meas1 * meas1.conj()).real.sum(dim=1 if adj_meas else 0)
        frob2 = (meas2 * meas2.conj()).real.sum(dim=1 if adj_meas else 0)
        meas1 -= meas2
        dist = (meas1 * meas1.conj()).real.sum(dim=1 if adj_meas else 0)
        #
        frob1_all[idxs] = frob1
        frob2_all[idxs] = frob2
        dist_all[idxs] = dist
    #
    f1, f2, d = frob1_all.mean(), frob2_all.mean(), dist_all.mean()
    return ((f1, f2, d), (frob1_all, frob2_all, dist_all))


# ##############################################################################
# # SCREE
# ##############################################################################
def scree_bounds(S, err_frob):
    """A-posteriori scree bounds for low-rank approximations.

    Reference: `[TYUC2019, 6.5] <https://arxiv.org/abs/1902.08651>`_.

    When we do a sketched eigen- or singular-decomposition of a linear
    operator, we often don't know what is the effective rank of said operator,
    and whether we took enough sketched measurements to cover it.
    A scree analysis is a tool to estimate the smallest rank that captures
    a sufficient amount of the actual matrix, typically by looking at
    inflection points in the curve associated with sharp decays in
    spectral norm.

    A common issue with sketched methods is that we don't have access to the
    actual spectrum, and thus we can't perform a scree analysis. Instead,
    it is possible to use estimates from :func:`apost_error` to retrieve
    upper and lower scree bounds, which ideally would still inform us about
    the effective rank of the original operator.

    :param S: Vector of the estimated eigen/singular values, expected to
      contain entries in non-ascending magnitude.
    :param err_frob: A-posteriori estimate of ``frob_norm(M - M')``, where
      ``M'`` is the recovery of ``M`` with singular values ``S``, as returned
      by :func:`apost_error` (note that here the norm is not squared).

    Usage example::

      U, S, Vh = ssvd(A)
      Ahat = CompositeLinOp((("US", U * S), ("Vh", Vh)))
      (f1, _, err), _ = apost_error(A, Ahat, "...")
      scree_lo, scree_hi = scree_bounds(S, err**0.5)
    """
    if (abs(S).diff() > 0).any():
        raise ValueError(
            "Provided S must be given in non-ascending magnitude!"
        )
    approx_norm = S.norm()
    # Nonincreasing curve with the (estimated) residual energies, going from
    # approx_norm down to 0
    residuals = (S**2).flip(0).cumsum(0).flip(0) ** 0.5
    # the lower scree bound is assumed to be given by the normalized
    # residuals, since the least-squares recovery scree tends to decay faster
    # than hte actual scree. Note that we divide by approx_norm, because the
    # lower bound must start at 1 (dividing estimated residuals by original
    # norm causes a mismatch and the resulting curve is not a scree)
    lo_scree = (residuals / approx_norm) ** 2
    # now the upper scree bound is given by adding the estimated error on
    # top of the lower bound.
    hi_scree = ((residuals + err_frob) / approx_norm) ** 2
    #
    return lo_scree, hi_scree
