#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""A-posteriori utilities for sketched decompositions.

Given a (potentially matrix-free) linear operator, and our obtained low-rank
decomposition, we would like to know how similar they are. In
`[TYUC2019] <https://arxiv.org/abs/1902.08651>`_, a method is presented to
estimate the Frobenius residual error. Furthermore, a probabilistic bound to
said estimator is presented, providing confidence bounds for the estimation.
More measures increase said confidence.

Finally, a "scree plot" method is also introduced, providing upper and lower
bounds to estimate the resudual Frobenius error as a function of the rank.
This can be used to estimate the actual rank of our original linear operator,
and to choose the rank of our recovery.

This module implements this functionality.
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

    Retrieves probabilistic error bounds presented in
    `[TYUC2019, 6.4] <https://arxiv.org/abs/1902.08651>`_, see module docstring
    for details. Note that this function does not perform the error estimation,
    it merely informs about how noisy would be the estimation given the desired
    number of measurements.

    :param int num_measurements: How many Gaussian measurements will be
      performed for the a-posteriori error estimation. More measurements means
      tighter error bounds.
    :param float rel_err: A float between 0 and 1, indicating the relative
      error that we want to consider. If ``x`` is the actual error we want to
      estimate, and ``y`` is our estimation, this function will return the
      probability that ``y`` is in the range ``[(1-rel_err)x, (1+rel_err)x]``.
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
    r"""A-posteriori sketch error estimation.

    This function implements the error estimation procedure discussed in
    `[TYUC2019, 6] <https://arxiv.org/abs/1902.08651>`_.

    It performs ``num_measurements`` random vector multiplications on linear
    operators ``lop1`` and ``lop2``, resulting in estimators for their
    respective Frobenius norms, as well as for their difference.

    .. note::
      The test measurements performed here are assumed to be random and
      independent of how either ``lop`` was obtained. For this reason, it is
      advised to pick a combination of ``noise_type`` and ``seed`` that does
      not overlap with anything used before.

    :param lop1: Linear operator with a ``shape=(h, w)`` attribute and
      implementing a left (``x @ mat``) or right ``(mat @ x)`` matmul op.
      It is assumed to match the given ``device`` and ``dtype``.
    :param lop2: Linear operator to compare ``lop1`` against. It must match
      in shape, device and dtype.
    :param num_meas: Number of measurements that will be performed on ``lop1``
      and ``lop2``.
    :param adjoint: If true, left-matmul is used, otherwise right-matmul.
    :returns: A tuple ``((f1_mean, f2_mean, diff_mean), (f1, f2, diff))``,
      where the last 3 elements are lists with the ``L_2^2`` norms of each
      random measurement for the first matrix (f1), second matrix (f2) and
      difference between matrices (diff). The first 3 elements are the averages
      over all given measurements. The final estimate for the error between
      ``lop1`` and ``lop2`` is then ``diff_mean``. To get confidence bounds
      on this estimate, use ``a_posteriori_error_bounds``.
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
def scree_bounds(S, ori_frob, err_frob):
    """A-posteriori scree bounds for low-rank approximations.

    Reference: `[TYUC2019, 6.5] <https://arxiv.org/abs/1902.08651>`_.

    When we do a sketched eigen- or singular-decomposition of a linear operator,
    we don't generally know a-priori where is the effective rank of said
    operator. A scree analysis is an a-posteriori tool to estimate the smallest
    rank that captures a sufficient amount of the actual matrix.

    It does so by efficiently retrieving upper and lower bounds for the actual
    proportion of residual energy that remains, as we increase the rank of our
    estimated matrix. Typically we look for an initial sharp decay, followed
    by an "elbow" at a low point, after which residual energy stops decaying
    fast. The location of the elbow is a good indicator for the effective rank,
    but quantitative methods are also possible (see [TYUC2019]).

    :param S: Vector of the estimated eigen/singular values, expected to
      contain entries in non-ascending magnitude.
    :param ori_frob: Estimate (or exact if available) ``frob_norm(M)``, where
      ``M`` is the original linop that we are decomposing. This quantity is
      estimated by :func:`.a_posteriori_error`, but note that it returns
      ``frob_norm^2``, while this function requires the Frobenius norm itself.
    :param err_frob: A-posteriori estimate of ``frob_norm(M - M')``, where
      ``M'`` is the recovery of ``M`` with singular values ``S``.

    Usage example::

      f1_pow2, f2_pow2, res_pow2 = a_posteriori_error(M, M', ...)
      S = get_singular_values(M')  # in non-ascending magnitude
      scree_lo, scree_hi = scree_bounds(S, f1_pow2**0.5, res_pow2**0.5)
      # plot the scree bounds to find an elbow as a cutting point for rank(M).
    """
    if (abs(S).diff() > 0).any():
        raise ValueError(
            "Provided S must be given in non-ascending magnitude!"
        )
    # residuals is a vector of len(S), where the ith entry contains the
    # Frobenius norm of M' after an ith-rank deflation, while progressing by
    # descending magnitude. The first entry has a rank 0 deflation, i.e. it
    # contains frob(M') entirely. The last entry has a rank R-1 deflation, i.e.
    # it is the magnitude of the smallest value in S.
    residuals = (S**2).flip(0).cumsum(0).flip(0) ** 0.5
    # now instead of normalizing dividing by sum(S_squared)**0.5, which uses M',
    # we obtain both the lower and upper bounds by using M instead, this is
    # given by ori_frob.
    lo_scree = (residuals / ori_frob) ** 2
    hi_scree = ((residuals + err_frob) / ori_frob) ** 2
    return lo_scree, hi_scree
