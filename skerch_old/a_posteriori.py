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

from .utils import gaussian_noise


################################################################################
# # A POSTERIORI ERROR ESTIMATION
# ##############################################################################
def a_posteriori_error_bounds(num_measurements, rel_err):
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
    assert 0 <= rel_err <= 1, "rel_err expected between 0 and 1"
    experr = math.exp(rel_err)
    meas_half = num_measurements / 2
    #
    lo_p = (experr * (1 - rel_err)) ** meas_half
    hi_p = (experr / (1 + rel_err)) ** (-meas_half)
    #
    result = {
        f"P(err<={1 - rel_err}x)": lo_p,
        f"P(err>={1 + rel_err}x)": hi_p,
    }
    return result


def a_posteriori_error(
    lop1,
    lop2,
    num_measurements,
    seed=0b1110101001010101011,
    dtype=torch.float64,
    device="cpu",
    adjoint=False,
):
    r"""A-posteriori sketch error estimation.

    This function implements the error estimation procedure presented in
    `[TYUC2019, 6] <https://arxiv.org/abs/1902.08651>`_.
    For that, it performs ``num_measurements`` gaussian vector
    multiplications on linear operators ``lop1`` and ``lop2``, and then
    compares the results.
    The :math:`\ell_2` error between the obtained measurements is then an
    estimation of the Frobenius error between ``lop1`` and ``lop2``.

    :param lop1: Linear operator with a ``shape=(h, w)`` attribute and
      implementing a left (``x @ mat``) or right ``(mat @ x)`` matmul op.
      It is assumed to be compatible with the given torch ``dtype``.
    :param lop2: See ``lop1``.
    :param adjoint: If true, left-matmul is used, otherwise right-matmul.
    :returns: A tuple ``((f1_mean, f2_mean, diff_mean), (f1, f2, diff))``,
      where the last 3 elements are lists with the ``L_2^2`` norms of each
      random measurement for the first matrix (f1), second matrix (f2) and
      difference between matrices (diff). The first 3 elements are the averages
      over all given measurements. The final estimate for the error between
      ``lop1`` and ``lop2`` is then ``diff_mean``. To get confidence bounds
      on this estimate, use ``a_posteriori_error_bounds``.
    """
    assert lop1.shape == lop2.shape, "Mismatching shapes!"
    h, w = lop1.shape
    #
    frob1_pow2, frob2_pow2, diff_pow2 = [], [], []
    for i in range(num_measurements):
        rand = gaussian_noise(
            h if adjoint else w,
            mean=0.0,
            std=1.0,
            seed=seed + i,
            dtype=dtype,
            device=device,
        )
        rand_np = None
        if adjoint:
            meas1 = rand @ lop1
            meas2 = rand @ lop2
        else:
            try:
                meas1 = lop1 @ rand
            except TypeError:
                rand_np = rand.cpu().numpy() if rand_np is None else rand_np
                meas1 = torch.from_numpy(lop1 @ rand_np)
            try:
                meas2 = lop2 @ rand
            except TypeError:
                rand_np = rand.cpu().numpy() if rand_np is None else rand_np
                meas2 = torch.from_numpy(lop2 @ rand_np)
        #
        frob1_pow2.append(sum(meas1**2).item())
        frob2_pow2.append(sum(meas2**2).item())
        diff_pow2.append(sum((meas1 - meas2) ** 2).item())
    #
    frob1_pow2_mean = sum(frob1_pow2) / num_measurements
    frob2_pow2_mean = sum(frob2_pow2) / num_measurements
    diff_pow2_mean = sum(diff_pow2) / num_measurements
    return (
        (frob1_pow2_mean, frob2_pow2_mean, diff_pow2_mean),
        (frob1_pow2, frob2_pow2, diff_pow2),
    )


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

    :param S: Vector of the estimated spectrum/singular values, expected to
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
    assert (
        abs(S).diff() <= 0
    ).all(), "Provided S must be given in non-ascending magnitude!"
    S_squared = S**2
    # residuals is a vector of len(S), where the ith entry contains the
    # Frobenius norm of M' after an ith-rank deflation, while progressing by
    # descending magnitude. The first entry has a rank 0 deflation, i.e. it
    # contains frob(M') entirely. The last entry has a rank R-1 deflation, i.e.
    # it is the magnitude of the smallest value in S.
    residuals = S_squared.flip(0).cumsum(0).flip(0) ** 0.5
    # now instead of normalizing dividing by sum(S_squared)**0.5, which uses M',
    # we obtain both the lower and upper bounds by using M instead, this is
    # given by ori_frob.
    lo_scree = (residuals / ori_frob) ** 2
    hi_scree = ((residuals + err_frob) / ori_frob) ** 2
    return lo_scree, hi_scree
