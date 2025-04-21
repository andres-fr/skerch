#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to support sketched estimation of any (sub-)diagonals.

Including:

* Hadamard pattern to compute subdiagonals as ``mean(hadamard(v) * lop @ v)``,
  for ``v_i = SSRFT[i, :]``.
* In-core routine to compute arbitrary subdiagonals of linops.
"""


import torch

from .distributed_decompositions import orthogonalize, pinvert
from .linops import CompositeLinOp, NegOrthProjLinOp
from .ssrft import SSRFT


# ##############################################################################
# # HADAMARD
# ##############################################################################
def subdiag_hadamard_pattern(v, diag_idxs, use_fft=False):
    r"""Shifted copies of vectors for subdiagonal Hutchinson estimation.

    Given a square linear operator :math:`A`, and random vectors
    :math:`v \sim \mathcal{R}` with :math:`\mathbb{E}[v v^T] = I`, consider
    this generalized formulation of the Hutchinson diagonal estimator:

    .. math::

      f(A) =
      \mathbb{E}_{v \sim \mathcal{R}} \big[ \varphi(v) \odot Av \big]

    If the :math:`\varphi` function is the identity, then :math:`f(A)` equals
    the main diagonal of :math:`A`. If e.g. :math:`\varphi` shifts the entries
    in :math:`v` downwards by ``k`` positions, we get the ``k``-th subdiagonal.

    .. seealso::

      `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston and Yuji
      Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds
      and an improved algorithm”*.  CoRR abs/2201.10684.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param diag_idxs: Iterator with integers corresponding to the subdiagonal
      indices to include, e.g. 0 corresponds to the main diagonal, 1 to the
      diagonal above, -1 to the diagonal below, and so on.
    :param use_fft: If false, shifted copies of ``v`` are pasted on the result.
      This requires only ``len(v)``  memory, but has ``len(v) * len(diag_idxs)``
      time complexity. If this argument is true, an FFT convolution is used
      instead. This requires at least ``4 * len(v)`` memory, but the arithmetic
      has a complexity of ``len(v) * log(len(v))``, which can be advantageous
      whenever ``len(diag_idxs)`` becomes very large.
    :returns: A vector of same shape, type and device as ``v``, composed of
      shifted copies of ``v`` as given by ``diag_idxs``.
    """
    len_v = len(v)
    if use_fft:
        # create a buffer of zeros to avoid circular conv and store the
        # convolutional impulse response
        buff = torch.zeros(2 * len_v, dtype=v.dtype, device=v.device)
        # padded FFT to avoid circular convolution
        buff[:len_v] = v
        V = torch.fft.rfft(buff)
        # now we can write the impulse response on buff
        buff[:len_v] = 0
        for idx in diag_idxs:
            buff[idx] = 1
        # non-circular FFT convolution:
        V *= torch.fft.rfft(buff)
        V = torch.fft.irfft(V)[:len_v]
        return V
    else:
        result = torch.zeros_like(v)
        for idx in diag_idxs:
            if idx == 0:
                result += v
            elif idx > 0:
                result[idx:] += v[:-idx]
            else:
                result[:idx] += v[-idx:]
        return result


# ##############################################################################
# # SKETCHED (SUB)-DIAGONAL ESTIMATORS
# ##############################################################################
def subdiagpp(
    lop,
    num_meas,
    lop_dtype,
    lop_device,
    seed=0b1110101001010101011,
    deflation_rank=0,
    diag_idx=0,
):
    """Matrix-free (sub-) diagonal sketched estimator via Diag++.

    .. seealso::

      Reference for this algorithm:

      `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston and Yuji
      Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds
      and an improved algorithm”*.  CoRR abs/2201.10684.

    Given a linear operator ``lop``, implements an unbiased diagonal estimator,
    composed of orthogonal deflation followed by Hutchinson estimation.

    The (optional) deflation is based on efficiently obtaining a thin matrix of
    orthonormal columns ``Q``, such that:

    ``lop = (Q @ Q.T @ lop) + ((I - Q@Q.T) @ lop)``

    Then, the (sub-) diagonal entries for the first component can be computed
    exactly and efficiently by picking rows of ``Q``. The second, "deflated"
    term is then subject to Hutchinson estimation:

    ``diag_k(defl_lop) = expectation(hadamard_k(v) * defl_lop @ v)``,

    This is achieved by sampling random ``v`` vectors from a SSRFT operator,
    and shifting them using :func:`subdiag_hadamard_pattern` to capture the
    desired ``k``-th subdiagonal.

    The final estimation is then the sum of the exact first component plus the
    estimated, deflated component.

    :param lop: The linear operator to extract (sub-) diagonals from. It must
      implement a ``lop.shape = (h, w)`` attribute as well as the left- and
      right- matmul operator ``@``, interacting with torch tensors.
    :param num_meas: Number of samples ``v`` used to compute the expectation.
    :param lop_dtype: Datatype of ``lop``.
    :param lop_device: Device of ``lop``.
    :param seed: Random seed for the ``v`` random SSRFT vectors.
    :param deflation_rank: Rank of the deflation matrix ``Q`` to be computed.
    :param diag_idx: Position of the (sub-) diagonal to compute. 0 corresponds
      to the main diagonal, +1 to the diagonal above, -1 to the diagonal below,
      and so on. Note that if diagonals are too far away from the main diagonal
      (or if ``lop`` is small enough), it may be worth it to directly perform
      one exact measurement per diagonal entry and not use ``Diag++`` at all.
    :returns: The tuple ``(result, defl, (top_norm, bottom_norm))``. The first
      element is the estimated diagonal. The second is the computed deflation
      matrix, or ``None`` if ``deflation_rank=0``. The last element is the
      Euclidean norm of the deflation and the deflated parts of the diagonal,
      respectively, which can be used to roughly diagnose the contribution of
      each part of the algorithm.
    """
    h, w = lop.shape
    if h != w:
        raise ValueError("Only square linear operators supported!")
    dims = h
    abs_diag_idx = abs(diag_idx)
    result_buff = torch.zeros(
        dims - abs_diag_idx, dtype=lop_dtype, device=lop_device
    )
    deflation_matrix = None
    # first compute top-rank orth proj to deflate lop
    if deflation_rank > 0:
        # deflate lop: first compute a few random measurements
        ssrft_defl = SSRFT((deflation_rank, dims), seed=seed + 100)
        deflation_matrix = torch.empty(
            (dims, deflation_rank), dtype=lop_dtype, device=lop_device
        )
        for i in range(deflation_rank):
            deflation_matrix[:, i] = lop @ ssrft_defl.get_row(
                i, lop_dtype, lop_device
            )
        # orthogonalize measurements to get deflated lop
        orthogonalize(deflation_matrix, overwrite=True)
        negproj = NegOrthProjLinOp(deflation_matrix)
        deflated_lop = CompositeLinOp((("negproj", negproj), ("lop", lop)))
    else:
        # no deflation
        deflated_lop = lop
    # estimate deflated diagonal. We don't use subdiag_hadamard_pattern
    # because this direct way is more memory-efficient.
    squares_buff = torch.zeros_like(result_buff)
    if num_meas > 0:
        ssrft = SSRFT((num_meas, dims), seed=seed)
        for i in range(num_meas):
            v = ssrft.get_row(i, lop_dtype, lop_device)
            if diag_idx == 0:
                result_buff += v * (v @ deflated_lop)
                squares_buff += v * v
            elif diag_idx > 0:
                result_buff += (
                    v[:-abs_diag_idx] * (v @ deflated_lop)[abs_diag_idx:]
                )
            elif diag_idx < 0:
                result_buff += (
                    v[abs_diag_idx:] * (v @ deflated_lop)[:-abs_diag_idx]
                )
        #
        result_buff /= squares_buff
    bottom_norm = result_buff.norm().item()
    # add estimated deflated diagonal to exact top-rank diagonal
    top_norm = 0
    buf2 = result_buff.clone()
    if deflation_rank > 0:
        for i in range(len(result_buff)):
            row = i if (diag_idx > 0) else i + abs_diag_idx
            col = i if (diag_idx <= 0) else i + abs_diag_idx
            entry = ((deflation_matrix @ deflation_matrix[row]) @ lop)[col]
            result_buff[i] += entry
            top_norm += entry**2
        top_norm = top_norm**0.5
    #
    return result_buff, deflation_matrix, (top_norm, bottom_norm)


def xdiag(
    lop,
    num_meas,
    lop_dtype,
    lop_device,
    seed=0b1110101001010101011,
    with_variance=True,
):
    """Matrix-free (sub-) diagonal sketched estimator via XTrace.

    .. seealso::

      Reference for this algorithm:

      `[ETW2024] <https://arxiv.org/pdf/2301.07825>`_: Ethan N. Epperly,
      Joel A. Tropp, Robert J. Webber. 2024. *“XTrace: Making the most of every
      sample in stochastic trace estimation”*. SIAM Journal on Matrix Analysis
      and Applications.
    """
    h, w = lop.shape
    if h != w:
        raise ValueError("Only square linear operators supported!")
    if num_meas % 2 != 0:
        raise ValueError("num_meas must be multiple of 2!")
    half_meas = num_meas // 2
    dims = h
    # compute and orthogonalize random measurements
    rand_lop = SSRFT((half_meas, dims), seed=seed + 100)
    # from .utils import rademacher_noise

    # rand_lop = (
    #     rademacher_noise((dims, half_meas), seed=seed + 100, device="cpu")
    #     .to(lop_dtype)
    #     .to(lop_device)
    # )
    meas = torch.empty((dims, half_meas), dtype=lop_dtype, device=lop_device)
    for i in range(half_meas):
        meas[:, i] = lop @ rand_lop.get_row(i, lop_dtype, lop_device)
        # meas[:, i] = lop @ rand_lop[:, i]
    Q, R = orthogonalize(meas, overwrite=False, return_R=True)
    # compute Q-top diagonal as preliminary result (efficient and exact)
    Qt_lop = torch.empty_like(Q.T)
    for i in range(half_meas):
        Qt_lop[i, :] = Q[:, i] @ lop
    top_result = (Q * Qt_lop.T).sum(1)
    # compute S-matrix, needed to form the estimators via rank-1 deflations
    S = pinvert(R.T)
    S /= torch.linalg.norm(S, dim=0)
    # refine preliminary result with rank-1 deflations
    top_negproj = torch.zeros_like(top_result)
    numerator = torch.zeros_like(top_result)
    denominator = torch.zeros_like(top_result)
    if with_variance:
        var_buffer1 = torch.zeros_like(Q)
        var_buffer2 = torch.zeros_like(Q)
    for i in range(half_meas):
        v_i = rand_lop.get_row(i, lop_dtype, lop_device)
        # v_i = rand_lop[:, i]
        s_i = S[:, i]
        #
        # Qt_lop_vi = Qt_lop @ v_i
        # defl = Qt_lop_vi - (s_i.T @ Qt_lop_vi) * s_i
        # numer_i = v_i * (meas[:, i] - Q @ defl)
        # numerator += numer_i
        # denominator += v_i * v_i

        #
        right_i = Qt_lop @ v_i
        left_i = s_i @ Qt_lop
        #
        negproj = left_i * (Q @ s_i)
        top_negproj -= negproj
        numer = v_i * (meas[:, i] - (Q @ (right_i - (s_i.T @ right_i) * s_i)))
        numerator += numer
        denominator += v_i * v_i
        #
        if with_variance:
            var_buffer1[:, i] = top_result - negproj
            var_buffer2[:, i] = numer
    #
    result = top_result + (top_negproj / half_meas)
    result += numerator / denominator
    var = (
        (var_buffer1 + (half_meas * var_buffer2.T / denominator).T).var(1)
        if with_variance
        else None
    )
    return result, (Q, S, rand_lop), var
