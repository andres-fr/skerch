#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Sketched estimation of any (sub-)diagonals.


TODO:
* Test for subdiagonals and different shapes
* Extend test for linear combinations of subdiags (create function that maps)
"""


import torch

from .distributed_decompositions import orthogonalize
from .linops import CompositeLinOp, NegOrthProjLinOp
from .ssrft import SSRFT

from .utils import rademacher_noise


# ##############################################################################
# #
# ##############################################################################
def subdiag_hadamard_pattern(v, subdiag_idxs):
    """Map random vector into a pattern for subdiagonal estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param subdiag_idxs: Collection of integers corresponding to the
      subdiag indices to include.
    """
    # note: subdiag_idxs could be optimized to be an iterator and save memory
    breakpoint()

    v1 = torch.range(1, 20).to(torch.float32)
    v2 = torch.zeros_like(v1)
    for idx in diag_idxs:
        v2[idx] = 1
    # circular convolution
    V = torch.fft.rfft(v1)
    V *= torch.fft.rfft(v2)
    V = torch.fft.irfft(V)


def subdiagpp(
    num_meas,
    lop,
    lop_dtype,
    lop_device,
    seed=0b1110101001010101011,
    deflation_rank=0,
    diag_idx=0,
):
    """ """
    h, w = lop.shape
    assert h == w, "Only square linear operators supported!"
    dims = h
    abs_diag_idx = abs(diag_idx)
    result_buff = torch.zeros(
        dims - abs_diag_idx, dtype=lop_dtype, device=lop_device
    )
    deflation_matrix = None
    if num_meas > 0:
        squares_buff = torch.zeros_like(result_buff)
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
    # estimate deflated diagonal
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
                squares_buff += v[:-abs_diag_idx] * v[:-abs_diag_idx]
            elif diag_idx < 0:
                result_buff += (
                    v[abs_diag_idx:] * (v @ deflated_lop)[:-abs_diag_idx]
                )
                squares_buff += v[abs_diag_idx:] * v[abs_diag_idx:]
        result_buff /= squares_buff
    bottom_norm = result_buff.norm().item()
    # add estimated deflated diagonal to exact top-rank diagonal
    top_norm = 0
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
