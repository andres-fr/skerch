#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to support sketched estimation of any (sub-)diagonals.

"""


import torch

from .distributed_decompositions import orthogonalize
from .linops import CompositeLinOp, NegOrthProjLinOp
from .ssrft import SSRFT

from .utils import rademacher_noise


# ##############################################################################
# # SKETCHED (SUB)-DIAGONAL ESTIMATOR
# ##############################################################################
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


# ##############################################################################
# # SKETCHED (SUB)-DIAGONAL LINOP
# ##############################################################################
def subdiag_hadamard_pattern(v, diag_idxs, use_fft=False):
    """Map random vector into a pattern for subdiagonal estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param subdiag_idxs: Iterator with integers corresponding to the subdiagonal
      indices to include, e.g. 0 corresponds to the main diagonal, 1 to the
      diagonal above, -1 to the diagonal below, and so on.
    :param use_fft: If false, shifted copies of ``v`` are pasted on the result.
      This requires only ``len(v)``  memory, but has ``len(v) * len(diag_idxs)``
      time complexity. If this argument is true, an FFT convolution is used
      instead. This requires at least ``4 * len(v)`` memory, but the arithmetic
      has a complexity of ``len(v) * log(len(v))``, which can be advantageous
      whenever ``len(diag_idxs)`` becomes very large.
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


def serrated_hadamard_pattern(v, chunksize, use_fft=False):
    """
    IMPLEMENT SERRATED PATTERN:
    * IF WITH FFT, DO A FULL TRIANGLE, AND THEN SUBTRACT THE BLOCKS
    * IF WITHOUT: grab a copy, progressively zero out idxs and add-shifted

    For example, given a 10-dimensional vector, and a serrated pattern with
    ``chunksize=3``, the returned vector will have the following entries:

    * ``v1``
    * ``v1 + v2``
    * ``v1 + v2 + v3``
    * ``v4``
    * ``v4 + v5``
    * ``v4 + v5 + v6``
    * ``v7``
    * ``v7 + v8``
    * ``v7 + v8 + v9``
    * ``v10``
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be a positive scalar!")
    #
    if use_fft:
        result = subdiag_hadamard_pattern(v, range(len(v)), use_fft=True)
        for i in range(0, len(v), chunksize):
            offset = sum(v[i : (i + chunksize)])
            result[(i + chunksize) :] -= offset
    else:
        result = v.clone()
        for i in range(1, chunksize):
            for j in range(0, i):
                target_len = len(result[i::chunksize])
                result[i::chunksize] += v[j::chunksize][:target_len]
    #
    return result


from .linops import BaseLinOp


"""
TODO:

1. Implement hadamard pattern for arbitrary chunks:
  * Given is a (supposedly random iid zero mean) vector
  * Returned is a vector of same shape
  *


Given:
* a pointer to a main linop with a shape
* optionally, a deflation matrix
*
* a set of diagonals


Note:
* If we wanna deflate, we rather do this before and pass the deflated linop
* If we wanna pick up the corners separately:
  * Apply a few extremal axis-aligned measurements,

we want a linop
"""


class DiagonalLinOp(BaseLinOp):
    r"""Diagonal linear operator.

    Given a vector ``v`` of ``d`` dimensions, this class implements a diagonal
    linear operator of shape ``(d, d)`` via left- and right-matrix
    multiplication, as well as the ``shape`` attribute, only requiring linear
    (:math:`\mathcal{O}(d)`) memory and runtime.
    """

    MAX_PRINT_ENTRIES = 20

    def __init__(self, diag):
        """:param diag: Vector to be casted as diagonal linop."""
        if len(diag.shape) != 1:
            raise BadShapeError("Diag must be a vector!")
        self.diag = diag
        super().__init__((len(diag),) * 2)  # this sets self.shape also

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=False)
        if len(x.shape) == 2:
            result = (x.T * self.diag).T
        else:
            # due to torch warning, can't transpose shapes other than 2
            result = x * self.diag
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=True)
        result = x * self.diag
        return result

    def __repr__(self):
        """Returns a string in the form <DiagonalLinOp(shape)[v1, v2, ...]>."""
        clsname = self.__class__.__name__
        diagstr = ", ".join(
            [str(x.item()) for x in self.diag[: self.MAX_PRINT_ENTRIES]]
        )
        if len(self.diag) > self.MAX_PRINT_ENTRIES:
            diagstr += "..."
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})[{diagstr}]>"
        return s
