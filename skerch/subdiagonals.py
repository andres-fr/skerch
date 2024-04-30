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
                # squares_buff += v * v
            elif diag_idx > 0:
                result_buff += (
                    v[:-abs_diag_idx] * (v @ deflated_lop)[abs_diag_idx:]
                )
            elif diag_idx < 0:
                result_buff += (
                    v[abs_diag_idx:] * (v @ deflated_lop)[:-abs_diag_idx]
                )
        # note that here we would typically apply result_buff /= squares_buff
        # but l2 norm of SSRFT rows is always 1, so no need.
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


def serrated_hadamard_pattern(
    v, chunksize, with_main_diagonal=True, use_fft=False
):
    """Auxiliary pattern for block-triangular estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param with_main_diagonal: If true, the main diagonal will be included
          in the patterns, otherwise excluded.

    For example, given a 10-dimensional vector, the corresponding serrated
    pattern with ``chunksize=3`` and with main diagonal will yield the
    following entries:

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

    If main diagonal is excluded, it will look like this instead:

    * ``0``
    * ``v1``
    * ``v1 + v2``
    * ``0``
    * ``v4``
    * ``v4 + v5``
    * ``0``
    * ``v7``
    * ``v7 + v8``
    * ``0``
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be a positive scalar!")
    #
    if use_fft:
        idxs = range(len(v)) if with_main_diagonal else range(1, len(v))
        result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
        for i in range(0, len(v), chunksize):
            offset = sum(v[i : (i + chunksize)])
            result[(i + chunksize) :] -= offset
    else:
        if with_main_diagonal:
            result = v.clone()
        else:
            result = torch.zeros_like(v)
        for i in range(1, chunksize):
            for j in range(0, i):
                target_len = len(result[i::chunksize])
                result[i::chunksize] += v[j::chunksize][:target_len]
    #
    return result


from .linops import BaseLinOp


class TriangularLinOp(BaseLinOp):
    """Given a full linear operator, compute products with one of its triangles.

    The triangle of a linear operator can be approximated from the full operator
    via a "staircase pattern" of exact measurements, whose computation is exact
    and fast. For example, 10 measurements in a ``(1000, 1000)`` operators
    yields one step covering ``lop[100:, :100]``, the next step covering
    ``lop[200, 100:200]``, and so on. The more measurements, the more closely
    the full triangle is approximated. Then, the linear operation including
    the remaining steps (leftovers from the staircase pattern) are then
    estimated with the help of  :fun:`serrated_hadamard_pattern`, completing
    the triangular approximation.
    """

    def __init__(
        self,
        lop,
        num_staircase_measurements=10,
        num_serrated_measurements=0,
        lower=True,
        with_main_diagonal=True,
        seed=0b1110101001010101011,
    ):
        """
        :param lop: A square linear operator of order ``dims``, such that
          ``self @ v`` will equal ``triangle(lop) @ v``.
        :param num_staircase_measurements: The exact part of the triangular
          approximation, comprising measurements following a "staircase"
          pattern. Runtime grows linearly with this parameter. Memory is
          unaffected. Approximation error shrinks with ``1 / measurements``.
          It is recommended to have this as high as possible.
        :param num_serrated_measurements: The leftover entries from the
          staircase measurements are approximated here using an extension of
          the Hutchinson diagonal estimator. This estimator generally requires
          many measurements to be informative, and it can even be counter-
          productive if not enough measurements are given. If the staircase
          measurements are close enough, consider setting this to 0. Otherwise
          make sure to provide a sufficiently high number of measurements.
        :param lower: If true, the lower triangle will be computed. Otherwise
          the upper triangle.
        :param with_main_diagonal: If true, the main diagonal will be included
          in the operator, otherwise excluded. If you already have precomuted
          the diagonal elsewhere, consider excluding it from this approximation,
          and adding it separately.
        :param seed: Seed for the random SSRFT measurements used in the
          serrated estimator.
        """
        h, w = lop.shape
        assert h == w, "Only square linear operators supported!"
        self.dims = h
        self.lop = lop
        self.n_serrat = num_serrated_measurements
        assert (
            num_staircase_measurements <= self.dims
        ), "More staircase measurements than dimensions??"
        assert (
            self.n_serrat <= self.dims
        ), "More serrated measurements than dimensions??"
        assert (
            num_staircase_measurements + self.n_serrat
        ) <= self.dims, "More total measurements than dimensions??"

        self.lower = lower
        self.with_main = with_main_diagonal
        self.stair_steps, self.dims_serrat = self._get_chunk_dims(
            self.dims, num_staircase_measurements
        )
        self.ssrft = SSRFT((self.n_serrat, self.dims), seed=seed)
        super().__init__(lop.shape)  # this sets self.shape also

    @staticmethod
    def _get_chunk_dims(dims, num_stair_meas):
        """ """
        div, _ = divmod(dims, num_stair_meas)
        stair_steps = torch.tensor([div] * num_stair_meas).cumsum(0)
        serrated_dims = div
        return stair_steps, serrated_dims

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=False)
        #
        buff = torch.zeros_like(x)
        result = torch.zeros_like(x)
        #
        beg = 0
        for step in self.stair_steps:
            # prepare entry buffer and apply to linop
            end = step
            buff[beg:end] = x[beg:end]
            result[step:] += (self.lop @ buff)[step:]
            # reset buffer and prepare next measurement
            buff[beg:end] = 0
            beg = end
        #
        if self.n_serrat > 0:
            buff_v = torch.empty_like(buff)
            buff *= 0
        for i in range(self.n_serrat):
            buff_v[:] = self.ssrft.get_row(i, x.dtype, x.device)
            buff[:] += serrated_hadamard_pattern(buff_v, self.dims_serrat) * (
                self.lop @ (buff_v * x)
            )
        return result + buff

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=True)
        breakpoint()

    def __repr__(self):
        """Readable string version of this object.

        Returns a string in the form
        <TriangularlLinOp[lop](lower/upper)>.
        """
        clsname = self.__class__.__name__
        lopstr = self.lop.__repr__()
        typestr = "lower" if self.lower else "upper"
        result = f"<{clsname}[{lopstr}]({typestr})>"
        return result


"""
TODO:

* Refactor diag estimator to work with general Hadamards and SSRFT.
* Finish tri linop, accept also upper and left-matmul. utest
  * factor out staircase and serrated into their submethods
* Add submatrix linop, and utest

Check that we are indeed ready to do our experiments, or do we need more?
What about LR or diagonal deflations? we would need a "subtraction" linop?


* we may want to implement a submatrix operator
* deflation is a thing! is there a way to translate arbitrary Hadamard
  patterns into the deflated projector?
"""
