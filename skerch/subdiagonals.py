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
    v, chunksize, with_main_diagonal=True, lower=True, use_fft=False
):
    """Auxiliary pattern for block-triangular estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param with_main_diagonal: If true, the main diagonal will be included
          in the patterns, otherwise excluded.

    For example, given a 10-dimensional vector, the corresponding serrated
    pattern with ``chunksize=3, with_main_diagonal=True, lower=True`` yields
    the following entries:

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

    And if ``lower=False``, it will look like this instead:

    * ``v1``
    * ``v4 + v3 + v2``
    * ``v4 + v3``
    * ``v4``
    * ``v7 + v6 + v5``
    * ``v7 + v6``
    * ``v7``
    * ``v10 + v9 + v8``
    * ``v10 + v9``
    * ``v10``
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be a positive scalar!")
    #
    len_v = len(v)
    if use_fft:
        if lower:
            idxs = range(len_v) if with_main_diagonal else range(1, len_v)
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, chunksize):
                mark = i + chunksize
                offset = sum(v[i:mark])
                result[mark:] -= offset
        else:
            idxs = (
                range(0, -len_v, -1)
                if with_main_diagonal
                else range(-1, -len_v, -1)
            )
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, chunksize):
                mark = len_v - (i + chunksize)
                offset = sum(v[mark : (mark + chunksize)])
                result[:mark] -= offset
    else:
        if with_main_diagonal:
            result = v.clone()
        else:
            result = torch.zeros_like(v)
        #
        for i in range(len_v - 1):
            block_n, block_i = divmod(i + 1, chunksize)
            if block_i == 0:
                continue
            # get indices for result[out_beg:out_end] = v[beg:end]
            if lower:
                beg = block_n * chunksize
                end = beg + block_i
                out_end = min(beg + chunksize, len_v)
                out_beg = out_end - block_i
            else:
                end = len_v - (block_n * chunksize)
                beg = end - block_i
                out_beg = max(0, end - chunksize)
                out_end = out_beg + block_i
            # add to serrated pattern
            result[out_beg:out_end] += v[beg:end]
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
        use_fft=True,
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
        :param use_fft: Whether to use FFT when calling
        :fun:`serrated_hadamard_pattern`, used for the serrated measurements.
          FFT slightly decreases precision and needs a few buffer vectors of
          ``dims`` size, but greatly improves runtime for larger ``dims``.
        """
        h, w = lop.shape
        assert h == w, "Only square linear operators supported!"
        self.dims = h
        self.lop = lop
        self.n_serrat = num_serrated_measurements
        self.use_fft = use_fft
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
        self.stair_steps, self.stair_width = self._get_chunk_dims(
            self.dims, num_staircase_measurements
        )
        self.ssrft = SSRFT((self.n_serrat, self.dims), seed=seed)
        super().__init__(lop.shape)  # this sets self.shape also

    @staticmethod
    def _get_chunk_dims(dims, num_stair_meas):
        """ """
        stair_width = dims // (num_stair_meas + 1)
        stair_steps = torch.arange(0, dims, stair_width)
        return stair_steps, stair_width

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """

        serrated_hadamard_pattern(
            torch.ones(10),
            3,
            with_main_diagonal=True,
            lower=False,
            use_fft=False,
        )

        self.check_input(x, adjoint=False)
        #
        buff = torch.zeros_like(x)
        result = torch.zeros_like(x)
        # add step computations to result
        for beg, end in zip(self.stair_steps[:-1], self.stair_steps[1:]):
            if self.lower:
                buff[beg:end] = x[beg:end]
                result[end:] += (self.lop @ buff)[end:]
                buff[beg:end] = 0
            else:
                buff[end:] = x[end:]
                result[beg:end] += (self.lop @ buff)[beg:end]
                buff[end:] = 0
        # add serrated Hutchinson estimator to result
        for i in range(self.n_serrat):
            buff[:] = self.ssrft.get_row(i, x.dtype, x.device)
            result[:] += serrated_hadamard_pattern(
                buff, self.stair_width, lower=self.lower, use_fft=self.use_fft
            ) * (self.lop @ (buff * x))
        return result

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
