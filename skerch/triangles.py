#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to support sketched estimation of upper/lower triangulars.

Given an arbitrary square ``linop``, this module implements matrix-free
functionality to compute left- and right matrix-multiplications of both
``lower_triang(linop)`` and ``upper_triang(linop)``.

This is done by first exactly computing a limited number of "main rectangles",
a.k.a. "steps", that fully belong to the triangular submatrix, and then
estimating the "serrated pattern" that they leave as residual.
The addition of the steps plus the serrated pattern forms the full triangle.
See :class:`TriangularLinOp` for more details.

Contents:

* Hadamard pattern to estimate block-triangular matrix-vector products using
  Hutchinson's method (used by :class:`TriangularLinOp`).
* The :class:`TriangularLinOp` linear operator implementing triangular
  matrix-vector products for arbitrary square linops.
"""


import torch

from .distributed_decompositions import orthogonalize
from .linops import CompositeLinOp, NegOrthProjLinOp
from .ssrft import SSRFT

from .utils import rademacher_noise
from .linops import BaseLinOp

from .subdiagonals import subdiag_hadamard_pattern


# ##############################################################################
# # HADAMARD
# ##############################################################################
def serrated_hadamard_pattern(
    v, blocksize, with_main_diagonal=True, lower=True, use_fft=False
):
    """Shifted copies of vectors for block-triangular Hutchinson estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param with_main_diagonal: If true, the main diagonal will be included
          in the patterns, otherwise excluded.
    :param lower: If true, the block-triangles will be below the diagonal,
      otherwise above.
    :param use_fft: See :fun:`subdiag_hadamard_pattern`.
    :returns: A vector of same shape, type and device as ``v``, composed of
      shifted copies of ``v`` following a block-triangular pattern.

    For example, given a 10-dimensional vector, the corresponding serrated
    pattern with ``blocksize=3, with_main_diagonal=True, lower=True`` yields
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
    if blocksize < 1:
        raise ValueError("Block size must be a positive scalar!")
    #
    len_v = len(v)
    if use_fft:
        if lower:
            idxs = range(len_v) if with_main_diagonal else range(1, len_v)
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, blocksize):
                mark = i + blocksize
                offset = sum(v[i:mark])
                result[mark:] -= offset
        else:
            idxs = (
                range(0, -len_v, -1)
                if with_main_diagonal
                else range(-1, -len_v, -1)
            )
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, blocksize):
                mark = len_v - (i + blocksize)
                offset = sum(v[mark : (mark + blocksize)])
                result[:mark] -= offset
    else:
        if with_main_diagonal:
            result = v.clone()
        else:
            result = torch.zeros_like(v)
        #
        for i in range(len_v - 1):
            block_n, block_i = divmod(i + 1, blocksize)
            if block_i == 0:
                continue
            # get indices for result[out_beg:out_end] = v[beg:end]
            if lower:
                beg = block_n * blocksize
                end = beg + block_i
                out_end = min(beg + blocksize, len_v)
                out_beg = out_end - block_i
            else:
                end = len_v - (block_n * blocksize)
                beg = end - block_i
                out_beg = max(0, end - blocksize)
                out_end = out_beg + block_i
            # add to serrated pattern
            result[out_beg:out_end] += v[beg:end]
    #
    return result


# ##############################################################################
# # TRIANGULAR LINOP
# ##############################################################################
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
          ``self @ v`` will equal ``triangle(lop) @ v``. It must implement a
          ``lop.shape = (h, w)`` attribute as well as the left- and right-
          matmul operator ``@``, interacting with torch tensors.
        :param num_staircase_measurements: The exact part of the triangular
          approximation, comprising measurements following a "staircase"
          pattern. Runtime grows linearly with this parameter. Memory is
          unaffected. Approximation error shrinks with ``1 / measurements``.
          It can be 0, but since this part is exact and efficient, it probably
          should be a substantial part of the total measurements.
        :param num_serrated_measurements: The leftover entries from the
          staircase measurements are approximated here using an extension of
          the Hutchinson diagonal estimator. This estimator generally requires
          many measurements to be informative, and it can even be counter-
          productive if not enough measurements are given. If ``lop``is not
          diagonally dominant, consider setting this to 0 for a sufficiently
          good approximation via ``staircase_measurements``. Otherwise,
          make sure to provide a sufficiently high number of measurements.
        :param lower: If true, lower triangular matmuls will be computed.
          Otherwise, upper triangular.
        :param with_main_diagonal: If true, the main diagonal will be included
          in the triangle, otherwise excluded. If you already have precomuted
          the diagonal elsewhere, consider excluding it from this approximation,
          and adding it separately.
        :param seed: Seed for the random SSRFT measurements used in the
          serrated estimator.
        :param use_fft: Whether to use FFT for the serrated estimation. See
          :fun:`subdiag_hadamard_pattern` for more details.
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
