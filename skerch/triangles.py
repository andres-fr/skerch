#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to support sketched estimation of upper/lower triangulars.

Given an arbitrary square ``linop``, this module implements matrix-free
functionality to compute left- and right matrix-multiplications of both
``lower_triang(linop)`` and ``upper_triang(linop)``.

This is done by first exactly computing a limited number of "main rectangles",
a.k.a. "steps", that fully belong to the triangular submatrix, and then
estimating the "serrated pattern" that they leave as residual via modified
Hutchinson. The addition of the steps plus the serrated pattern forms the full
triangle. See :class:`TriangularLinOp` and the
:ref:`Examples section<Examples>` for more details.

Contents:

* Hadamard pattern to estimate block-triangular matrix-vector products using
  Hutchinson's method (used by :class:`TriangularLinOp`).
* The :class:`TriangularLinOp` linear operator implementing triangular
  matrix-vector products for arbitrary square linops.
"""


import torch

from .linops import BaseLinOp
from .ssrft import SSRFT
from .subdiagonals import subdiag_hadamard_pattern
from .utils import BadShapeError


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
    :param use_fft: See :func:`subdiag_hadamard_pattern`.
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
    r"""Given a square linop, compute products with one of its triangles.

    The triangle of a linear operator can be approximated from the full operator
    via a "staircase pattern" of exact measurements, whose computation is exact
    and fast. For example, given an operator of shape ``(1000, 1000)``, and
    stairs of size 100, yields 9 exact measurements strictly under the diagonal,
    the first one covering ``lop[100:, :100]``, the next one
    ``lop[200, 100:200]``, and so on. The more measurements, the more closely
    the full triangle is approximated.

    Note that this staircase pattern leaves a block-triangular section of the
    linop untouched (near the main diagonal). This part can be then estimated
    with the help of  :func:`serrated_hadamard_pattern`, completing the
    triangular approximation, as follows:


    Given a square linear operator :math:`A`, and random vectors
    :math:`v \sim \mathcal{R}` with :math:`\mathbb{E}[v v^T] = I`, consider
    the generalized Hutchinson diagonal estimator:

    .. math::

      f(A) =
      \mathbb{E}_{v \sim \mathcal{R}} \big[ \varphi(v) \odot Av \big]

    In this case, if the :math:`\varphi` function follows a "serrated
    Hadamard pattern", :math:`f(A)` will equal a block-triangular subset of
    :math:`A`.
    """

    def __init__(
        self,
        lop,
        stair_width=None,
        num_hutch_measurements=0,
        lower=True,
        with_main_diagonal=True,
        seed=0b1110101001010101011,
        use_fft=True,
    ):
        """Init method.

        :param lop: A square linear operator of order ``dims``, such that
          ``self @ v`` will equal ``triangle(lop) @ v``. It must implement a
          ``lop.shape = (dims, dims)`` attribute as well as the left- and right-
          matmul operator ``@``, interacting with torch tensors.
        :param stair_width: Width of each step in the staircase pattern. If
          it is 1, a total of ``dims`` exact measurements will be performed.
          If it equals ``dims``, no exact measurements will be performed (since
          the staircase pattern would cover the full triangle). The step size
          regulates this trade-off: Ideally, we want as many exact measurements
          as possible, but not too many. If no value is provided, ``dims // 2``
          is chosen by default, such that only 1 exact measurement is performed.
        :param num_hutch_measurements: The leftover entries from the
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
          Hutchinson estimator.
        :param use_fft: Whether to use FFT for the Hutchinson estimation. See
          :func:`subdiag_hadamard_pattern` for more details.
        """
        h, w = lop.shape
        if h != w:
            raise BadShapeError("Only square linear operators supported!")
        self.dims = h
        if self.dims < 1:
            raise BadShapeError("Empty linear operators not supported!")
        if stair_width is None:
            stair_width = max(1, self.dims // 2)
        assert stair_width > 0, "Stair width must be a positive int!"
        self.lop = lop
        self.use_fft = use_fft
        self.lower = lower
        self.with_main = with_main_diagonal
        self.n_hutch = num_hutch_measurements
        #
        self.stair_width = stair_width
        num_stair_meas = sum(
            1 for _ in self._iter_stairs(self.dims, stair_width)
        )
        self.quack = num_stair_meas
        assert (
            num_stair_meas <= self.dims
        ), "More staircase measurements than dimensions??"
        assert (
            self.n_hutch <= self.dims
        ), "More Hutchinson measurements than dimensions??"
        assert (
            num_stair_meas + self.n_hutch
        ) <= self.dims, "More total measurements than dimensions??"
        #
        self.ssrft = SSRFT((self.n_hutch, self.dims), seed=seed)
        super().__init__(lop.shape)  # this sets self.shape also

    @staticmethod
    def _iter_stairs(dims, stair_width):
        """Helper method to iterate over staircase indices.

        This method implements an iterator that yields ``(begin, end)`` index
        pairs for each staircase-pattern step. It terminates before ``end``
        is equal or greater than ``self.dims``, since only full steps are
        considered.
        """
        beg, end = 0, stair_width
        while end < dims:
            yield (beg, end)
            beg = end
            end = beg + stair_width

    def _matmul_helper(self, x, adjoint=False):
        """Forward and adjoint triangular matrix multiplications.

        Since forward and adjoint matmul share many common computations, this
        method implements both at the same time. The specific mode can be
        dispatched using the ``adjoint`` parameter.
        """
        self.check_input(x, adjoint=adjoint)
        # we don't factorize this method because we want to share buff
        # across both loops to hopefully save memory
        buff = torch.zeros_like(x)
        result = torch.zeros_like(x)
        # add step computations to result
        for beg, end in self._iter_stairs(self.dims, self.stair_width):
            if (not adjoint) and self.lower:
                buff[beg:end] = x[beg:end]
                result[end:] += (self.lop @ buff)[end:]
                buff[beg:end] = 0
            elif (not adjoint) and (not self.lower):
                buff[end:] = x[end:]
                result[beg:end] += (self.lop @ buff)[beg:end]
                buff[end:] = 0
            elif adjoint and self.lower:
                buff[end:] = x[end:]
                result[beg:end] += (buff @ self.lop)[beg:end]
                buff[end:] = 0
            elif adjoint and (not self.lower):
                buff[beg:end] = x[beg:end]
                result[end:] += (buff @ self.lop)[end:]
                buff[beg:end] = 0
            else:
                raise RuntimeError("This should never happen")
        # add Hutchinson estimator to result
        for i in range(self.n_hutch):
            buff[:] = self.ssrft.get_row(i, x.dtype, x.device)
            result[:] += serrated_hadamard_pattern(
                buff,
                self.stair_width,
                self.with_main,
                lower=(self.lower ^ adjoint),  # pattern also depends on adj
                use_fft=self.use_fft,
            ) * (
                ((buff * x) @ self.lop) if adjoint else (self.lop @ (buff * x))
            )
        return result

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        return self._matmul_helper(x, adjoint=False)

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        return self._matmul_helper(x, adjoint=True)

    def __repr__(self):
        """Readable string version of this object.

        Returns a string in the form
        <TriangularlLinOp[lop](lower/upper)>.
        """
        clsname = self.__class__.__name__
        lopstr = self.lop.__repr__()
        lower_str = "lower" if self.lower else "upper"
        diag_str = "with" if self.with_main else "no"
        result = f"<{clsname}[{lopstr}]({lower_str}, {diag_str} main diag)>"
        return result
