#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Scrambled Subsampled Randomized Fourier Transform (SSRFT).

PyTorch-compatible implementation of the SSRFT (as a matrix-free linear operator
and other utilities), from
`[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_.
"""


import torch
import torch_dct as dct

from .linops import BaseRandomLinOp
from .utils import (
    BadShapeError,
    NoFlatError,
    rademacher_flip,
    randperm,
    gaussian_noise,
    rademacher_noise,
)


# ##############################################################################
# #
# ##############################################################################
class NoiseLinOp(BaseRandomLinOp):
    """Base class for noisy linear operators.

    Consider a matrix of shape ``(h, w)`` composed of random-generated entries.
    For very large dimensions, the ``h * w`` memory requirement is intractable.
    Instead, this matrix-free operator generates each row (or column) one by
    one during matrix multiplication, while respecting two properties:

    * Both forward and adjoint operations are deterministic given a random seed
    * Both forward and adjoint operations are consistent with each other

    Users need to override :meth:`.sample` with their desired way of
    producing rows/columns (as specified by the ``partition`` given at
    initialization).
    """

    PARTITIONS = {"row", "column", "longer", "shorter"}

    def __init__(self, shape, seed=0b1110101001010101011, partition="longer"):
        """Instantiates a random linear operator.

        :param shape: ``(height, width)`` of linear operator.
        :param int seed: Seed for random behaviour.
        :param partition: Which kind of vectors will be produced by
          :meth:`.sample`. They can correspond to columns or rows of this
          linear operator. Longer means that the larger dimension is
          automatically used (e.g. columns in a thin linop, rows in a fat
          linop). Longer is generally recommended as it involves less
          iterations and can leverage more parallelization.
        """
        super().__init__(shape)
        self.seed = seed
        #
        self.partition = partition
        if partition not in self.PARTITIONS:
            raise ValueError(f"partition must be one of {self.PARTITIONS}!")

    def _get_partition(self):
        """Dispatch behaviour for :meth:`.sample`.

        :returns: A boolean depending on the chosen partitioning behaviour.
          True value corresponds to column, and false to row.
        """
        # if row or column is hardcoded, use that partition
        if self.partition in {"row", "column"}:
            by_column = self.partition == "column"
        #
        elif self.shape[0] >= self.shape[1]:  # if linop is tall...
            by_column = 1 if (self.partition == "longer") else 0
        else:  # if linop is fat...
            by_column = 0 if (self.partition == "longer") else 1
        #
        return by_column

    def matmul(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        h, w = self.shape
        result = torch.zeros(h, device=x.device, dtype=x.dtype)
        by_column = self._get_partition()
        if by_column:
            for idx in range(w):
                result += x[idx] * self.sample(h, idx, x.device)
        else:
            for idx in range(h):
                result[idx] += (x * self.sample(w, idx, x.device)).sum()
        #
        return result

    def rmatmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        h, w = self.shape
        result = torch.zeros(w, device=x.device, dtype=x.dtype)
        by_column = self._get_partition()
        if by_column:
            for idx in range(w):
                result[idx] = (x * self.sample(h, idx, x.device)).sum()
        else:
            for idx in range(h):
                result += x[idx] * self.sample(w, idx, x.device)
        #
        return result

    def sample(self, dims, idx, device):
        """Method used to sample random entries for this linear operator.

        Override this method with the desired behaviour. E.g. the following
        code results in a random matrix with i.i.d. Rademacher noise entries.
        Note that noise is generated on CPU to ensure reproducibility::

          r = rademacher_noise(dims, seed=idx + self.seed, device="cpu")
          return r.to(device)

        :param dims: Length of the produced random vector.
        :param idx: Index of the row/column to be sampled. Can be combined with
          ``self.seed`` to induce random behaviour.
        :param device: Device of the input vector that was used to call the
          matrix multiplication. The output of this method should match this
          device.
        """
        raise NotImplementedError


class GaussianIidLinOp(NoiseLinOp):
    """Random linear operator with standard i.i.d. Gaussian entries."""

    def sample(self, dims, idx, device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        result = gaussian_noise(dims, seed=idx + self.seed, device="cpu")
        return result.to(device)


class RademacherIidLinOp(NoiseLinOp):
    """Random linear operator with i.i.d. Rademacher entries."""

    def sample(self, dims, idx, device):
        """Samples a vector with standard Rademacher i.i.d. noise.

        See base class definition for details.
        """
        result = rademacher_noise(dims, seed=idx + self.seed, device="cpu")
        return result.to(device)
