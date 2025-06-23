#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for linear operators.

This PyTorch library is intended to work with (potentially matrix-free) linear
operators that support left- and right- matrix multiplication (via e.g.
``v @ lop`` and ``lop @ v``) as well as the ``.shape`` attribute.

This module implements basic support for said functionality, as well as some
default linear operators (composite, diagonal...).
"""


import torch

from .utils import BadShapeError


# ##############################################################################
# # CORE LINOP FUNCTIONALITY
# ##############################################################################
def linop_to_matrix(lop, dtype=torch.float32, device="cpu", adjoint=False):
    """Convert a linop to a matrix via one-hot matrix-vector products."""
    h, w = lop.shape
    result = torch.zeros(lop.shape, dtype=dtype, device=device)
    if adjoint:
        oh = torch.zeros(h, dtype=dtype, device=device)
        for i in range(h):
            oh *= 0
            oh[i] = 1
            result[i, :] = oh @ lop
    else:
        oh = torch.zeros(w, dtype=dtype, device=device)
        for i in range(w):
            oh *= 0
            oh[i] = 1
            result[:, i] = lop @ oh
    #
    return result


class BaseLinOp:
    """Base class for linear operators.

    Implements the ``.shape`` attribute and basic matmul functionality with
    vectors and matrices (also via the ``@`` operator). Intended to be
    extended with further functionality via :meth:`matmul` and :meth:`rmatmul`.

    :param shape: ``(height, width)`` of linear operator.
    """

    def __init__(self, shape):
        """Initializer. See class docstring."""
        if len(shape) != 2:
            raise BadShapeError("Shape must be a (height, width) pair!")
        self.shape = shape

    def __repr__(self):
        """Returns a string in the form <classname(shape)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})>"
        return s

    @staticmethod
    def check_input(x, lop_shape, adjoint):
        """Checking that input is a mat/vec of the right shape.

        :param x: The input to this linear operator.
        :param bool adjoint: If true, ``x @ self`` is assumed, otherwise
          ``self @ x``.
        """
        if not len(x.shape) in {1, 2}:
            raise BadShapeError("Only vector or matrix supported!")
        if adjoint:
            if x.shape[-1] != lop_shape[0]:
                raise BadShapeError(
                    f"Mismatching shapes! {x.shape} <--> {lop_shape}"
                )
        if not adjoint:
            if x.shape[0] != lop_shape[1]:
                raise BadShapeError(
                    f"Mismatching shapes! {lop_shape} <--> {x.shape}"
                )

    @classmethod
    def matmul_vectorizer(cls, x, lop_shape, vecmul_fn, rvecmul_fn, adjoint):
        """ """
        lop_h, lop_w = lop_shape
        cls.check_input(x, lop_shape, adjoint=adjoint)
        if len(x.shape) == 1:
            result = rvecmul_fn(x) if adjoint else vecmul_fn(x)
        else:
            x_h, x_w = x.shape
            num_vecs = x_h if adjoint else x_w
            result = torch.zeros(
                (num_vecs, lop_w) if adjoint else (lop_h, num_vecs),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(num_vecs):
                if adjoint:
                    result[i, :] = rvecmul_fn(x[i, :])
                else:
                    result[:, i] = vecmul_fn(x[:, i])
        return result

    def vecmul(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        :param x: Expected to be a vector of shape ``(w,)`` where this linop
          has shape ``(h, w)``. Note that inputs of shape ``(w, k)`` will be
          automatically passed to this method as ``k`` vectors of length ``w``.
        :returns: A vector of shape ``(h,)`` equaling ``self @ x``.
        """
        raise NotImplementedError

    def rvecmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        Like :meth:`matmul`, but with adjoint shapes.
        """
        raise NotImplementedError

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to :meth:`.matmul` for vectors or matrices."""
        result = self.matmul_vectorizer(
            x, self.shape, self.vecmul, self.rvecmul, adjoint=False
        )
        return result

    def __rmatmul__(self, x):
        """Convenience wrapper to :meth:`.rmatmul` for vectors or matrices."""
        result = self.matmul_vectorizer(
            x, self.shape, self.vecmul, self.rvecmul, adjoint=True
        )
        return result

    def __imatmul__(self, x):
        """Assignment matmul operator ``@=``.

        .. note::
          This method is deactivated by default since linear operators may be
          matrix-free.
        """
        raise NotImplementedError(
            "Unsupported matmul assignment: not matrix-free compatible!"
        )

    def t(self):
        """(Hermitian) transposition."""
        return TransposedLinOp(self)


class ByVectorLinOp(BaseLinOp):
    """Matrix-free linop computed vector by vector.

    This matrix-free operator generates each row (or column) one by
    one during matrix multiplication. Users can decide which modality to use
    at instantiation via the ``by_row`` boolean flag.

    Extensions of this class should implement the :meth:`sample` method
    accordingly to return vectors of the right shape.

    .. note::
      The ``by_row`` flag has implications in terms of memory and runtime.
      If true, for a ``lop`` of shape ``(h, w)``, the ``lop @ x`` matrix-vector
      multiplication will call :meth:`get_vector`` ``h`` times, and perform
      ``h`` dot products of dimension ``w``. If false, it will perform ``w``
      scalar products of dimension ``h``. In the case of ``x @ lop``,
      the scalar and dot products are swapped.

      Therefore, developers need to override :meth:`get_vector` taking this
      flag into account, and users should set it to the scenario that is most
      efficient (e.g. by-column is generally more efficient when ``h > w``).
    """

    def __init__(self, shape, by_row=False):
        """Initializer. See class docstring."""
        super().__init__(shape)
        self.by_row = by_row

    def get_vector(self, idx, device):
        """Method to gather vector entries for this linear operator.

        Override this method with the desired behaviour. For a shape of
        ``(h, w)``, it should return vectors of length ``w`` if
        ``self.by_row`` is true, and lenght ``h`` otherwise.

        :param idx: Index of the row/column to be sampled. It will go from 0
          to ``dims - 1``, both included, where ``dims`` is ``h`` if
          ``self.by_row`` is true, and ``w`` otherwise.
        :param device: The returned vector is expected to be on this device.
        """
        raise NotImplementedError

    def vecmul(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See class docstring and parent class for more details.
        """
        h, w = self.shape
        result = torch.zeros(h, device=x.device, dtype=x.dtype)
        if self.by_row:
            for idx in range(h):
                result[idx] = (x * self.get_vector(idx, x.device)).sum()
        else:
            for idx in range(w):
                result += x[idx] * self.get_vector(idx, x.device)
        return result

    def rvecmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See class dostring and parent class for more details.
        """
        h, w = self.shape
        result = torch.zeros(w, device=x.device, dtype=x.dtype)
        if self.by_row:
            for idx in range(h):
                result += x[idx] * self.get_vector(idx, x.device)
        else:
            for idx in range(w):
                result[idx] = (x * self.get_vector(idx, x.device)).sum()
        #
        return result


class TransposedLinOp:
    """ """

    def __init__(self, lop):
        """Initializer. See class docstring."""
        if isinstance(lop, TransposedLinOp):
            raise ValueError("LinOp is already transposed! use x.lop")
        self.lop = lop
        self.shape = self.lop.shape[::-1]

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to :meth:`.matmul` for vectors or matrices."""
        x_vec = len(x.shape) == 1
        result = BaseLinOp.matmul_vectorizer(
            x.conj() if x_vec else x.H,
            self.lop.shape,
            self.lop.vecmul,
            self.lop.rvecmul,
            adjoint=True,
        )
        result = result.conj() if x_vec else result.H
        return result

    def __rmatmul__(self, x):
        """Convenience wrapper to :meth:`.rmatmul` for vectors or matrices."""
        x_vec = len(x.shape) == 1
        result = BaseLinOp.matmul_vectorizer(
            x.conj() if x_vec else x.H,
            self.lop.shape,
            self.lop.vecmul,
            self.lop.rvecmul,
            adjoint=False,
        )
        result = result.conj() if x_vec else result.H
        return result

    def t(self):
        """Undo transposition"""
        return self.lop


# ##############################################################################
# # NOISY LINOP FUNCTIONALITY
# ##############################################################################
class BaseRandomLinOp(BaseLinOp):
    """Linear operators with pseudo-random behaviour.

    Like the base LinOp, but with a ``seed`` attribute that can be used to
    control random behaviour.

    :param shape: ``(height, width)`` of linear operator.
    :param int seed: Seed for random behaviour.
    """

    def __init__(self, shape, seed=0b1110101001010101011):
        """Initializer. See class docstring."""
        super().__init__(shape)
        self.seed = seed

    def __repr__(self):
        """Returns a string in the form <classname(shape), seed>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}), seed={self.seed}>"
        return s


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
