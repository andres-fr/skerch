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

from .utils import BadShapeError, NoFlatError, gaussian_noise


# ##############################################################################
# # BASE LINEAR OPERATOR
# ##############################################################################
class BaseLinOp:
    """Base class for linear operators.

    Provides the ``.shape`` attribute and basic matmul functionality with
    vectors and matrices (also via the ``@`` operator). Intended to be
    extended with further functionality.
    """

    def __init__(self, shape):
        """:param shape: ``(height, width)`` of linear operator."""
        if len(shape) != 2:
            raise BadShapeError("Shape must be a (height, width) pair!")
        self.shape = shape

    def __repr__(self):
        """Returns a string in the form <classname(shape)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})>"
        return s

    def check_input(self, x, adjoint):
        """Checking that input has compatible shape.

        :param x: The input to this linear operator.
        :param bool adjoint: If true, ``x @ self`` is assumed, otherwise
          ``self @ x``.
        """
        try:
            assert len(x.shape) in {
                1,
                2,
            }, "Only vector or matrix input supported"
            #
            if adjoint:
                assert (
                    x.shape[-1] == self.shape[0]
                ), f"Mismatching shapes! {x.shape} <--> {self.shape}"
            else:
                assert (
                    x.shape[0] == self.shape[1]
                ), f"Mismatching shapes! {self.shape} <--> {x.shape}"
        except AssertionError as ae:
            raise BadShapeError from ae

    def matmul(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        :param x: Expected a tensor of shape ``(w,)``.
          Note that shapes ``(w, k)`` will be automatically passed as ``k``
          vectors of length ``w``.
        """
        raise NotImplementedError

    def rmatmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        :param x: Expected a tensor of shape ``(h,)``.
          Note that shapes ``(h, k)`` will be automatically passed as ``k``
          vectors of length ``h``.
        """
        raise NotImplementedError

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to :meth:`.matmul`."""
        self.check_input(x, adjoint=False)
        try:
            return self.matmul(x)
        except NoFlatError:
            result = torch.zeros((self.shape[0], x.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[1]):
                result[:, i] = self.matmul(x[:, i])
            return result

    def __rmatmul__(self, x):
        """Convenience wrapper to :meth:`.rmatmul`."""
        self.check_input(x, adjoint=True)
        try:
            return self.rmatmul(x)
        except NoFlatError:
            result = torch.zeros((x.shape[0], self.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[0]):
                result[i, :] = self.rmatmul(x[i, :])
            return result

    def __imatmul__(self, x):
        """Assignment matmul operator ``@=``.

        .. note::
          This method is deactivated by default since linear operators may be
          matrix-free.
        """
        raise NotImplementedError("Matmul assignment not supported!")


# ##############################################################################
# # BASE LINEAR OPERATORS INVOLVING RANDOMNESS
# ##############################################################################
class BaseRandomLinOp(BaseLinOp):
    """Linear operators with pseudo-random behaviour.

    Like the base LinOp, but with a ``seed`` attribute that can be used to
    control random behaviour.
    """

    def __init__(self, shape, seed=0b1110101001010101011):
        """Instantiates a random linear operator.

        :param shape: ``(height, width)`` of linear operator.
        :param int seed: Seed for random behaviour.
        """
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


class GaussianIidLinOp(NoiseLinOp):
    """Random linear operator with i.i.d. Gaussian entries."""

    def sample(self, dims, idx, device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        result = gaussian_noise(dims, seed=idx + self.seed, device="cpu")
        return result.to(device)


# ##############################################################################
# # COMPOSITE LINEAR OPERATORS
# ##############################################################################
class CompositeLinOp:
    """Composite linear operator.

    This class composes an ordered collection of operators ``[A, B, C, ...]``
    into ``A @ B @ C ...``.

    .. note::

      Using this class could be more inefficient than directly computing the
      composed operator, e.g. if ``A.shape = (1, 1000)`` and
      ``B.shape = (1000, 1)``, then computing the scalar ``C = A @ B`` and then
      applying it is more efficient than keeping a ``CompositeLinearoperator``
      with ``A, B`` (in terms of both memory and computation). This class does
      not check for such cases, users are encouraged to take this into account.
    """

    def __init__(self, named_operators):
        """Instantiates a composite linear operator.

        :param named_operators: Ordered collection in the form
          ``[(n_1, o_1), ...]`` where each ``n_i`` is a string with the name of
          operator ``o_i``. Each ``o_i`` operator must implement ``__matmul__``
          and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
          object will then become the composite operator ``o_1 @ o_2 @ ...``
        """
        self.names, self.operators = zip(*named_operators)
        shapes = [o.shape for o in self.operators]
        for (_h1, w1), (h2, _w2) in zip(shapes[:-1], shapes[1:]):
            if w1 != h2:
                raise BadShapeError(f"Mismatching shapes in {shapes}!")
        self.shape = (shapes[0][0], shapes[-1][-1])

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        result = self.operators[-1] @ x
        for o in reversed(self.operators[:-1]):
            result = o @ result
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        result = x @ self.operators[0]
        for o in self.operators[1:]:
            result = result @ o
        return result

    def __repr__(self):
        """Returns a string in the form op1 @ op2 @ op3 ..."""
        result = " @ ".join(self.names)
        return result


class SumLinOp(BaseLinOp):
    """Sum of linear operators.

    Given a collection of same-shape linear operators ``A, B, C, ...``, this
    clsas behaves like the sum ``A + B + C ...``.
    """

    def __init__(self, named_operators):
        """Instantiates a summation linear operator.

        :param named_operators: Collection in the form ``{(n_1, o_1), ...}``
          where each ``n_i`` is a string with the name of operator ``o_i``.
          Each ``o_i`` operator must implement ``__matmul__``
          and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
          object will then become the operator ``o_1 + o_2 + ...``
        """
        self.names, self.operators = zip(*named_operators)
        shapes = [o.shape for o in self.operators]
        if not shapes:
            raise ValueError(f"Empty linop collection? {named_operators}")
        for shape in shapes:
            if shape != shapes[0]:
                raise BadShapeError(f"All shapes must be equal! {shapes}")
        super().__init__(shapes[0])  # this sets self.shape also

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=False)
        result = self.operators[0] @ x
        for o in self.operators[1:]:
            result += o @ x
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=True)
        result = x @ self.operators[0]
        for o in self.operators[1:]:
            result += x @ o
        return result

    def __repr__(self):
        """Returns a string in the form op1 + op2 + op3 ..."""
        result = " + ".join(self.names)
        return result


# ##############################################################################
# # DIAGONAL LINEAR OPERATORS
# ##############################################################################
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


class BandedLinOp(BaseLinOp):
    r"""Banded linear operator (composed of diagonals).

    Given a collection of :math:`k` vectors, this class implements a banded
    linear operator, where each given vector is a (sub)-diagonal. It is
    composed by :math:`k` :class:`DiagonalLinOp` operators, thus its memory
    and runtime is equivalent to storing and running the individual diagonals.

    .. note::
      All given vectors must be of appropriate length with respect to their
      position. For example, a square tridiagonal matrix of shape ``(n, n)``
      has a main diagonal at index 0 with length ``n``, and two subdiagonals at
      indices 1, -1 with length ``n - 1``. Still, this linop can also be
      non-square (unless it is symmetric), as long as all given diagonals fit
      in the implicit shape.

    Usage example::

      diags = {0: some_diag, 1: some_superdiag, -1, some_subdiag}
      B = BandedLinOp(diags, symmetric=False)
      w = B @ v
    """

    MAX_PRINT_ENTRIES = 20

    @staticmethod
    def __initial_checks(indexed_diags, symmetric):
        """Performs input checks right at initialization."""
        # extract diagonal lengths and check they are vectors
        diag_lengths = {}
        for idx in sorted(indexed_diags):
            diag = indexed_diags[idx]
            if len(diag.shape) == 1:
                diag_lengths[idx] = len(diag)
            else:
                raise BadShapeError("All diagonals must be vectors!")
        if not diag_lengths:
            raise ValueError(f"Empty linop dict? {indexed_diags}")
        # symmetric mode does not accept negative indices
        if symmetric:
            if any(idx < 0 for idx in indexed_diags):
                raise BadShapeError(
                    "Symmetric banded linop only admits nonnegative indices!"
                    + f" {diag_lengths}"
                )
        #
        return diag_lengths

    def __init__(self, indexed_diags, symmetric=False):
        """Instantiates a banded linear operator.

        :param indexed_diags: Dictionary in the form ``{idx: diag, ...}`` where
          ``diag`` is a torch vector containing a diagonal, and ``idx``
          indicates the location of the diagonal: 0 is the main diagonal, 1 the
          superdiagonal (``lop[i, i+1]``), -1 the subdiagonal, and so on.
        :param symmetric: If true, only diagonals with nonnegative indices are
          admitted. Each positive index will be replicated as a negative one.

        .. note::
          The shape of this linear operator is implicitly given by the
          diagonal lengths and indices. Any inconsistent input will result in
          a ``BadShapeError``. In particular, symmetric banded matrices must
          also be square.
        """
        # extract diagonal lengths and check they are vectors
        # also check that symmetric mode does not accept negative indices
        diag_lengths = self.__initial_checks(indexed_diags, symmetric)
        # figure out the smallest matrix that fits all diagonals
        # note that in symmetric mode we need to add the negative indices
        end_coords = {}
        height, width = 0, 0
        for idx, length in diag_lengths.items():
            i0, j0 = (0, idx) if idx >= 0 else (abs(idx), 0)
            i1, j1 = i0 + length, j0 + length
            height, width = max(height, i1), max(width, j1)
            end_coords[idx] = (i1, j1)
        if symmetric:
            for idx, length in diag_lengths.items():
                if idx > 0:
                    i0, j0, i1, j1 = idx, 0, idx + length, length
                    height, width = max(height, i1), max(width, j1)
                    end_coords[-idx] = (i1, j1)
        # check that all given diagonals fit the linop shape tightly
        inconsistent_idxs = set()
        for idx, (i1, j1) in end_coords.items():
            if (i1 != height) and (j1 != width):
                inconsistent_idxs.add(idx)
        #
        if inconsistent_idxs:
            raise BadShapeError(
                f"Inconsistent diagonal indices/lengths! {diag_lengths}, "
                + f"triggered by indices {inconsistent_idxs} "
                + f" for shape {(height, width)}"
            )
        # if symmetric, linop must be square
        if symmetric:
            if height != width:
                raise BadShapeError(
                    f"Symmetric banded linop must be square! {diag_lengths}"
                )
        # done checking, initialize object
        self.diags = {i: DiagonalLinOp(d) for i, d in indexed_diags.items()}
        self.symmetric = symmetric
        super().__init__((height, width))

    def __matmul_helper(self, x, adjoint=False):
        """Helper method to perform multiple diagonal matmuls."""
        self.check_input(x, adjoint=adjoint)
        #
        diags = {}
        for idx, diag in self.diags.items():
            diags[idx] = diag
            if self.symmetric and idx > 0:
                diags[-idx] = diag
        #
        outdim = self.shape[1] if adjoint else self.shape[0]
        result = torch.zeros(outdim, dtype=x.dtype, device=x.device)
        #
        for idx, d in diags.items():
            if adjoint:
                in_beg = abs(min(idx, 0))
                out_beg = max(idx, 0)
            else:
                in_beg = max(idx, 0)
                out_beg = abs(min(idx, 0))
            dlen = d.shape[0]
            result[out_beg : out_beg + dlen] += d @ x[in_beg : in_beg + dlen]
        #
        return result

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        return self.__matmul_helper(x, adjoint=False)

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        return self.__matmul_helper(x, adjoint=True)

    def __repr__(self):
        """Returns a string in the form <BandedLinOp(shape)[idx1,..., sym]>."""
        clsname = self.__class__.__name__
        idxs = ", ".join(str(idx) for idx in sorted(self.diags))
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]})[{idxs}, "
            + f"sym={self.symmetric}]>"
        )
        return s

    def to_matrix(self):
        """Convert this linear operator into a matrix."""
        # check that all diagonals are of same dtype and device
        dtypes, devices = zip(
            *((d.diag.dtype, d.diag.device) for d in self.diags.values())
        )
        if len(set(dtypes)) > 1:
            raise RuntimeError(f"Inconsistent diagonal dtypes! {dtypes}")
        if len(set(devices)) > 1:
            raise RuntimeError(f"Inconsistent diagonal devices! {devices}")
        # create and populate resulting matrix
        result = torch.zeros(self.shape, dtype=dtypes[0], device=devices[0])
        for idx, diag in self.diags.items():
            dlen = len(diag.diag)
            if idx >= 0:
                result[range(0, dlen), range(idx, dlen + idx)] = diag.diag
            else:
                idx = abs(idx)
                result[range(idx, dlen + idx), range(0, dlen)] = diag.diag
        #
        if self.symmetric:
            result = result + result.T
            result[range(len(result)), range(len(result))] /= 2
        #
        return result


# ##############################################################################
# # PROJECTION LINEAR OPERATORS
# ##############################################################################
class OrthProjLinOp(BaseLinOp):
    r"""Linear operator for an orthogonal projector.

    Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
    columns, this class implements the orthogonal projector onto its span,
    i.e. :math:`Q Q^T`.
    """

    def __init__(self, Q):
        """Object initialization.

        :param Q: Linear operator of shape ``(h, w)`` with ``h >= w``, expected
          to be orthonormal (i.e. columns with unit Euclidean norm and all
          orthogonal to each other). It must implement the left and right
          matmul operations via the ``@`` operator.

        .. warning::

          This class assumes ``Q`` is orthonormal, but this is not checked.
        """
        self.Q = Q
        #
        h, w = Q.shape
        if w > h:
            raise BadShapeError("Projector matrix must be thin!")
        super().__init__((h, h))  # this sets self.shape also

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=False)
        if len(x.shape) == 1:
            result = self.Q @ (x @ self.Q)
        elif len(x.shape) == 2:
            result = self.Q @ (x.T @ self.Q).T
        else:
            raise RuntimeError(f"Unsupported input shape: {x.shape}")
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=True)
        if len(x.shape) == 1:
            result = self.Q @ (x @ self.Q)
            return result
        elif len(x.shape) == 2:
            result = self.Q @ (x @ self.Q).T
            return result.T
        else:
            raise RuntimeError(f"Unsupported input shape: {x.shape}")

    def __repr__(self):
        """Returns a string in the form <OrthProjLinOp(Q_shape)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.Q.shape})>"
        return s


class NegOrthProjLinOp(OrthProjLinOp):
    """Linear operator for a negative orthogonal projector.

    Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
    columns, this class implements the orthogonal projector :math:`Q Q^T`.
    Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
    columns, this class implements the orthogonal projector onto the space
    orthogonal to its span, i.e. :math:`(I - Q Q^T)`.
    """

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        return x - super().__matmul__(x)

    def __rmatmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        return x - super().__rmatmul__(x)


# ##############################################################################
# # TORCH INTEROPERABILITY
# ##############################################################################
class TorchLinOpWrapper:
    """Linear operator that always accepts and produces PyTorch tensors.

    Since this library operates mainly with PyTorch tensors, but some useful
    LinOps interface with e.g. NumPy arrays instead, this mixin class acts as a
    wraper on the ``__matmul__`` and ``__rmatmul__`` operators,
    so that the operator expects and returns torch tensors, even when the
    wrapped operator interfaces with NumPy/HDF5. Usage example::

      # extend NumPy linear operator via multiple inheritance
      class TorchWrappedLinOp(TorchLinOpWrapper, LinOp):
          pass
      lop = TorchWrappedLinOp(...)  # instantiate normally
      w = lop @ v  # now v can be a PyTorch tensor
    """

    @staticmethod
    def _input_wrapper(x):
        """Helper method to admit PyTorch tensors."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy(), x.device
        else:
            return x, None

    @staticmethod
    def _output_wrapper(x, torch_device=None):
        """Helper method to produce PyTorch tensors."""
        if torch_device is not None:
            return torch.from_numpy(x).to(torch_device)
        else:
            return x

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        Casts given PyTorch tensor into NumPy before applying matmul.
        Then casts produced result back into same PyTorch datatype and device.
        """
        x, device = self._input_wrapper(x)
        result = self._output_wrapper(super().__matmul__(x), device)
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        Casts given PyTorch tensor into NumPy before applying matmul.
        Then casts produced result back into same PyTorch datatype and device.
        """
        x, device = self._input_wrapper(x)
        result = self._output_wrapper(super().__rmatmul__(x), device)
        return result

    def __repr__(self):
        """Returns a string in the form TorchLinOpWrapper<LinOp ...>."""
        wrapper = self.__class__.__name__
        result = f"{wrapper}<{super().__repr__()}>"
        return result
