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
# # AGGREGATE LINEAR OPERATORS
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


    :param named_operators: Ordered collection in the form
      ``[(n_1, o_1), ...]`` where each ``n_i`` is a string with the name of
      operator ``o_i``. Each ``o_i`` operator must implement ``__matmul__``
      and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
      object will then become the composite operator ``o_1 @ o_2 @ ...``
    """

    def __init__(self, named_operators):
        """Initializer. See class docstring."""
        self.names, self.operators = zip(*named_operators)
        shapes = [o.shape for o in self.operators]
        if not shapes:
            raise ValueError(f"Empty linop collection? {named_operators}")
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

    Given a collection of same-shape linear operators ``A, B, C, D ...``, this
    class behaves like the sum ``A + B + C - D ...``.

    :param named_signed_operators: Collection in the form
      ``{(n_1, s_i, o_1), ...}`` where each ``n_i`` is a string with the name
      of operator ``o_i``, and ``s_i`` is a boolean: if true, this operator
      is to be added, otherwise subtracted.
      Each ``o_i`` operator must implement ``__matmul__``
      and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
      object will then become the operator ``o_1 + o_2 - ...``
    """

    def __init__(self, named_signed_operators):
        """Instantiates a summation linear operator."""
        self.names, self.signs, self.operators = zip(*named_signed_operators)
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
        self.check_input(x, self.shape, adjoint=False)
        result = self.operators[0] @ (x if self.signs[0] else -x)
        for o, s in zip(self.operators[1:], self.signs[1:]):
            if s:
                result += o @ x
            else:
                result -= o @ x
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, self.shape, adjoint=True)
        result = x @ self.operators[0]
        for o in self.operators[1:]:
            result += x @ o
        return result

    def __repr__(self):
        """Returns a string in the form op1 + op2 + op3 ..."""
        signs_str = ["+ " if s else "- " for s in self.signs]
        result = ("-" if not self.signs[0] else "") + self.names[0]
        for s, n in zip(self.signs[1:], self.names[1:]):
            result += (" + " if s else " - ") + n
        return result


# # ##############################################################################
# # # DIAGONAL LINEAR OPERATORS
# # ##############################################################################
# class DiagonalLinOp(BaseLinOp):
#     r"""Diagonal linear operator.

#     Given a vector ``v`` of ``d`` dimensions, this class implements a diagonal
#     linear operator of shape ``(d, d)`` via left- and right-matrix
#     multiplication, as well as the ``shape`` attribute, only requiring linear
#     (:math:`\mathcal{O}(d)`) memory and runtime.
#     """

#     MAX_PRINT_ENTRIES = 20

#     def __init__(self, diag):
#         """:param diag: Vector to be casted as diagonal linop."""
#         if len(diag.shape) != 1:
#             raise BadShapeError("Diag must be a vector!")
#         self.diag = diag
#         super().__init__((len(diag),) * 2)  # this sets self.shape also

#     def __matmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         See parent class for more details.
#         """
#         self.check_input(x, self.shape, adjoint=False)
#         if len(x.shape) == 2:
#             result = (x.T * self.diag).T
#         else:
#             # due to torch warning, can't transpose shapes other than 2
#             result = x * self.diag
#         return result

#     def __rmatmul__(self, x):
#         """Adjoint (left) matrix-vector multiplication ``x @ self``.

#         See parent class for more details.
#         """
#         self.check_input(x, self.shape, adjoint=True)
#         result = x * self.diag
#         return result

#     def __repr__(self):
#         """Returns a string in the form <DiagonalLinOp(shape)[v1, v2, ...]>."""
#         clsname = self.__class__.__name__
#         diagstr = ", ".join(
#             [str(x.item()) for x in self.diag[: self.MAX_PRINT_ENTRIES]]
#         )
#         if len(self.diag) > self.MAX_PRINT_ENTRIES:
#             diagstr += "..."
#         s = f"<{clsname}({self.shape[0]}x{self.shape[1]})[{diagstr}]>"
#         return s


# class BandedLinOp(BaseLinOp):
#     r"""Banded linear operator (composed of diagonals).

#     Given a collection of :math:`k` vectors, this class implements a banded
#     linear operator, where each given vector is a (sub)-diagonal. It is
#     composed by :math:`k` :class:`DiagonalLinOp` operators, thus its memory
#     and runtime is equivalent to storing and running the individual diagonals.

#     .. note::
#       All given vectors must be of appropriate length with respect to their
#       position. For example, a square tridiagonal matrix of shape ``(n, n)``
#       has a main diagonal at index 0 with length ``n``, and two subdiagonals at
#       indices 1, -1 with length ``n - 1``. Still, this linop can also be
#       non-square (unless it is symmetric), as long as all given diagonals fit
#       in the implicit shape.

#     Usage example::

#       diags = {0: some_diag, 1: some_superdiag, -1, some_subdiag}
#       B = BandedLinOp(diags, symmetric=False)
#       w = B @ v
#     """

#     MAX_PRINT_ENTRIES = 20

#     @staticmethod
#     def __initial_checks(indexed_diags, symmetric):
#         """Performs input checks right at initialization."""
#         # extract diagonal lengths and check they are vectors
#         diag_lengths = {}
#         for idx in sorted(indexed_diags):
#             diag = indexed_diags[idx]
#             if len(diag.shape) == 1:
#                 diag_lengths[idx] = len(diag)
#             else:
#                 raise BadShapeError("All diagonals must be vectors!")
#         if not diag_lengths:
#             raise ValueError(f"Empty linop dict? {indexed_diags}")
#         # symmetric mode does not accept negative indices
#         if symmetric:
#             if any(idx < 0 for idx in indexed_diags):
#                 raise BadShapeError(
#                     "Symmetric banded linop only admits nonnegative indices!"
#                     + f" {diag_lengths}"
#                 )
#         #
#         return diag_lengths

#     def __init__(self, indexed_diags, symmetric=False):
#         """Instantiates a banded linear operator.

#         :param indexed_diags: Dictionary in the form ``{idx: diag, ...}`` where
#           ``diag`` is a torch vector containing a diagonal, and ``idx``
#           indicates the location of the diagonal: 0 is the main diagonal, 1 the
#           superdiagonal (``lop[i, i+1]``), -1 the subdiagonal, and so on.
#         :param symmetric: If true, only diagonals with nonnegative indices are
#           admitted. Each positive index will be replicated as a negative one.

#         .. note::
#           The shape of this linear operator is implicitly given by the
#           diagonal lengths and indices. Any inconsistent input will result in
#           a ``BadShapeError``. In particular, symmetric banded matrices must
#           also be square.
#         """
#         # extract diagonal lengths and check they are vectors
#         # also check that symmetric mode does not accept negative indices
#         diag_lengths = self.__initial_checks(indexed_diags, symmetric)
#         # figure out the smallest matrix that fits all diagonals
#         # note that in symmetric mode we need to add the negative indices
#         end_coords = {}
#         height, width = 0, 0
#         for idx, length in diag_lengths.items():
#             i0, j0 = (0, idx) if idx >= 0 else (abs(idx), 0)
#             i1, j1 = i0 + length, j0 + length
#             height, width = max(height, i1), max(width, j1)
#             end_coords[idx] = (i1, j1)
#         if symmetric:
#             for idx, length in diag_lengths.items():
#                 if idx > 0:
#                     i0, j0, i1, j1 = idx, 0, idx + length, length
#                     height, width = max(height, i1), max(width, j1)
#                     end_coords[-idx] = (i1, j1)
#         # check that all given diagonals fit the linop shape tightly
#         inconsistent_idxs = set()
#         for idx, (i1, j1) in end_coords.items():
#             if (i1 != height) and (j1 != width):
#                 inconsistent_idxs.add(idx)
#         #
#         if inconsistent_idxs:
#             raise BadShapeError(
#                 f"Inconsistent diagonal indices/lengths! {diag_lengths}, "
#                 + f"triggered by indices {inconsistent_idxs} "
#                 + f" for shape {(height, width)}"
#             )
#         # if symmetric, linop must be square
#         if symmetric:
#             if height != width:
#                 raise BadShapeError(
#                     f"Symmetric banded linop must be square! {diag_lengths}"
#                 )
#         # done checking, initialize object
#         self.diags = {i: DiagonalLinOp(d) for i, d in indexed_diags.items()}
#         self.symmetric = symmetric
#         super().__init__((height, width))

#     def __matmul_helper(self, x, adjoint=False):
#         """Helper method to perform multiple diagonal matmuls."""
#         self.check_input(x, self.shape, adjoint=adjoint)
#         #
#         diags = {}
#         for idx, diag in self.diags.items():
#             diags[idx] = diag
#             if self.symmetric and idx > 0:
#                 diags[-idx] = diag
#         #
#         outdim = self.shape[1] if adjoint else self.shape[0]
#         result = torch.zeros(outdim, dtype=x.dtype, device=x.device)
#         #
#         for idx, d in diags.items():
#             if adjoint:
#                 in_beg = abs(min(idx, 0))
#                 out_beg = max(idx, 0)
#             else:
#                 in_beg = max(idx, 0)
#                 out_beg = abs(min(idx, 0))
#             dlen = d.shape[0]
#             result[out_beg : out_beg + dlen] += d @ x[in_beg : in_beg + dlen]
#         #
#         return result

#     def __matmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         See parent class for more details.
#         """
#         return self.__matmul_helper(x, adjoint=False)

#     def __rmatmul__(self, x):
#         """Adjoint (left) matrix-vector multiplication ``x @ self``.

#         See parent class for more details.
#         """
#         return self.__matmul_helper(x, adjoint=True)

#     def __repr__(self):
#         """Returns a string in the form <BandedLinOp(shape)[idx1,..., sym]>."""
#         clsname = self.__class__.__name__
#         idxs = ", ".join(str(idx) for idx in sorted(self.diags))
#         s = (
#             f"<{clsname}({self.shape[0]}x{self.shape[1]})[{idxs}, "
#             + f"sym={self.symmetric}]>"
#         )
#         return s

#     def to_matrix(self):
#         """Convert this linear operator into a matrix."""
#         # check that all diagonals are of same dtype and device
#         dtypes, devices = zip(
#             *((d.diag.dtype, d.diag.device) for d in self.diags.values())
#         )
#         if len(set(dtypes)) > 1:
#             raise RuntimeError(f"Inconsistent diagonal dtypes! {dtypes}")
#         if len(set(devices)) > 1:
#             raise RuntimeError(f"Inconsistent diagonal devices! {devices}")
#         # create and populate resulting matrix
#         result = torch.zeros(self.shape, dtype=dtypes[0], device=devices[0])
#         for idx, diag in self.diags.items():
#             dlen = len(diag.diag)
#             if idx >= 0:
#                 result[range(0, dlen), range(idx, dlen + idx)] = diag.diag
#             else:
#                 idx = abs(idx)
#                 result[range(idx, dlen + idx), range(0, dlen)] = diag.diag
#         #
#         if self.symmetric:
#             result = result + result.T
#             result[range(len(result)), range(len(result))] /= 2
#         #
#         return result


# # ##############################################################################
# # # PROJECTION LINEAR OPERATORS
# # ##############################################################################
# class OrthProjLinOp(BaseLinOp):
#     r"""Linear operator for an orthogonal projector.

#     Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
#     columns, this class implements the orthogonal projector onto its span,
#     i.e. :math:`Q Q^T`.
#     """

#     def __init__(self, Q):
#         """Object initialization.

#         :param Q: Linear operator of shape ``(h, w)`` with ``h >= w``, expected
#           to be orthonormal (i.e. columns with unit Euclidean norm and all
#           orthogonal to each other). It must implement the left and right
#           matmul operations via the ``@`` operator.

#         .. warning::

#           This class assumes ``Q`` is orthonormal, but this is not checked.
#         """
#         self.Q = Q
#         #
#         h, w = Q.shape
#         if w > h:
#             raise BadShapeError("Projector matrix must be thin!")
#         super().__init__((h, h))  # this sets self.shape also

#     def __matmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         See parent class for more details.
#         """
#         self.check_input(x, self.shape, adjoint=False)
#         if len(x.shape) == 1:
#             result = self.Q @ (x @ self.Q)
#         elif len(x.shape) == 2:
#             result = self.Q @ (x.T @ self.Q).T
#         else:
#             raise RuntimeError(f"Unsupported input shape: {x.shape}")
#         return result

#     def __rmatmul__(self, x):
#         """Adjoint (left) matrix-vector multiplication ``x @ self``.

#         See parent class for more details.
#         """
#         self.check_input(x, self.shape, adjoint=True)
#         if len(x.shape) == 1:
#             result = self.Q @ (x @ self.Q)
#             return result
#         elif len(x.shape) == 2:
#             result = self.Q @ (x @ self.Q).T
#             return result.T
#         else:
#             raise RuntimeError(f"Unsupported input shape: {x.shape}")

#     def __repr__(self):
#         """Returns a string in the form <OrthProjLinOp(Q_shape)>."""
#         clsname = self.__class__.__name__
#         s = f"<{clsname}({self.Q.shape})>"
#         return s


# class NegOrthProjLinOp(OrthProjLinOp):
#     """Linear operator for a negative orthogonal projector.

#     Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
#     columns, this class implements the orthogonal projector :math:`Q Q^T`.
#     Given a "thin" matrix (i.e. height >= width) :math:`Q` of orthonormal
#     columns, this class implements the orthogonal projector onto the space
#     orthogonal to its span, i.e. :math:`(I - Q Q^T)`.
#     """

#     def __matmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         See parent class for more details.
#         """
#         return x - super().__matmul__(x)

#     def __rmatmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         See parent class for more details.
#         """
#         return x - super().__rmatmul__(x)


# # ##############################################################################
# # # TORCH INTEROPERABILITY
# # ##############################################################################
# class TorchLinOpWrapper:
#     """Linear operator that always accepts and produces PyTorch tensors.

#     Since this library operates mainly with PyTorch tensors, but some useful
#     LinOps interface with e.g. NumPy arrays instead, this mixin class acts as a
#     wraper on the ``__matmul__`` and ``__rmatmul__`` operators,
#     so that the operator expects and returns torch tensors, even when the
#     wrapped operator interfaces with NumPy/HDF5. Usage example::

#       # extend NumPy linear operator via multiple inheritance
#       class TorchWrappedLinOp(TorchLinOpWrapper, LinOp):
#           pass
#       lop = TorchWrappedLinOp(...)  # instantiate normally
#       w = lop @ v  # now v can be a PyTorch tensor
#     """

#     @staticmethod
#     def _input_wrapper(x):
#         """Helper method to admit PyTorch tensors."""
#         if isinstance(x, torch.Tensor):
#             return x.cpu().numpy(), x.device
#         else:
#             return x, None

#     @staticmethod
#     def _output_wrapper(x, torch_device=None):
#         """Helper method to produce PyTorch tensors."""
#         if torch_device is not None:
#             return torch.from_numpy(x).to(torch_device)
#         else:
#             return x

#     def __matmul__(self, x):
#         """Forward (right) matrix-vector multiplication ``self @ x``.

#         Casts given PyTorch tensor into NumPy before applying matmul.
#         Then casts produced result back into same PyTorch datatype and device.
#         """
#         x, device = self._input_wrapper(x)
#         result = self._output_wrapper(super().__matmul__(x), device)
#         return result

#     def __rmatmul__(self, x):
#         """Adjoint (left) matrix-vector multiplication ``x @ self``.

#         Casts given PyTorch tensor into NumPy before applying matmul.
#         Then casts produced result back into same PyTorch datatype and device.
#         """
#         x, device = self._input_wrapper(x)
#         result = self._output_wrapper(super().__rmatmul__(x), device)
#         return result

#     def __repr__(self):
#         """Returns a string in the form TorchLinOpWrapper<LinOp ...>."""
#         wrapper = self.__class__.__name__
#         result = f"{wrapper}<{super().__repr__()}>"
#         return result
