#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for linear operators.

This PyTorch library is intended to work with (potentially matrix-free) linear
operators that support left- and right- matrix multiplication (via e.g.
``v @ lop`` and ``lop @ v``) as well as the ``.shape`` attribute.

This module implements basic support for said functionality, as well as some
default linear operators (composite, diagonal...).
"""


import warnings
import torch

from .utils import BadShapeError, htr, COMPLEX_DTYPES


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
    extended with further functionality via :meth:`.matmul` and
    :meth:`.rmatmul` (:meth:`.vecmul` and :meth:`.rvecmul` also possible, but
    matrix-matrix multiplications should be prefered to leverage
    parallelization).

    :param shape: ``(height, width)`` of linear operator.
    :param batch: If ``matmul`` is implemented and we are e.g. multiplying
      ``self @ x`` where ``x`` has ``k`` vectors, the default behaviour for
       ``batch=None`` is to attempt a matmul with all ``k`` vectors at once,
       and then go one-by-one if anything goes wrong. Setting an integer
       ``batch`` size allows users to overcome memory errors by limiting the
       matmul operation to at most ``batch`` at once, while keeping
       parallelization within that batch.
    """

    def __init__(self, shape, batch=None):
        """Initializer. See class docstring."""
        try:
            h, w = shape
        except Exception as e:
            raise ValueError(f"Malformed shape? {shape}")
        if len(shape) != 2:
            raise BadShapeError("Shape must be a (height, width) pair!")
        if h <= 0 or w <= 0:
            raise BadShapeError(f"Empty linop with shape {shape}?")
        self.shape = shape
        self.batch = batch

    def __repr__(self):
        """Returns a string in the form <classname(shape)>."""
        clsname = self.__class__.__name__
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){batch_s}>"
        return s

    @staticmethod
    def check_input(x, lop_shape, adjoint):
        """Checking that input is a mat/vec of the right shape.

        :param x: The input to this linear operator. It should be either a
          vector or a matrix of matching shape.
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

    def __matmul_vectorizer(self, x, adjoint=False):
        """Helper method to run ``vecmul``s, one vector of ``x`` at a time.

        :param x: Assumed to be a matrix of matching dimensions. If ``x`` is
          a vector, call :meth:`.vecmul` or :meth:`.rvecmul` directly.
        """
        lop_h, lop_w = self.shape
        x_h, x_w = x.shape
        num_vecs = x_h if adjoint else x_w
        #
        result = torch.zeros(
            (num_vecs, lop_w) if adjoint else (lop_h, num_vecs),
            dtype=x.dtype,
            device=x.device,
        )
        for i in range(num_vecs):
            if adjoint:
                result[i, :] = self.rvecmul(x[i, :])
            else:
                result[:, i] = self.vecmul(x[:, i])
        #
        return result

    def __matmul_batcher(self, x, adjoint=False, batch=None):
        """Helper to dispatch between (batched) ``matmul`` and ``vecmul``."""
        self.check_input(x, self.shape, adjoint)
        # if x is a vector, just run vecmul
        if len(x.shape) == 1:
            result = self.rvecmul(x) if adjoint else self.vecmul(x)
        # if no batch given: try matmul, if any issue warn and try vecmul
        elif batch is None:
            try:
                result = self.rmatmul(x) if adjoint else self.matmul(x)
            except Exception as ee:
                warnings.warn(
                    f"{self} couldn't run matmat due to: '{ee}' "
                    "Running matvec instead",
                    RuntimeWarning,
                )
                result = self.__matmul_vectorizer(x, adjoint)
        # if batch given: try matmul with that batch, if any issue raise error
        # since desired batch is not possible
        else:
            num_vecs = x.shape[0 if adjoint else 1]
            result = []
            for beg in range(0, num_vecs, batch):
                end = beg + batch
                try:
                    if adjoint:
                        result.append(self.rmatmul(x[beg:end, :]))
                    else:
                        result.append(self.matmul(x[:, beg:end]))
                except Exception as e:
                    msg = (
                        f"Couldn't perform batched matmat! Implement matmul "
                        f"or set batch to None {self, x}"
                    )
                    raise RuntimeError(msg) from e
            stack = torch.vstack if adjoint else torch.hstack
            result = stack(result)
        #
        return result

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to :meth:`.matmul` for vectors or matrices."""
        result = self.__matmul_batcher(x, adjoint=False, batch=self.batch)
        return result

    def __rmatmul__(self, x):
        """Convenience wrapper to :meth:`.rmatmul` for vectors or matrices."""
        result = self.__matmul_batcher(x, adjoint=True, batch=self.batch)
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

    # to-be-implemented in class extensions
    def vecmul(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        .. note::

          If :meth:`.matmul` is implemented, ``@`` will try that first, to
          leverage parallelization. If anything goes wrong, it will fallback to
          this method, which processes a single vector each time.

        :param x: Expected to be a vector of shape ``(w,)`` where this linop
          has shape ``(h, w)``. Note that inputs of shape ``(w, k)`` will be
          automatically passed to this method as ``k`` vectors of length ``w``.
        :returns: A vector of shape ``(h,)`` equaling ``self @ x``.
        """
        return self.matmul(x)

    def rvecmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        Like :meth:`.vecmul`, but with adjoint shapes.
        """
        return self.rmatmul(x)

    def matmul(self, x):
        """ """
        raise NotImplementedError("Matmul not implemented!")

    def rmatmul(self, x):
        """ """
        raise NotImplementedError("Rmatmul not implemented!")


class ByVectorLinOp(BaseLinOp):
    """Matrix-free linop computed vector by vector.

    This matrix-free operator generates each row (or column) one by
    one during matrix multiplication. Useful when memory is a bottleneck.

    Users can decide which modality to use at instantiation via the ``by_row``
    boolean flag. Extensions of this class should implement :meth:`.get_vector`
    accordingly to return vectors of the right shape.

    .. note::
      The ``by_row`` flag has implications in terms of memory and runtime.
      If true, for a ``lop`` of shape ``(h, w)``, the ``lop @ x`` matrix-vector
      multiplication will call :meth:`.get_vector`` ``h`` times, and perform
      ``h`` dot products of dimension ``w``. If false, it will perform ``w``
      scalar products of dimension ``h``. In the case of ``x @ lop``,
      the scalar and dot products are swapped.

      Therefore, developers need to override :meth:`.get_vector` taking this
      flag into account, and users should set it to the scenario that is most
      efficient (e.g. by-column is generally more efficient when ``h > w``).
    """

    def __init__(self, shape, by_row=False):
        """Initializer. See class docstring."""
        super().__init__(shape)
        self.by_row = by_row

    def get_vector(self, idx, input_device):
        """Method to gather vector entries for this linear operator.

        Override this method with the desired behaviour. For a shape of
        ``(h, w)``, it should return vectors of length ``w`` if
        ``self.by_row`` is true, and lenght ``h`` otherwise.

        :param idx: Index of the row/column to be sampled. It will go from 0
          to ``dims - 1``, both included, where ``dims`` is ``h`` if
          ``self.by_row`` is true, and ``w`` otherwise.
        :param input_device: The device of the input tensor that this linop
          was called on. The output of this method should generally be in the
          same device.
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
    """Hermitian transposition of a linear operator.

    :param lop: Any object supporting a shape ``(h, w)`` attribute as well as
      the right- and left- matmul operator ``@``.

    Given a linear operator :math:`A`, real or complex, this class wraps its
    functionality, such that ``TransposedLinOp(lop)`` behaves line the
    (Hermitian) transposition :math:`A^H`. This is done by swapping dimensions
    and matmul methods leveraging the following identity:

    :math:`A^H v = ((A^H v)^H)^H = (v^H A)^H`.
    """

    def __init__(self, lop):
        """Initializer. See class docstring."""
        if isinstance(lop, TransposedLinOp):
            raise ValueError("LinOp is already transposed! use x.lop")
        self.lop = lop
        self.shape = self.lop.shape[::-1]

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to :meth:`.matmul` for vectors or matrices."""
        # (A.H @ x) = (A.H @ x).H.H = (x.H @ A).H
        # x_vec = len(x.shape) == 1
        # result = (x.conj().T @ self.lop).T
        # # result.conj() did not work with multiprocessing (bug?)
        # try:
        #     result.imag *= -1
        # except RuntimeError:
        #     pass

        # (A.H @ x) = (A.H @ x).H.H = (x.H @ A).H
        # in-place conjugation did not work with multiprocessing (bug?)
        result = htr((htr(x, in_place=False) @ self.lop), in_place=False)
        return result

    def __rmatmul__(self, x):
        """Convenience wrapper to :meth:`.rmatmul` for vectors or matrices."""
        # (x @ A.H) = (x @ A.H).H.H = (A @ x.H).H
        # x_vec = len(x.shape) == 1
        # result = (self.lop @ x.conj().T).T
        # # result.conj() did not work with multiprocessing (bug?)
        # try:
        #     result.imag *= -1
        # except RuntimeError:
        #     pass

        # (x @ A.H) = (x @ A.H).H.H = (A @ x.H).H
        # in-place conjugation did not work with multiprocessing (bug?)
        result = htr(self.lop @ htr(x, in_place=False), in_place=False)
        return result

    def t(self):
        """Undo transposition"""
        return self.lop

    def __repr__(self):
        """Returns a string in the form (str(lop)).H"""
        return f"({str(self.lop)}).H"


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
        """Instantiates a summation linear operator. See class docstring."""
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
        for o, s in zip(self.operators[1:], self.signs[1:]):
            if s:
                result += x @ o
            else:
                result -= x @ o
        return result

    def __repr__(self):
        """Returns a string in the form op1 + op2 + op3 ..."""
        signs_str = ["+ " if s else "- " for s in self.signs]
        result = ("-" if not self.signs[0] else "") + self.names[0]
        for s, n in zip(self.signs[1:], self.names[1:]):
            result += (" + " if s else " - ") + n
        return result


# ##############################################################################
# # DIAGONAL/BANDED LINEAR OPERATORS
# ##############################################################################
class DiagonalLinOp(BaseLinOp):
    r"""Diagonal linear operator.

    Given a vector ``v`` of ``d`` dimensions, this class implements a diagonal
    linear operator of shape ``(d, d)`` via left- and right-matrix
    multiplication, as well as the ``shape`` attribute, only requiring linear
    (:math:`\mathcal{O}(d)`) memory and runtime.

    :param diag: Vector to be casted as diagonal linop.
    """

    MAX_PRINT_ENTRIES = 20

    def __init__(self, diag):
        """Initializer. See class docstring."""
        if len(diag.shape) != 1 or diag.numel() <= 0:
            raise BadShapeError("Diag must be a nonempty vector!")
        self.diag = diag
        super().__init__((len(diag),) * 2)  # this sets self.shape also

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        self.check_input(x, self.shape, adjoint=False)
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
        self.check_input(x, self.shape, adjoint=True)
        result = x * self.diag
        return result

    def to_matrix(self):
        """Efficiently convert this linear operator into a matrix."""
        result = torch.diag(self.diag)
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
      The shape of this linear operator is implicitly given by the
      diagonal lengths and indices. Any inconsistent input will result in
      a ``BadShapeError``. In particular, symmetric banded matrices must
      also be square.
      For example, a square tridiagonal matrix of shape ``(n, n)``
      has a main diagonal at index 0 with length ``n``, and two subdiagonals at
      indices 1, -1 with length ``n - 1``. Still, this linop can also be
      non-square (unless it is symmetric), as long as all given diagonals fit
      in the implicit shape.

    Usage example::

      diags = {0: some_diag, 1: some_superdiag, -1, some_subdiag}
      B = BandedLinOp(diags, symmetric=False)
      w = B @ v


    :param indexed_diags: Dictionary in the form ``{idx: diag, ...}`` where
      ``diag`` is a torch vector containing a diagonal, and ``idx``
      indicates the location of the diagonal: 0 is the main diagonal, 1 the
      superdiagonal (``lop[i, i+1]``), -1 the subdiagonal, and so on.
    :param symmetric: If true, only diagonals with nonnegative indices are
      admitted. Each positive index will be replicated as a negative one.
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
        """Initializer. See class docstring."""
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
        self.check_input(x, self.shape, adjoint=adjoint)
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
        """Efficiently convert this linear operator into a matrix."""
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
