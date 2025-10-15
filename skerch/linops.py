#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Basic utilities for (matrix-free) linear operators."""


import torch

from .utils import BadShapeError, htr


# ##############################################################################
# # CORE LINOP FUNCTIONALITY
# ##############################################################################
def linop_to_matrix(lop, dtype=torch.float32, device="cpu", adjoint=False):
    """Convert a linop to a matrix via one-hot matrix-vector products.

    :param adjoint: If true, one-hot products are done via ``v_i @ lop``.
      Otherwise ``lop @ v_i``.
    :returns: Matrix equivalent to the given ``lop``, on the given ``dtype``
      and ``device``.
    """
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


def check_linop_input(x, lop_shape, adjoint):
    """Checks that input is a matrix or vector of the right shape.

    :param x: The input to this linear operator. It should be either a
      vector or a matrix of matching shape to ``lop_shape``.
    :param bool adjoint: If true, ``x @ lop`` is assumed, otherwise
      ``lop @ x``.
    :raises: :class:`BadShapeError` if there is any shape mismatch.
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


class BaseLinOp:
    """Base class for linear operators.

    Implements the ``.shape`` attribute and basic matmul functionality with
    vectors and matrices (also via the ``@`` operator). Intended to be
    extended with further functionality via :meth:`.matmul` and
    :meth:`.rmatmul`. See documentation and codebase for examples.

    .. note::
      Inputs to ``matmul, rmatmul`` can be vectors or matrices, but
      implementations are responsibe to handle matrix-matrix parallelization.

    :param shape: ``(height, width)`` of linear operator.
    :param batch: When calling ``self @ x`` against a matrix with several
      vectors, the default ``batch=None`` will call :meth:`.matmul` with the
      whole ``x`` matrix at once. If a different batch is provided, ``x``
      will be split in chunks of ``batch`` vectors, which will be fed to
      :meth:`.matmul` sequentially. This is useful to eg. prevent
      out-of-memory errors due to processing too large chunks at once.
    """

    def __init__(self, shape, batch=None):
        """Initializer. See class docstring."""
        try:
            h, w = shape
        except Exception as e:
            raise ValueError(f"Malformed shape? {shape}") from e
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

    def _matmul_helper(self, x, adjoint=False):
        """Reshape and distribute input to the corresponding operation.

        :param x: Vector or matrix to be multiplied with.
        :param adjoint: True if adjoint matmul is intended.
        :returns: Result of the matrix multiplication.
        """
        check_linop_input(x, self.shape, adjoint)
        xvec = len(x.shape) == 1
        num_vecs = 1 if xvec else x.shape[0 if adjoint else 1]
        #
        if xvec:
            x = x.unsqueeze(0 if adjoint else 1)
        #
        if (self.batch is None) or xvec:
            result = self.rmatmul(x) if adjoint else self.matmul(x)
        else:
            result = []
            for beg in range(0, num_vecs, self.batch):
                end = beg + self.batch
                if adjoint:
                    result.append(self.rmatmul(x[beg:end, :]))
                else:
                    result.append(self.matmul(x[:, beg:end]))
            result = torch.vstack(result) if adjoint else torch.hstack(result)
        #
        if xvec and len(result.shape) == 2:
            result.squeeze_()
        return result

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to perform matmul on vectors or matrices."""
        result = self._matmul_helper(x, adjoint=False)
        return result

    def __rmatmul__(self, x):
        """Convenience wrapper to perform rmatmul on vectors or matrices."""
        result = self._matmul_helper(x, adjoint=True)
        return result

    def __imatmul__(self, x):
        """Assignment matmul operator ``@=``.

        .. warning::
          This method is deactivated by default since linear operators may be
          matrix-free.
        """
        raise NotImplementedError(
            "Unsupported matmul assignment: not matrix-free compatible!"
        )

    def t(self):
        """Return (Hermitian) transposition of this lop.

        Equivalent to ``TransposedLinOp(self)``, see :class:`TransposedLinOp`.
        """
        return TransposedLinOp(self)

    def matmul(self, x):
        """Implement with the desired functionality. See class docstring."""
        raise NotImplementedError("Matmul not implemented!")

    def rmatmul(self, x):
        """Implement with the desired functionality. See class docstring."""
        raise NotImplementedError("Rmatmul not implemented!")


class ByBlockLinOp(BaseLinOp):
    """Matrix-free operator computed by blocks of submatrices.

    Consider a large matrix that does not fit in memory. Still, we can perform
    matrix multiplications by sequentially loading smaller blocks in memory,
    and then aggregating the result. This is the main motivation for this
    by-block linear operator:

    * At instantiation, users determine the ``by_row`` and ``blocksize``
      parameters, which determine how many rows/columns will be internally
      used at once.
    * At runtime, this class calls sequentially the :meth:`get_block` method,
      performs the partial matrix-multiplications and aggregates them.
    * At development, extensions of this class only have to implement
      :meth:`.get_block` accordingly to return blocks of the right shape.

    .. note::
      The ``by_row`` flag has implications in terms of memory and runtime.
      If true, for a ``lop`` of shape ``(h, w)`` and block size 1, the
      ``lop @ x`` matrix-vector multiplication will call :meth:`.get_vector`
      ``h`` times, and perform ``h`` dot products of dimension ``w``. If false,
      it will perform ``w`` scalar products of dimension ``h``. In the case of
      ``x @ lop``, the number of scalar and dot products are swapped.

      Therefore, developers need to override :meth:`.get_block` taking this
      flag into account, and users should set it to the scenario that is most
      efficient (e.g. by-column is generally more efficient when ``h > w``).
      See :class:`skerch.measurements` for examples.

    :param by_row: If true, blocks are groups of rows, otherwise columns.
    :param blocksize: If integer, determines how many rows/columns each block
      has, and a row will be a matrix.
    :param batch: If the input to matmul is a matrix itself, this determines
      how many vectors are computed at once. Note that this is different to
      ``blocksize``, which refers to the blocks of *this* linop.
    """

    def __init__(self, shape, by_row=False, batch=None, blocksize=1):
        """Initializer. See class docstring."""
        if blocksize < 1:
            raise ValueError("Block size must be positive!")
        super().__init__(shape, batch)
        self.by_row = by_row
        self.blocksize = blocksize
        self.num_vecs = shape[0] if by_row else shape[1]
        self.num_blocks = self.get_idx_coords(self.num_vecs - 1)[0] + 1

    def get_vector_idxs(self, block_idx):
        """Retrieves a range with vector indices corresponding to a block.

        :param block_idx: Integer between ``0`` and ``self.num_blocks - 1``.
        :returns: ``range(beg, end)`` with the vector indices corresponding
          to the ``block_idx`` block.
        """
        if not (0 <= block_idx < self.num_blocks):
            raise ValueError(f"Block idx out of bounds! {block_idx}")
        #
        beg = block_idx * self.blocksize
        end = min(self.num_vecs, beg + self.blocksize)
        return range(beg, end)

    def get_idx_coords(self, idx):
        """Retrieve vector index in block coordinates.

        Useful if we want to retrieve a given vector from this linop: since
        this linop is defined in a by-block fashion, we first need to know
        which block to retrieve, and then which vector index within the
        retrieved block.

        :param idx: Integer between ``0`` and ``N - 1``, where ``N`` is
          the number of rows, if ``by_row`` is true, or columns otherwise.
        :returns: Pair of ints ``(b, v)``, such that ``vec_idx`` is the
          ``v``-th element of the ``b``-th block, e.g. in a by-row linop,
          ``lop[idx] = lop.get_block(idxs, dtype, device)[idx]``.
        """
        if not (0 <= idx < self.num_vecs):
            raise ValueError(f"Vector idx out of bounds! {idx}")
        block_idx, vec_idx = divmod(idx, self.blocksize)
        return (block_idx, vec_idx)

    def get_block(self, block_idx, input_dtype, input_device):
        """Method to gather a block (matrix) from this linear operator.

        Override this method with the desired behaviour. For a shape of
        ``(h, w)``, it should return matrices of shape ``(block, w)`` if
        ``self.by_row`` is true, and ``(h, block)`` otherwise.

        .. note::
          If ``blocksize==1``, returning vectors may work in some cases, but
          it is recommended to still return matrices (where one of the
          dimensions equals 1).

        :param block_idx: Index of the block to be returned. Use the auxiliary
          method :meth:`get_vector_idxs` if you need to know which vector
          indices are associated to this block index. The attributes
          ``self.num_vecs, self.num_blocks`` may also be helpful.
        :param input_dtype: The dtype of the input tensor that this linop
          was called on. The output of this method should generally be in the
          same device.
        :param input_device: The device of the input tensor that this linop
          was called on. The output of this method should generally be in the
          same device.
        """
        raise NotImplementedError

    def get_blocks(self, dtype, device="cpu"):
        """Yields all blocks in ascending order with the corresponding idxs."""
        for b_i in range(self.num_blocks):
            block = self.get_block(b_i, dtype, device)
            idxs = self.get_vector_idxs(b_i)
            yield block, idxs

    def to_matrix(self, dtype, device="cpu"):
        """Converts this linop to a matrix of given ``dtype``, ``device``."""
        result = torch.empty(self.shape, dtype=dtype, device=device)
        #
        for block, idxs in self.get_blocks(dtype, device):
            if self.by_row:
                result[idxs, :] = block
            else:
                result[:, idxs] = block
        #
        return result

    def _bb_matmul_helper(self, x, adjoint=False):
        """Reshape and distribute input to the corresponding operation.

        :param x: Vector or matrix to be multiplied with.
        :param adjoint: True if adjoint matmul is intended.
        :returns: Result of the matrix multiplication.
        """
        h, w = self.shape
        if adjoint:
            n, xw = x.shape
            result = torch.zeros((n, w), device=x.device, dtype=x.dtype)
        else:
            xh, n = x.shape
            result = torch.zeros((h, n), device=x.device, dtype=x.dtype)
        #
        for b_i in range(self.num_blocks):
            idxs = self.get_vector_idxs(b_i)
            if adjoint and self.by_row:
                result += x[:, idxs] @ self.get_block(b_i, x.dtype, x.device)
            elif adjoint and not self.by_row:
                result[:, idxs] = x @ self.get_block(b_i, x.dtype, x.device)
            elif not adjoint and self.by_row:
                result[idxs, :] = self.get_block(b_i, x.dtype, x.device) @ x
            elif not adjoint and not self.by_row:
                result += self.get_block(b_i, x.dtype, x.device) @ x[idxs, :]
            else:
                raise RuntimeError("This should never happen!")
        #
        return result

    def matmul(self, x):
        """Performs right matrix-multiplication."""
        return self._bb_matmul_helper(x, adjoint=False)

    def rmatmul(self, x):
        """Performs left (adjoint) matrix-multiplication."""
        return self._bb_matmul_helper(x, adjoint=True)

    def __repr__(self):
        """Returns a string in the form <classname(shape), attr=value, ...>."""
        clsname = self.__class__.__name__
        byrow_s = ", by row" if self.by_row else ", by col"
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        block_s = f", blocksize={self.blocksize}"
        #
        feats = f"{byrow_s}{batch_s}{block_s}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){feats}>"
        return s


class TransposedLinOp:
    """Hermitian transposition of a linear operator.

    Given a linear operator :math:`A`, real or complex, this class wraps its
    functionality, such that ``TransposedLinOp(lop)`` behaves line the
    (Hermitian) transposition :math:`A^H`. This is done by swapping dimensions
    and matmul methods leveraging the following identity:

    :math:`A^H v = ((A^H v)^H)^H = (v^H A)^H`.

    Usage example::

      lopT = TransposedLinOp(lop)

    :param lop: Any object supporting a shape ``(h, w)`` attribute as well as
      the right- and left- matmul operator ``@``.
    """

    def __init__(self, lop):
        """Initializer. See class docstring."""
        if isinstance(lop, TransposedLinOp):
            raise ValueError("LinOp is already transposed! use x.lop")
        self.lop = lop
        self.shape = self.lop.shape[::-1]

    # operator interfaces
    def __matmul__(self, x):
        """Convenience wrapper to perform matmul on vectors or matrices."""
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
        """Convenience wrapper to perform rmatmul on vectors or matrices."""
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
        """Undo transposition: returns original ``lop``."""
        return self.lop

    def __repr__(self):
        """Returns a string in the form (str(lop)).H."""
        return f"({str(self.lop)}).H"


# ##############################################################################
# # AGGREGATE LINEAR OPERATORS
# ##############################################################################
class CompositeLinOp:
    """Matrix-free composite linear operator.

    This class composes an ordered collection of operators ``[A, B, C, ...]``
    into ``A @ B @ C ...`` in a matrix-free fashion.

    .. warning::
      Using this class could be more inefficient than directly computing the
      composed operator, e.g. if ``A.shape = (1, 1000)`` and
      ``B.shape = (1000, 1)``, then computing the scalar ``C = A @ B`` and then
      applying it is more efficient than keeping a ``CompositeLinearoperator``
      with ``A, B`` (in terms of both memory and computation). This class does
      not check for such cases, users are encouraged to take this into account.
      Note that composite linops can also be nested.


    :param named_operators: Ordered collection in the form
      ``[(n_1, o_1), ...]`` where each ``n_i`` is a string with the name of
      operator ``o_i``. Each ``o_i`` operator must implement ``__matmul__``
      and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
      object will then become the composite operator ``o_1 @ o_2 @ ...``
    """

    def __init__(self, named_operators):
        """Initializer. See class docstring."""
        if not named_operators:
            raise ValueError(f"Empty linop collection? {named_operators}")
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
    """Matrix-free sum of linear operators.

    Given a collection of same-shape linear operators ``A, B, C, D ...``, this
    class implements the sum ``A + B + C - D ...`` in a matrix-free fashion.

    :param named_signed_operators: Collection in the form
      ``{(n_1, s_i, o_1), ...}`` where each ``n_i`` is a string with the name
      of operator ``o_i``, and ``s_i`` is a boolean: if true, this operator
      is to be added, otherwise subtracted.
      Each ``o_i`` operator must implement ``__matmul__``
      and ``__rmatmul__`` as well as the ``shape = (h, w)`` attribute. This
      object will then become the operator ``o_1 + o_2 - ...``. All operators
      must also have same shape.
    """

    def __init__(self, named_signed_operators):
        """Instantiates a summation linear operator. See class docstring."""
        if not named_signed_operators:
            raise ValueError(
                f"Empty linop collection? {named_signed_operators}"
            )
        self.names, self.signs, self.operators = zip(*named_signed_operators)
        shapes = [o.shape for o in self.operators]
        for shape in shapes:
            if shape != shapes[0]:
                raise BadShapeError(f"All shapes must be equal! {shapes}")
        super().__init__(shapes[0])  # this sets self.shape also

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        check_linop_input(x, self.shape, adjoint=False)
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
        check_linop_input(x, self.shape, adjoint=True)
        result = x @ self.operators[0]
        for o, s in zip(self.operators[1:], self.signs[1:]):
            if s:
                result += x @ o
            else:
                result -= x @ o
        return result

    def __repr__(self):
        """Returns a string in the form op1 + op2 + op3 ..."""
        result = ("-" if not self.signs[0] else "") + self.names[0]
        for s, n in zip(self.signs[1:], self.names[1:]):
            result += (" + " if s else " - ") + n
        return result


# ##############################################################################
# # DIAGONAL/BANDED LINEAR OPERATORS
# ##############################################################################
class DiagonalLinOp(BaseLinOp):
    r"""Diagonal matrix-free linear operator.

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
        check_linop_input(x, self.shape, adjoint=False)
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
        check_linop_input(x, self.shape, adjoint=True)
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
    r"""Banded matrix-free linear operator.

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
                + f" for shape {(height, width)} and symmetric={symmetric}."
            )
        # diags must be of same dtype and device
        # diag_dtypes = [d.dtype for d in indexed_diags.values()]
        # diag_devices = [d.device for d in indexed_diags.values()]
        diag_dtypes, diag_devices = zip(
            *((d.dtype, d.device) for d in indexed_diags.values())
        )
        if len(set(diag_dtypes)) != 1 or len(set(diag_devices)) != 1:
            raise ValueError("Inconsistent diagonal dtypes/devices!")
        # done checking, initialize object
        self.diags = {i: DiagonalLinOp(d) for i, d in indexed_diags.items()}
        self.symmetric = symmetric
        super().__init__((height, width))

    def __matmul_helper(self, x, adjoint=False):
        """Helper method to perform multiple diagonal matmuls."""
        check_linop_input(x, self.shape, adjoint=adjoint)
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
        """Convert this linear operator into a matrix.

        Datatype and device are borrowed from the first diagonal that was
        passed to the constructor.
        """
        # check that all diagonals are of same dtype and device
        dtypes, devices = zip(
            *((d.diag.dtype, d.diag.device) for d in self.diags.values())
        )
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

    Since ``skerch`` is built on top of PyTorch, but some other useful
    LinOps interface with e.g. NumPy arrays instead, this mixin class acts as a
    wraper on the ``__matmul__`` and ``__rmatmul__`` operators,
    so that the operator expects and returns torch tensors, even when the
    wrapped operator interfaces with NumPy/HDF5.

    This facilitates interoperability between ``skerch`` and other python
    linops. Usage example::

      # extend NumPy linear operator via multiple inheritance
      class TorchWrappedNumpyLinOp(TorchLinOpWrapper, NumpyLinOp):
          pass
      lop = TorchWrappedNumpyLinOp(...)  # instantiate normally
      w = lop @ v  # now v can be a PyTorch tensor on any device
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
        wrapper = "TorchLinOpWrapper"  # self.__class__.__name__
        result = f"{wrapper}<{super().__repr__()}>"
        return result
