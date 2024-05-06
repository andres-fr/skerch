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

from .utils import BadShapeError, NoFlatError


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
# # BASE LINEAR OPERATOR INVOLVING RANDOMNESS
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


# ##############################################################################
# # COMPOSITE LINEAR OPERATOR
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


# ##############################################################################
# # DIAGONAL LINEAR OPERATOR
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


# ##############################################################################
# # DIAGONAL LINEAR OPERATOR
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
        result = self.Q @ (x.T @ self.Q).T
        return result

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        self.check_input(x, adjoint=True)
        result = self.Q @ (x @ self.Q).T
        return result.T

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
# # SKETCHED (SUB)-DIAGONAL LINOP
# ##############################################################################
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
