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
