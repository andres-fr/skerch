#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for aggregate linops (composite, sum,...).

* Formal tests:
  - Providing an empty collection triggers a ValueError
  - Repr creates correct strings
  - Mismatching shapes trigger a BadShapeError
  - (adjoint-) matmul with vector of wrong size triggers BadShapeError
  - Composing a mix of linops and matrices works
"""


import pytest
import torch

from skerch.linops import (
    CompositeLinOp,
    SumLinOp,
    TransposedLinOp,
    linop_to_matrix,
)
from skerch.utils import BadShapeError, gaussian_noise

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 3e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def sum_shapes(request):
    """Shapes for correctness of SumLinOp."""
    result = [
        ((1, 1),),
        ((1, 1), (1, 1)),
        ((5, 1), (5, 1), (5, 1)),
        ((1, 5), (1, 5), (1, 5)),
        ((13, 23), (13, 23)),
        ((100, 101), (100, 101)),
    ]
    if request.config.getoption("--quick"):
        result = result[:5]
    return result


@pytest.fixture
def composite_sizes(request):
    """Chained shapes for the composite linear operator."""
    result = [
        (1, 1, 1),
        (1, 5, 1),
        (5, 1, 5),
        (10, 10, 10, 10),
        (1, 5, 50, 5, 1),
    ]
    if request.config.getoption("--quick"):
        result = result[:-1]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
class ScalarLinOp:
    """Basic scalar linop with shape and matmul, but is not a baselinop."""

    def __init__(self, shape, scale=1.0):
        """Initializer. See class docstring."""
        self.shape = shape
        self.scale = scale

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``."""
        return x * self.scale

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``."""
        return x * self.scale

    def __repr__(self):
        """Returns a string in the form <classname(shape)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})>"
        return s


# ##############################################################################
# # TESTS
# ##############################################################################
def test_aggregate_formal_and_basic_correctness(sum_shapes):
    """Formal/basic correctness test case for aggregate linops.

    When creating aggregate linops, tests that:

    * Providing an empty collection triggers a ValueError
    * Repr creates correct strings
    * Mismatching shapes trigger a BadShapeError
    * (adjoint-) matmul with vector of wrong size triggers BadShapeError

    And also:
    * Linops look as they should when converted to matrix (also when mixing
      matrices and linops)
    """
    # empty input raises error
    with pytest.raises(ValueError):
        _ = SumLinOp([])
    with pytest.raises(ValueError):
        _ = CompositeLinOp([])
    # correct string conversion
    hw = (5, 5)
    m, l1, l2 = (torch.ones(hw), ScalarLinOp(hw, 1.0), ScalarLinOp(hw, 2.0))
    assert (
        str(SumLinOp((("M", 1, m), ("L1", 1, l1), ("L2", 1, l2))))
        == "M + L1 + L2"
    ), "Unexpected repr for sum linop!"
    assert (
        str(SumLinOp((("M", 0, m), ("L1", 0, l1), ("L2", 0, l2))))
        == "-M - L1 - L2"
    ), "Unexpected repr for sum linop!"
    assert (
        str(CompositeLinOp((("M", m), ("L1", l1), ("L2", l2)))) == "M @ L1 @ L2"
    ), "Unexpected repr for composite linop!"
    # matrices and linops can be mixed and result is correct
    slop = SumLinOp((("M", 1, m), ("L1", 1, l1), ("L2", 1, l2)))
    clop = CompositeLinOp((("M", m), ("L1", l1), ("L2", l2)))
    assert (
        linop_to_matrix(slop, m.dtype, m.device) == m + torch.eye(len(m)) * 3
    ).all(), "Incorrect sum linop to matrix?"
    assert (
        linop_to_matrix(clop, m.dtype, m.device) == m * 2
    ).all(), "Incorrect composite linop to matrix?"
    # shape inconsistencies raise errors
    for shapes in sum_shapes:
        # create list of linops with inconsistent shapes
        linops = [ScalarLinOp(s) for s in shapes]
        linops.append(ScalarLinOp((shapes[0][0] + 1, shapes[0][1])))
        # test that sum linop raises error
        with pytest.raises(BadShapeError):
            _ = SumLinOp((f"L{i}", 1, l) for i, l in enumerate(linops))
        # test that composite linop raises error
        with pytest.raises(BadShapeError):
            _ = CompositeLinOp((f"L{i}", l) for i, l in enumerate(linops))


def test_sum_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    sum_shapes,
):
    """Test case for correctness of ``SumLinOp``.

    Creates a set of random matrices. Then composes them explicitly, and
    creates a ``SumLinOp``. Tests that:
    * Matmul and rmatmul with linop (fwd and adj) is same as with matrix
    * Transposed linop is correct (fwd and adj)
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, _ in dtypes_tols.items():
                for shapes in sum_shapes:
                    # sample random submatrices
                    submatrices = []
                    for i, shape in enumerate(shapes):
                        submatrices.append(
                            gaussian_noise(
                                shape,
                                seed=seed + i,
                                dtype=dtype,
                                device=device,
                            )
                        )
                    # all-sum
                    mat = sum(submatrices)
                    lop = SumLinOp(
                        (f"M_{i}", 1, m) for i, m in enumerate(submatrices, 1)
                    )
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=False)
                    assert (mat == lopmat).all(), "Incorrect sum+ (fwd) linop!"
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=True)
                    assert (mat == lopmat).all(), "Incorrect sum+ (adj) linop!"
                    lopT = TransposedLinOp(lop)
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=False
                    )
                    assert (
                        mat.H == lopmatT
                    ).all(), "Incorrect sum+ transposition! (fwd)"
                    lopmatT = linop_to_matrix(lopT, dtype, device, adjoint=True)
                    assert (
                        mat.H == lopmatT
                    ).all(), "Incorrect sum+ transposition! (adj)"
                    # alternating sum and diff: M1 - M2 + M3 ...
                    mat = submatrices[0].clone()
                    for i, m in enumerate(submatrices[1:]):
                        if i % 2 == 0:
                            mat -= m
                        else:
                            mat += m
                    lop = SumLinOp(
                        (f"M_{i}", i % 2, m)
                        for i, m in enumerate(submatrices, 1)
                    )
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=False)
                    assert (
                        mat == lopmat
                    ).all(), "Incorrect alternating sum linop! (fwd)"
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=True)
                    assert (
                        mat == lopmat
                    ).all(), "Incorrect alternating sum linop! (adj)"
                    lopT = TransposedLinOp(lop)
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=False
                    )
                    assert (
                        mat.H == lopmatT
                    ).all(), "Incorrect alternating sum transposition! (fwd)"
                    lopmatT = linop_to_matrix(lopT, dtype, device, adjoint=True)
                    assert (
                        mat.H == lopmatT
                    ).all(), "Incorrect alternating sum transposition! (adj)"


def test_composite_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    composite_sizes,
):
    """Test case for correctness of ``CompositeLinOp``.

    Creates a set of random matrices. Then composes them explicitly, and
    creates a ``CompositeLinOp``. Tests that:
    * Matmul and rmatmul with linop (fwd and adj) is same as with matrix
    * Transposed linop is correct (fwd and adj)
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for sizes in composite_sizes:
                    # sample random submatrices
                    submatrices = []
                    for i, (h, w) in enumerate(zip(sizes[:-1], sizes[1:])):
                        submatrices.append(
                            gaussian_noise(
                                (h, w),
                                seed=seed + i,
                                dtype=dtype,
                                device=device,
                            )
                        )
                    mat = submatrices[0]
                    for m in submatrices[1:]:
                        mat = mat @ m
                    lop = CompositeLinOp(
                        (f"M_{i}", m) for i, m in enumerate(submatrices, 1)
                    )
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=False)
                    assert torch.allclose(
                        mat, lopmat, atol=tol
                    ), "Incorrect composite linop! (fwd)"
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=True)
                    assert torch.allclose(
                        mat, lopmat, atol=tol
                    ), "Incorrect composite linop! (adj)"
                    lopT = TransposedLinOp(lop)
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=False
                    )
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect composite transposition! (fwd)"
                    lopmatT = linop_to_matrix(lopT, dtype, device, adjoint=True)
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect composite transposition! (adj)"
