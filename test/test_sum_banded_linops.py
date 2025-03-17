#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for sum and banded linops."""


import pytest
import torch

from skerch.linops import BandedLinOp, SumLinOp
from skerch.utils import BadShapeError, gaussian_noise

from . import linop_to_matrix, rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes():
    """Absolute error tolerance for float64."""
    return [torch.float64, torch.float32]


@pytest.fixture
def sum_shapes(request):
    """Shapes for correctness of SumLinOp."""
    result = [
        ((1, 1),),
        ((1, 1), (1, 1)),
        ((5, 1), (5, 1), (5, 1)),
        ((1, 5), (1, 5), (1, 5)),
        ((23, 35), (23, 35)),
        ((100, 100), (100, 100)),
        ((100, 101), (100, 101)),
    ]
    if request.config.getoption("--quick"):
        result = result[:5]
    return result


@pytest.fixture
def banded_configs(request):
    """Configurations for the Banded linear operator.

    :returns: List of configs where each config is in the form
      ``[expected_shape, symmetric, (idx1, dim1), (idx2, dim2), ...]``.
    """
    result = [
        # just main diagonal
        [(10, 10), False, (0, 10)],
        [(10, 10), True, (0, 10)],
        # just subdiagonal
        [(13, 13), True, (2, 11)],
        [(11, 13), False, (2, 11)],
        [(8, 5), False, (-3, 5)],
        # tridiagonal
        [(5, 5), False, (0, 5), (1, 4), (-1, 4)],
        [(5, 5), True, (0, 5), (1, 4)],
        # rectangular fat and tall
        [(5, 15), False, (0, 5), (1, 5), (10, 5)],
        [(15, 5), False, (0, 5), (-1, 5), (-10, 5)],
        # mixed up, asymmetric
        [(10, 15), False, (0, 10), (5, 10), (-1, 9)],
        # mixed up, symmetric
        [(10, 10), True, (1, 9), (3, 7), (6, 4)],
    ]
    return result


# ##############################################################################
# # TESTS
# ##############################################################################
def test_sum_correctness(
    rng_seeds,
    torch_devices,
    dtypes,
    sum_shapes,
):
    """Test case for correctness of ``SumLinOp``.

    Creates a set of random matrices. Then sums them explicitly, and creates
    a ``SumLinOp``. Tests that:

    * Forward matmul with the linop yields same result as explicit
    * Adjoint matmul with the linop yields same result as explicit
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype in dtypes:
                for shapes in sum_shapes:
                    # sample random submatrices
                    submatrices = []
                    for i, shape in enumerate(shapes):
                        submatrices.append(
                            gaussian_noise(
                                shape, seed=seed + i, dtype=dtype, device=device
                            )
                        )
                    # add submatrices, either explicit or implicit
                    explicit = sum(submatrices)
                    lop = SumLinOp(
                        (f"M_{i}", m) for i, m in enumerate(submatrices)
                    )
                    mat1 = linop_to_matrix(lop, dtype, device, adjoint=False)
                    mat2 = linop_to_matrix(lop, dtype, device, adjoint=True)
                    #
                    assert (
                        explicit == mat1
                    ).all(), "Incorrect SumLinOp via forward matmul?"
                    assert (
                        explicit == mat2
                    ).all(), "Incorrect SumLinOp via adjoint matmul?"


def test_sum_shapes(sum_shapes):
    """Test case for shape consistency of ``SumLinOp``.

    When creating a ``SumLinOp``, tests that:

    * Providing an empty collection triggers a ValueError
    * Mismatching shapes triggers a BadShapeError
    * (adjoint-) matmul with vector of wrong size triggers BadShapeError
    """
    with pytest.raises(ValueError):
        lop = SumLinOp([])
    #
    for shapes in sum_shapes:
        # ignore cases with less than 2 matrices (no possible inconsistency)
        if len(shapes) < 2:
            continue
        # sample random submatrices of different sizes
        submatrices = []
        for i, shape in enumerate(shapes):
            bad_shape = (shape[0] + i, shape[1] + i)
            submatrices.append(torch.empty(bad_shape))
        # creating linop should trigger error
        with pytest.raises(BadShapeError):
            lop = SumLinOp((f"M_{i}", m) for i, m in enumerate(submatrices))
        # now create a linop with first element
        # and multiply with wrong-sized vectors:
        lop = SumLinOp((f"M_{i}", m) for i, m in enumerate(submatrices[:1]))
        v1, v2 = torch.zeros(lop.shape[0] + 1), torch.zeros(lop.shape[1] + 1)
        with pytest.raises(BadShapeError):
            _ = v1 @ lop
        with pytest.raises(BadShapeError):
            _ = lop @ v2


def test_banded_correctness(
    rng_seeds,
    torch_devices,
    dtypes,
    banded_configs,
):
    """Test case for correctness of ``BandedLinOp``.

    Creates a set of random subdiagonals, and a ``BandedLinOp`` from them.
    Then, explicitly creates a banded matrix using ``lop.to_matrix()``.
    Tests that:

    * Forward matmul with the linop yields same result as explicit
    * Adjoint matmul with the linop yields same result as explicit
    * Linop shape is same as expected
    * Linop.to_matrix shape is same as expected
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype in dtypes:
                for config in banded_configs:
                    expected_shape, sym, *diag_config = config
                    diags = {
                        idx: gaussian_noise(
                            dims, seed=seed + i, dtype=dtype, device=device
                        )
                        for i, (idx, dims) in enumerate(diag_config)
                    }
                    lop = BandedLinOp(diags, symmetric=sym)
                    #
                    explicit = lop.to_matrix()
                    mat1 = linop_to_matrix(lop, dtype, device, adjoint=False)
                    mat2 = linop_to_matrix(lop, dtype, device, adjoint=True)
                    #
                    assert (
                        explicit == mat1
                    ).all(), "Incorrect BandedLinOp via forward matmul?"
                    assert (
                        explicit == mat2
                    ).all(), "Incorrect BandedLinOp via adjoint matmul?"
                    #
                    assert (
                        expected_shape == lop.shape
                    ), "Incorrect BandedLinOp shape?"
                    assert (
                        expected_shape == explicit.shape
                    ), "Incorrect BandedLinOp.to_matrix shape?"


def test_banded_shapes(banded_configs):
    """Test case for shape consistency of ``BandedLinOp``.

    When creating a ``BandedLinOp``, tests that:

    * Providing an empty collection triggers a ValueError
    * Non-vector diagonals trigger BadShapeError
    * Inconsistent diagonals trigger BadShapeError
    * Negative indices in symmetric mode trigger BadShapeError
    * Non-square configurations in symmetric mode trigger BadShapeError

    Also:

    * (adjoint-) matmul with vector of wrong size triggers BadShapeError
    """
    # empty linop dict triggers error
    with pytest.raises(ValueError):
        lop = BandedLinOp({})
    # non-vector diagonals trigger error
    with pytest.raises(BadShapeError):
        lop = BandedLinOp({0: torch.zeros(5, 5)})
    # inconsistent diagonals trigger error
    with pytest.raises(BadShapeError):
        lop = BandedLinOp({0: torch.zeros(5), 1: torch.zeros(10)})
    # negative indices in symmetric mode trigger error
    with pytest.raises(BadShapeError):
        lop = BandedLinOp(
            {0: torch.zeros(5), -1: torch.zeros(4)}, symmetric=True
        )
    # non-square configurations in symmetric mode trigger error
    with pytest.raises(BadShapeError):
        lop = BandedLinOp(
            {0: torch.zeros(5), 1: torch.zeros(5)}, symmetric=True
        )
    #
    for config in banded_configs:
        expected_shape, sym, *diag_config = config
        diags = {
            idx: torch.empty(dims) for i, (idx, dims) in enumerate(diag_config)
        }
        # now create a linop with first element and multiply with wrong-sizes:
        lop = BandedLinOp(diags, symmetric=sym)
        v1, v2 = torch.zeros(lop.shape[0] + 1), torch.zeros(lop.shape[1] + 1)
        with pytest.raises(BadShapeError):
            _ = v1 @ lop
        with pytest.raises(BadShapeError):
            _ = lop @ v2
