#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for sum and banded linops."""


import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    TransposedLinOp,
    DiagonalLinOp,
    BandedLinOp,
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
def diagonal_sizes(request):
    """Dimensions for the diagonal linear operator."""
    result = [1, 3, 10, 30, 100]
    if request.config.getoption("--quick"):
        result = result[:4]
    return result


@pytest.fixture
def banded_configs():
    """Configurations for the Banded linear operator.

    :returns: List of correct configs where each config is in the form
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
# # DIAGONAL TESTS
# ##############################################################################
def test_diagonal_formal():
    """Formal test case for diagonal linops.

    Tests that:
    * Provided diagonal must be a vector, otherwise BadShapeError
    * Repr creates correct strings
    """
    # scalar input
    with pytest.raises(BadShapeError):
        _ = DiagonalLinOp(torch.tensor(1))
    # empty vector
    with pytest.raises(BadShapeError):
        _ = DiagonalLinOp(torch.ones(0))
    # matrix
    with pytest.raises(BadShapeError):
        _ = DiagonalLinOp(torch.ones(5, 5))
    # repr
    lop = DiagonalLinOp(torch.ones(2))
    assert (
        str(lop) == "<DiagonalLinOp(2x2)[1.0, 1.0]>"
    ), "Unexpected repr for diagonal linop!"


def test_diagonal_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    diagonal_sizes,
):
    """Test case for correctness of ``DiagonalLinOp``.

    For all seeds, devices dtypes and sizes, creates a diagonal linop and
    tests that:
    * the ``to_matrix`` method is correct
    * operationally it equals the explicit matrix
    * its transpose is correct
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for size in diagonal_sizes:
                    # sample random diagonal
                    diag = gaussian_noise(
                        size, seed=seed, dtype=dtype, device=device
                    )
                    mat = torch.diag(diag)
                    lop = DiagonalLinOp(diag)
                    # to_matrix correctness
                    assert (
                        lop.to_matrix() == mat
                    ).all(), "Incorrect to_matrix()!"
                    # fwd and adj matmul correctness
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=True)
                    assert (mat == lopmat).all(), "Incorrect (fwd) diagonal?"
                    lopmat = linop_to_matrix(lop, dtype, device, adjoint=False)
                    assert (mat == lopmat).all(), "Incorrect (adj) diagonal?"
                    # transpose
                    lopT = TransposedLinOp(lop)
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=False
                    )
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect transposition! (fwd)"
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=True
                    )
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect transposition! (adj)"


# ##############################################################################
# # BANDED TESTS
# ##############################################################################
def test_banded_formal():
    """Formal test case for banded linops.

    Tests that:
    * Repr creates correct strings
    * ValueError for empty input
    * BadShapeError for nonvector inputs
    * BadShapeError for negative diag idxs in symmetric linops
    * BadShapeError for inconsistent length/idx inputs
    * BadShapeError for nonsquare input if symmetric is True
    """
    # repr
    diags = {0: torch.zeros(5)}
    lop = BandedLinOp(diags, symmetric=False)
    assert (
        str(lop) == "<BandedLinOp(5x5)[0, sym=False]>"
    ), "Unexpected repr for diagonal linop!"
    # empty input
    diags = {}
    with pytest.raises(ValueError):
        _ = BandedLinOp(diags, symmetric=False)
    with pytest.raises(ValueError):
        _ = BandedLinOp(diags, symmetric=True)
    # empty input
    diags = {}
    with pytest.raises(ValueError):
        _ = BandedLinOp(diags, symmetric=False)
    with pytest.raises(ValueError):
        _ = BandedLinOp(diags, symmetric=True)
    # nonvector inputs
    diags = {0: torch.zeros(5, 2)}
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=False)
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=True)
    # negative idxs in symmetric
    diags = {-1: torch.zeros(5)}
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=True)
    # inconsistent lengths/idxs
    diags = {0: torch.zeros(5), 1: torch.zeros(6)}
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=False)
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=True)
    # nonsquare can't be symmetric
    diags = {0: torch.zeros(5), 1: torch.zeros(5)}
    with pytest.raises(BadShapeError):
        _ = BandedLinOp(diags, symmetric=True)


def test_banded_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    banded_configs,
):
    """Test case for correctness of ``BandedLinOp``.

    For all seeds, devices dtypes and sizes, creates a diagonal linop and
    tests that:
    * operationally it equals the explicit matrix via the ``to_matrix`` method
    * its transpose is correct
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for config in banded_configs:
                    expected_shape, sym, *diag_config = config
                    diags = {
                        idx: gaussian_noise(
                            dims, seed=seed + i, dtype=dtype, device=device
                        )
                        for i, (idx, dims) in enumerate(diag_config)
                    }
                    lop = BandedLinOp(diags, symmetric=sym)
                    # fwd and adj matmul correctness
                    lopmat = lop.to_matrix()
                    mat = linop_to_matrix(lop, dtype, device, adjoint=False)
                    assert torch.allclose(
                        lopmat, mat, atol=tol
                    ), "Incorrect (fwd) banded?"
                    mat = linop_to_matrix(lop, dtype, device, adjoint=True)
                    assert torch.allclose(
                        lopmat, mat, atol=tol
                    ), "Incorrect (adj) banded?"
                    # transpose
                    lopT = TransposedLinOp(lop)
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=False
                    )
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect transposition! (fwd)"
                    lopmatT = linop_to_matrix(
                        lopT, dtype, device, adjoint=True
                    )
                    assert torch.allclose(
                        mat.H, lopmatT, atol=tol
                    ), "Incorrect transposition! (adj)"
