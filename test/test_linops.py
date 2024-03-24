#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for linop (linear operator) utils."""


import pytest
import torch

from skerch.a_posteriori import a_posteriori_error
from skerch.linops import CompositeLinOp, DiagonalLinOp
from skerch.utils import BadShapeError, gaussian_noise

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def f64_atol():
    """Absolute error tolerance for float64."""
    result = {torch.float64: 1e-10}
    return result


@pytest.fixture
def f32_atol():
    """Absolute error tolerance for float32."""
    result = {torch.float32: 1e-5}
    return result


@pytest.fixture
def composite_sizes(request):
    """Chained shapes for the composite linear operator."""
    result = [
        (1, 1, 1),
        (1, 10, 1),
        (10, 1, 10),
        (1, 10, 100, 10, 1),
        (10, 1, 100, 1, 10),
        (1, 100, 1000, 1, 10),
        (100, 100, 100, 100, 100),
        (1000, 1000, 1000),
        (1000, 10, 1000),
        (10, 1000, 1000, 10),
    ]
    if request.config.getoption("--quick"):
        result = result[:5]
    return result


@pytest.fixture
def diagonal_sizes(request):
    """Dimensions for the diagonal linear operator."""
    result = [1, 3, 10, 30, 100, 300, 1000]
    if request.config.getoption("--quick"):
        result = result[:5]
    return result


@pytest.fixture
def num_test_measurements():
    """Number of a-posteriori measurements.

    Calling a_posteriori_error_bounds(100, 1) yields
    {'P(err<=0x)': 0.0, 'P(err>=2x)': 2.171579274145306e-07}
    So if we obtain a small number, chances that it is not actually small are
    extremely low.
    """
    return 100


# ##############################################################################
# # TESTS
# ##############################################################################
def test_composite_correctness(
    rng_seeds,
    torch_devices,
    f64_atol,
    f32_atol,
    composite_sizes,
    num_test_measurements,
):
    """Test case for correctness of ``CompositeLinOp``.

    Creates a chain of random submatrices, and then composes them explicity
    using ``CompositeLinOp``. Tests that:

    * Composed shapes are as expected
    * Explicit and implicit compositions are identical both with left and right
      matrix multiplication
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, atol in {**f32_atol, **f64_atol}.items():
                for sizes in composite_sizes:
                    # sample random submatrices
                    submatrices = []
                    for i, (h, w) in enumerate(zip(sizes[:-1], sizes[1:])):
                        mat = gaussian_noise(
                            (h, w), seed=seed + i, dtype=dtype, device=device
                        )
                        submatrices.append((f"{(h, w)}", mat))
                    # compose submatrices, either explicit or implicit
                    explicit = submatrices[0][1]
                    for _, mat in submatrices[1:]:
                        explicit = explicit @ mat
                    composite = CompositeLinOp(submatrices)
                    # check that shapes are correct
                    expected_shape = (sizes[0], sizes[-1])
                    assert (
                        composite.shape == expected_shape
                    ), "Incorrect composite shape?"
                    assert (
                        explicit.shape == expected_shape
                    ), "Incorrect explicit shape?"
                    # check that explicit and implicit are identical for both
                    # left and right matmul
                    for adjoint in [True, False]:
                        (_, _, err_estimate), _ = a_posteriori_error(
                            explicit,
                            composite,
                            num_test_measurements,
                            seed=seed + len(sizes),
                            dtype=dtype,
                            device=device,
                            adjoint=adjoint,
                        )
                        assert (
                            err_estimate / explicit.numel()
                        ) <= atol, "Incorrect composite operation!"


def test_composite_shapes(torch_devices, f32_atol, composite_sizes):
    """Test case for shape consistency of ``CompositeLinOp``.

    Checks that:
    * Mismatching matrix shapes give bad shape error
    * Bad input shapes (left and right) give bad shape error
    * Correct inputs can be both vectors and matrices
    """
    for device in torch_devices:
        for dtype, _atol in f32_atol.items():
            for sizes in composite_sizes:
                # create composite matrix with mismatching shapes, should
                # throw an error
                submatrices = []
                for i, (h, w) in enumerate(zip(sizes[:-1], sizes[1:])):
                    mat = gaussian_noise(
                        (h, w + 1), seed=i, dtype=dtype, device=device
                    )
                    submatrices.append((f"{(h, w)}", mat))
                with pytest.raises(BadShapeError):
                    implicit = CompositeLinOp(submatrices)
                # now create composite matrix with correct shapes, but feed
                # vectors with incorrect shapes. Should also throw an error
                submatrices = []
                for i, (h, w) in enumerate(zip(sizes[:-1], sizes[1:])):
                    mat = gaussian_noise(
                        (h, w), seed=i, dtype=dtype, device=device
                    )
                    submatrices.append((f"{(h, w)}", mat))
                implicit = CompositeLinOp(submatrices)
                comp_h, comp_w = implicit.shape
                #
                x = gaussian_noise(
                    comp_h + 1,
                    seed=len(sizes) + 1,
                    dtype=dtype,
                    device=device,
                )
                with pytest.raises(RuntimeError):
                    x @ implicit
                #
                x = gaussian_noise(
                    comp_w + 1,
                    seed=len(sizes) + 1,
                    dtype=dtype,
                    device=device,
                )
                with pytest.raises(RuntimeError):
                    implicit @ x
                # correct matmul shape against vectors
                x = gaussian_noise(comp_w, seed=0, dtype=dtype, device=device)
                assert (implicit @ x).shape == (
                    comp_h,
                ), "Unexpected right matmul shape!"
                x = gaussian_noise(comp_h, seed=0, dtype=dtype, device=device)
                assert (x @ implicit).shape == (
                    comp_w,
                ), "Unexpected left matmul shape!"
                # correct matmul shape against matrices
                x = gaussian_noise(
                    (comp_w, 2), seed=0, dtype=dtype, device=device
                )
                assert (implicit @ x).shape == (
                    comp_h,
                    2,
                ), "Unexpected right matmul shape!"

                x = gaussian_noise(
                    (2, comp_h), seed=0, dtype=dtype, device=device
                )
                assert (x @ implicit).shape == (
                    2,
                    comp_w,
                ), "Unexpected left matmul shape!"


def test_diagonal_correctness(
    rng_seeds,
    torch_devices,
    f64_atol,
    f32_atol,
    diagonal_sizes,
    num_test_measurements,
):
    """Test case for correctness of ``DiagonalLinOp``.

    Create a random vector, and build a diagonal matrix either explicitly or
    using ``DiagonalLinOp``. Test that:

    * Shapes are as expected
    * Explicit and implicit matrices are identical both with left and right
      matrix multiplication
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, atol in {**f32_atol, **f64_atol}.items():
                for size in diagonal_sizes:
                    # sample random diagonal
                    diag = gaussian_noise(
                        size, seed=seed, dtype=dtype, device=device
                    )
                    explicit = torch.diag(diag)
                    implicit = DiagonalLinOp(diag)
                    # check that shapes are correct
                    expected_shape = (size, size)
                    assert (
                        implicit.shape == expected_shape
                    ), "Incorrect implicit shape?"
                    assert (
                        explicit.shape == expected_shape
                    ), "Incorrect explicit shape?"
                    # check that explicit and implicit are identical for both
                    # left and right matmul
                    for adjoint in [True, False]:
                        (_, _, err_estimate), _ = a_posteriori_error(
                            explicit,
                            implicit,
                            num_test_measurements,
                            seed=seed + 1,
                            dtype=dtype,
                            device=device,
                            adjoint=adjoint,
                        )
                        assert (
                            err_estimate / explicit.numel()
                        ) <= atol, "Incorrect composite operation!"


def test_diagonal_shapes(torch_devices, f32_atol, diagonal_sizes):
    """Test case for shape consistencies of ``DiagonalLinOp``.

    Check that
    * Bad input shapes (left and right) give bad shape error
    * Correct inputs can be both vectors and matrices
    """
    for device in torch_devices:
        for dtype, _atol in f32_atol.items():
            for size in diagonal_sizes:
                # sample random diagonal
                diag = gaussian_noise(size, seed=1, dtype=dtype, device=device)
                implicit = DiagonalLinOp(diag)
                x = gaussian_noise(size + 1, seed=2, dtype=dtype, device=device)
                # if the commands below throw a BadShapeError, the test passes
                with pytest.raises(BadShapeError):
                    implicit @ x
                with pytest.raises(BadShapeError):
                    x @ implicit
                # correct matmul shape against vectors
                x = gaussian_noise(size, seed=2, dtype=dtype, device=device)
                assert (implicit @ x).shape == (
                    size,
                ), "Unexpected right matmul shape!"
                assert (x @ implicit).shape == (
                    size,
                ), "Unexpected left matmul shape!"
                # correct matmul shape against matrices
                x = gaussian_noise(
                    (size, 2), seed=2, dtype=dtype, device=device
                )
                assert (implicit @ x).shape == (
                    size,
                    2,
                ), "Unexpected right matmul shape!"
                assert (x.T @ implicit).shape == (
                    2,
                    size,
                ), "Unexpected left matmul shape!"
