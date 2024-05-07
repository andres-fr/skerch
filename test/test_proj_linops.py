#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for orthogonal projection linear operators."""


import pytest
import torch

from skerch.a_posteriori import a_posteriori_error
from skerch.linops import NegOrthProjLinOp, OrthProjLinOp
from skerch.utils import BadShapeError, gaussian_noise

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def heights_widths():
    """Shapes of matrices to QR decompose."""
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (1_000, 100),
    ]
    return result


@pytest.fixture
def tall_heights_widths():
    """Shapes of matrices with ``h > w``."""
    result = [
        (10, 1),
        (100, 10),
        (1000, 100),
    ]
    return result


@pytest.fixture
def dtypes_atols():
    """Datatypes and their absolute error tolerances."""
    result = {torch.float64: 1e-10, torch.float32: 1e-5}
    return result


@pytest.fixture
def num_test_measurements():
    """Number of a-posteriori measurements."""
    return 50


# ##############################################################################
# #
# ##############################################################################
def test_proj_correctness(
    rng_seeds,
    torch_devices,
    heights_widths,
    dtypes_atols,
    num_test_measurements,
):
    """Test case for orthogonal projectors.

    Draws random, thin orthonormal matrices, constructs the corresponding
    orthogonal and neg-orthogonal projectors, and tests that:

    * Each projector is symmetric
    * Each projector is idempotent
    * Both projectors are orthogonal to each other
    * Adding both projectors yields identity
    """
    for h, w in heights_widths:
        assert h >= w, "This test doesn't need/admit fat matrices!"
    #
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, atol in dtypes_atols.items():
                for h, w in heights_widths:
                    mat = gaussian_noise(
                        (h, w),
                        mean=0.0,
                        std=1.0,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    Q = torch.linalg.qr(mat)[0]
                    proj = OrthProjLinOp(Q)
                    negproj = NegOrthProjLinOp(Q)
                    #
                    meas_vectors = gaussian_noise(
                        (h, num_test_measurements),
                        mean=0.0,
                        std=1.0,
                        seed=seed + 123,
                        dtype=dtype,
                        device=device,
                    )
                    proj_meas = proj @ meas_vectors
                    negproj_meas = negproj @ meas_vectors
                    # test symmetry
                    assert torch.allclose(
                        proj_meas, (meas_vectors.T @ proj).T, atol=atol
                    ), "Projector is not symmetric!"
                    assert torch.allclose(
                        negproj_meas, (meas_vectors.T @ negproj).T, atol=atol
                    ), "Negative projector is not symmetric!"
                    # test idempotence
                    assert torch.allclose(
                        proj_meas, proj @ proj_meas, atol=atol
                    ), "Projector is not idempotent!"
                    assert torch.allclose(
                        negproj_meas, negproj @ negproj_meas, atol=atol
                    ), "Negative projector is not idempotent!"
                    # test orthogonality
                    assert (
                        abs(proj @ negproj_meas)
                    ).max() <= atol, (
                        "Projector and its negation are not orthogonal!"
                    )
                    # test identity
                    assert (
                        abs(proj_meas + negproj_meas - meas_vectors).max()
                        <= atol
                    ), "Projector and its negation are not identity!"


def test_proj_shapes(torch_devices, dtypes_atols, tall_heights_widths):
    """Test case for shape consistencies of projection linops.

    Check that
    * Bad ``Q`` shapes (fat) give bad shape error
    * Correct inputs can be both vectors and matrices
    """
    for device in torch_devices:
        for dtype in dtypes_atols.keys():
            for h, w in tall_heights_widths:
                """
                First try to create w,h and expect Badshapeerror

                then actually create, and feed a vec and a mat. should work.
                """
                # create mat. It doesn't need to be orthogonalized for this test
                mat = gaussian_noise(
                    (h, w),
                    mean=0.0,
                    std=1.0,
                    seed=12345,
                    dtype=dtype,
                    device=device,
                )
                # creating projectors with fat matrix should throw error
                with pytest.raises(BadShapeError):
                    OrthProjLinOp(mat.T)
                with pytest.raises(BadShapeError):
                    NegOrthProjLinOp(mat.T)
                # now create matrices and feed vectors and matrices on both
                # sides. No error should be thrown and shapes should match.
                test_vec = torch.ones(h, dtype=dtype, device=device)
                test_mat = torch.ones((h, 2), dtype=dtype, device=device)
                proj = OrthProjLinOp(mat)
                negproj = NegOrthProjLinOp(mat)
                #
                assert (proj @ test_vec).shape == (h,)
                assert (proj @ test_mat).shape == (h, 2)
                assert (test_vec.T @ proj).shape == (h,)
                assert (test_mat.T @ proj).shape == (2, h)
                #
                assert (negproj @ test_vec).shape == (h,)
                assert (negproj @ test_mat).shape == (h, 2)
                assert (test_vec.T @ negproj).shape == (h,)
                assert (test_mat.T @ negproj).shape == (2, h)
