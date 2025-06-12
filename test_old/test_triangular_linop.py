#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched triangular estimation."""


import matplotlib.pyplot as plt
import pytest
import torch

from skerch.synthmat import SynthMat
from skerch.triangles import TriangularLinOp, serrated_hadamard_pattern
from skerch.utils import BadShapeError, gaussian_noise

from . import rng_seeds, torch_devices  # noqa: F401


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def serrated_hadamard_dims_chunks():
    """Pairs of ``(matrix_order, step_size)`` to test serrated Hadamard."""
    result = [(100, 2), (100, 3), (100, 10), (100, 100)]
    return result


@pytest.fixture
def hadamard_atol():
    """Absolute tolerances to test serrated Hadamard."""
    result = {torch.float64: 1e-13, torch.float32: 1e-5}
    return result


@pytest.fixture
def num_tri_tests(request):
    """Number of random vectors to test for each triangular linop case."""
    result = 3
    if request.config.getoption("--quick"):
        result = 1
    return result


@pytest.fixture
def dim_rank_decay_sym_width_hutch_maindiag_rtol(request):
    """Test cases for triangular estimation.

    Entries are in the form
    ``(dim, rank, decay, sym, width, hutch, maindiag, rtol)``, where:

    * ``dim`` is the number of dimensions of a square test matrix of
      exp-decaying singular values
    * ``rank`` is the number of unit singular values in the test matrix
    * ``decay`` is the speed of the exp-decay of singular values after ``rank``.
      The larger, the faster decay.
    * ``sym`` is a boolean specifying whether the matrix is symmetric
    * ``width`` is the step width for the staircase exact measurements in the
      triangular linop.
    * ``hutch`` is the number of Hutchinson measurements for the "serrated"
      (block-diagonal) estimator in the triangular linop.
    * ``maindiag`` is a boolean specifying whether to include the main diagonal
      in the triangle or not.
    * ``rtol`` is the relative error tolerance for each given case.
    """
    dims, rank = 1000, 100
    if request.config.getoption("--quick"):
        dims, rank = 500, 50
    result = [
        # [fast decay, asym]: just step measurements do decently
        (dims, rank, 0.5, False, 5, 0, True, 0.15),
        # ... but adding Hutch helps
        (dims, rank, 0.5, False, 5, 300, True, 0.1),
        # [fast decay, sym]: stronger diag, just steps is much worse now
        (dims, rank, 0.5, True, 5, 0, True, 0.5),
        # Hutch helps a lot, but it is recommended to deflate diagonal.
        (dims, rank, 0.5, True, 5, 300, True, 0.25),
        # Not including the diagonal shows that this was indeed the culprit
        (dims, rank, 0.5, True, 5, 100, False, 0.1),
        # [slow decay]: behave same as slow decay
        (dims, rank, 0.01, False, 5, 0, True, 0.15),
        (dims, rank, 0.01, True, 5, 100, False, 0.1),
    ]
    return result


# ##############################################################################
# # TRIANGULAR LINOP
# ##############################################################################
def test_triangular_linop_correctness(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_width_hutch_maindiag_rtol,
    num_tri_tests,
):
    """Test case for approximated triangulars on larger matrices.

    See :func:`dim_rank_decay_sym_width_hutch_maindiag_rtol` docstring and
    implementation for details on the use cases.
    """
    for seed in rng_seeds:
        for dtype in (torch.float32,):  # speed up test by just running low res
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    stair_width,
                    hutch_meas,
                    maindiag,
                    rtol,
                ) in dim_rank_decay_sym_width_hutch_maindiag_rtol:
                    # create a linop for this seed/dtype/device and test case
                    # below, we will loop over lower/upper, left/right matmul
                    mat = SynthMat.exp_decay(
                        shape=(dim, dim),
                        rank=rank,
                        decay=decay,
                        symmetric=sym,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                        psd=False,
                    )
                    for lower in (True, False):
                        # create exact triangular and its approximation
                        diag_idx = int(not maindiag)  # if with main, idx=0
                        if lower:
                            tri = torch.tril(mat, diagonal=-diag_idx)
                        else:
                            tri = torch.triu(mat, diagonal=diag_idx)
                        tri_approx = TriangularLinOp(
                            mat,
                            stair_width,
                            hutch_meas,
                            lower=lower,
                            with_main_diagonal=maindiag,
                        )
                        # do the matmul, and check that error is within rtol
                        for i in range(num_tri_tests):
                            v = gaussian_noise(
                                dim,
                                seed=seed + i + 1,
                                dtype=dtype,
                                device=device,
                            )
                            for adjoint in (True, False):
                                if adjoint:
                                    tri_v = v @ tri
                                    tri_approx_v = v @ tri_approx
                                else:
                                    tri_v = tri @ v
                                    tri_approx_v = tri_approx @ v
                                #
                                dist = torch.dist(tri_v, tri_approx_v)
                                rel_err = (dist / torch.norm(tri_v)).item()
                                assert (
                                    rel_err <= rtol
                                ), "Incorrect triangular recovery!"


def test_triangular_linop_badshape():
    """Test case for non-square or empty inputs to triangular linop."""
    # non-square case
    mat = torch.ones(2, 3)
    with pytest.raises(BadShapeError):
        TriangularLinOp(
            mat,
            1,
            0,
            lower=True,
            with_main_diagonal=True,
        )
    # empty case
    mat = torch.ones(0, 0)
    with pytest.raises(BadShapeError):
        TriangularLinOp(
            mat,
            0,
            0,
            lower=True,
            with_main_diagonal=True,
        )


def test_triangular_linop_small_shapes():
    """Exact test cases for matrices of order 1, 2, 3 (both step and Hutch)."""
    # create test linops and vectors
    mat1 = torch.tensor([[1]], dtype=torch.float64)
    mat2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    mat3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64)
    v1 = torch.ones_like(mat1[0])
    v2 = torch.ones_like(mat2[0])
    v3 = torch.ones_like(mat3[0])

    # 1x1 lower with diagonal
    tri = TriangularLinOp(mat1, 1, 1, lower=True, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((1.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((1.0,))
    ), "Incorrect result!"
    # 1x1 lower without diagonal (Hutch)
    tri = TriangularLinOp(mat1, 1, 1, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    # 1x1 lower without diagonal (Step)
    tri = TriangularLinOp(mat1, 1, 0, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"

    # 1x1 upper with diagonal
    tri = TriangularLinOp(mat1, 1, 1, lower=False, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((1.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((1.0,))
    ), "Incorrect result!"
    # 1x1 upper without diagonal (Hutch)
    tri = TriangularLinOp(mat1, 1, 1, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    # 1x1 upper without diagonal (Step)
    tri = TriangularLinOp(mat1, 1, 0, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v1, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"
    assert torch.allclose(
        v1 @ tri, torch.DoubleTensor((0.0,))
    ), "Incorrect result!"

    # 2x2 lower with diagonal
    tri = TriangularLinOp(mat2, 2, 2, lower=True, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((1.0, 7))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((4.0, 4))
    ), "Incorrect result!"
    # 2x2 lower without diagonal (Hutch)
    tri = TriangularLinOp(mat2, 2, 2, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((0.0, 3))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((3.0, 0))
    ), "Incorrect result!"
    # 2x2 lower without diagonal (Step)
    tri = TriangularLinOp(mat2, 1, 0, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((0.0, 3))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((3.0, 0))
    ), "Incorrect result!"

    # 2x2 upper with diagonal
    tri = TriangularLinOp(mat2, 2, 2, lower=False, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((3.0, 4))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((1.0, 6))
    ), "Incorrect result!"
    # 2x2 upper without diagonal (Hutch)
    tri = TriangularLinOp(mat2, 2, 2, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((2.0, 0))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((0.0, 2))
    ), "Incorrect result!"
    # 2x2 upper without diagonal (Step)
    tri = TriangularLinOp(mat2, 1, 0, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v2, torch.DoubleTensor((2.0, 0))
    ), "Incorrect result!"
    assert torch.allclose(
        v2 @ tri, torch.DoubleTensor((0.0, 2))
    ), "Incorrect result!"

    # 3x3 lower with diagonal
    tri = TriangularLinOp(mat3, 3, 3, lower=True, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((1.0, 9, 24))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((12.0, 13, 9))
    ), "Incorrect result!"
    # 3x3 lower without diagonal (Hutch)
    tri = TriangularLinOp(mat3, 3, 3, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((0.0, 4, 15))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((11.0, 8, 0))
    ), "Incorrect result!"
    # 3x3 lower without diagonal (Step)
    tri = TriangularLinOp(mat3, 1, 0, lower=True, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((0.0, 4, 15))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((11.0, 8, 0))
    ), "Incorrect result!"

    # 3x3 upper with diagonal
    tri = TriangularLinOp(mat3, 3, 3, lower=False, with_main_diagonal=True)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((6.0, 11, 9))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((1.0, 7, 18))
    ), "Incorrect result!"
    # 3x3 upper without diagonal (Hutch)
    tri = TriangularLinOp(mat3, 3, 3, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((5.0, 6, 0))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((0.0, 2, 9))
    ), "Incorrect result!"
    # 3x3 upper without diagonal (Step)
    tri = TriangularLinOp(mat3, 1, 0, lower=False, with_main_diagonal=False)
    assert torch.allclose(
        tri @ v3, torch.DoubleTensor((5.0, 6, 0))
    ), "Incorrect result!"
    assert torch.allclose(
        v3 @ tri, torch.DoubleTensor((0.0, 2, 9))
    ), "Incorrect result!"


def test_triangular_linop_corner_cases(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_width_hutch_maindiag_rtol,
    num_tri_tests,
):
    """Testing corner cases for triangular linop.

    Cases:

    * Setting step width to 0 raises an error
    * Performing 0 measurements is accepted, but yields zeros
    * Requesting too many measurements raises an error
    """
    order = 10
    mat = torch.randn((order, order), dtype=torch.float64)
    v = torch.ones_like(mat[0])
    z = torch.zeros_like(mat[0])
    for lower in (True, False):
        for diag in (True, False):
            # zero step width raises error
            with pytest.raises(AssertionError):
                TriangularLinOp(
                    mat,
                    0,
                    0,
                    lower=lower,
                    with_main_diagonal=diag,
                )
            # 9 stair measurements + 2 Hutch = 11 measurements. Error!
            with pytest.raises(AssertionError):
                TriangularLinOp(
                    mat,
                    1,
                    2,
                    lower=lower,
                    with_main_diagonal=diag,
                )
            # 0 stair measurements + 11 Hutch, also error
            with pytest.raises(AssertionError):
                TriangularLinOp(
                    mat,
                    order,
                    order + 1,
                    lower=lower,
                    with_main_diagonal=diag,
                )
            # no measurements yields zero operator
            tri = TriangularLinOp(
                mat,
                order,
                0,
                lower=lower,
                with_main_diagonal=diag,
            )
            assert torch.allclose(tri @ v, z), "Result should be zeros!"
            assert torch.allclose(v @ tri, z), "Result should be zeros!"


# ##############################################################################
# # SERRATED HADAMARD
# ##############################################################################
def test_serrated_hadamard_pattern(
    rng_seeds,  # noqa: F811
    hadamard_atol,
    torch_devices,  # noqa: F811
    serrated_hadamard_dims_chunks,
):
    """Test case for serrated Hadamard pattern generator.

    This particular Hadamard pattern is useful to estimate block-triangulars
    using Hutchinson.

    This test case checks that serrated patterns of vectors of ones match
    expected values for:
    * Lower and upper-triangular patterns
    * FFT and plain implementations
    * With and without diagonal
    * Different block sizes

    Then, it checks that FFT and plain implementations yield same results for
    random vectors.
    """
    for seed in rng_seeds:
        for dtype, atol in hadamard_atol.items():
            for device in torch_devices:
                # Simple tests with vector of ones:
                v = torch.ones(10, dtype=dtype, device=device)
                # chunk size must be 1 or greater
                with pytest.raises(ValueError):
                    serrated_hadamard_pattern(v, 0)
                #
                had = serrated_hadamard_pattern(v, 1, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(v, 2, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(v, 4, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 1, with_main_diagonal=False, use_fft=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 2, with_main_diagonal=False, use_fft=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 4, with_main_diagonal=False, use_fft=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 1, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 2, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"

                had = serrated_hadamard_pattern(
                    v, 4, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [2, 1, 4, 3, 2, 1, 4, 3, 2, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 1, with_main_diagonal=False, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 2, with_main_diagonal=False, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(
                    v, 4, with_main_diagonal=False, use_fft=False, lower=False
                )
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                # random test with more complex responses
                for dims, chunk in serrated_hadamard_dims_chunks:
                    v = gaussian_noise(
                        dims,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    for diag in (True, False):
                        for lower in (True, False):
                            had = serrated_hadamard_pattern(
                                v,
                                chunk,
                                with_main_diagonal=diag,
                                use_fft=False,
                                lower=lower,
                            )
                            had_fft = serrated_hadamard_pattern(
                                v,
                                chunk,
                                with_main_diagonal=diag,
                                use_fft=True,
                                lower=lower,
                            )
                            assert torch.allclose(
                                had, had_fft, atol=atol
                            ), "Inconsistent FFT implementation of serrated!"
