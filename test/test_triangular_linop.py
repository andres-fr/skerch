#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched triangular estimation."""


import pytest
import torch

from skerch.triangles import serrated_hadamard_pattern, TriangularLinOp
from skerch.utils import gaussian_noise, BadShapeError
from skerch.synthmat import SynthMat
from . import rng_seeds, torch_devices  # noqa: F401


import matplotlib.pyplot as plt


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def serrated_hadamard_dims_chunks():
    """ """
    result = [(100, 2), (100, 3), (100, 10), (100, 100)]
    return result


@pytest.fixture
def hadamard_atol():
    """ """
    result = {torch.float64: 1e-13, torch.float32: 1e-5}
    return result


@pytest.fixture
def num_tri_tests(request):
    """ """
    result = 3
    if request.config.getoption("--quick"):
        result = 1
    return result


@pytest.fixture
def dim_rank_decay_sym_width_hutch_maindiag_rtol(request):
    """ """
    dims, rank = 1000, 100
    if request.config.getoption("--quick"):
        dims, rank = 500, 50
    result = [
        # COMMENT CASE
        # fast-decay:(dims, rank, 0.5, True, 20, 0, True, 0.01),
        (dims, rank, 0.5, False, 5, 400, True, 0.1),
        # (dims, rank, 0.5, False, 5, 400, True, 0.1),
    ]
    return result


# ##############################################################################
# # HADAMARD
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


# ##############################################################################
# # TRIANGULAR LINOP
# ##############################################################################
def test_triangular_linop_badshape():
    """Test case for non-square inputs to triangular linop."""
    mat = torch.ones(2, 3)
    with pytest.raises(BadShapeError):
        TriangularLinOp(
            mat,
            1,
            10,
            lower=True,
            with_main_diagonal=True,
        )


def test_triangular_linop_corner_cases(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_width_hutch_maindiag_rtol,
    num_tri_tests,
):
    """Test case for corner cases with triangular linop.

    * Empty and scalar matrices
    """
    for seed in rng_seeds:
        for dtype in (torch.float64, torch.float32):
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    step_width,
                    hutch_meas,
                    maindiag,
                    rtol,
                ) in dim_rank_decay_sym_width_hutch_maindiag_rtol:
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


def test_triangular_linop_correctness(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_width_hutch_maindiag_rtol,
    num_tri_tests,
):
    """ """
    for seed in rng_seeds:
        for dtype in (torch.float64, torch.float32):
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    step_width,
                    hutch_meas,
                    maindiag,
                    rtol,
                ) in dim_rank_decay_sym_width_hutch_maindiag_rtol:
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
                        diag_idx = int(not maindiag)  # if with main, idx=0
                        if lower:
                            tri = torch.tril(mat, diagonal=-diag_idx)
                        else:
                            tri = torch.triu(mat, diagonal=diag_idx)
                        #
                        tri_approx = TriangularLinOp(
                            mat,
                            step_width,
                            hutch_meas,
                            lower=lower,
                            with_main_diagonal=maindiag,
                        )
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
                                #
                                try:
                                    assert (
                                        rel_err <= rtol
                                    ), "Incorrect triangular recovery!"
                                except Exception as e:
                                    print(e)
                                    print(
                                        "\n\n!!!!",
                                        dim,
                                        rank,
                                        decay,
                                        sym,
                                        step_width,
                                        hutch_meas,
                                        maindiag,
                                        rtol,
                                    )
                                    print(f"lower={lower}")
                                    print(f"adjoint={adjoint}")
                                    print("relerr:", rel_err, "tol:", rtol)
                                    breakpoint()
                                    # plt.clf(); plt.plot(tri_v, color="black"); plt.plot(tri_approx_v); plt.show()
                                    # plt.clf(); plt.plot(tri_approx_v - tri_v); plt.show()

                        # print("\n\n\n", tri_approx._get_chunk_dims(10, 3))
                        # breakpoint()

                        # tri_approx._get_stair_width(10, 3)
                        # p list(tri_approx._iter_stairs(7, 3))
                        # breakpoint()
                        #

                        # plt.clf(); plt.imshow(tri); plt.show()
                        #
                        #
                        # rel_err = (dist / torch.norm(diag)).item()
                        # assert rel_err <= rtol, "Incorrect subdiag recovery!"
