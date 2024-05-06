#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched triangular estimation."""


import pytest
import torch

from skerch.triangles import serrated_hadamard_pattern, TriangularLinOp
from skerch.utils import gaussian_noise
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
    result = 10
    if request.config.getoption("--quick"):
        result = 3
    return result


@pytest.fixture
def dim_rank_decay_sym_step_serr_rtol(request):
    """


    WITH MAIN DIAG BELONGS WITH TOLERANCE AND NUM MEAS HERE
    """
    dims, rank = 1000, 100
    if request.config.getoption("--quick"):
        dims, rank = 500, 50
    result = [
        # fast-decay: just Hutch does poorly, but better if symmetric
        (dims, rank, 0.5, True, [0], round(dims * 0.995), 0, 0.01),
        # (dims, rank, 0.5, False, [0], round(dims * 0.995), 0, 0.1),
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
def test_triangular_linop(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_step_serr_rtol,
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
                    step_meas,
                    serrat_meas,
                    rtol,
                ) in dim_rank_decay_sym_step_serr_rtol:
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
                        if lower:
                            tri = torch.tril(mat, diagonal=-int(with_main_diag))
                        else:
                            tri = torch.triu(mat, diagonal=int(with_main_diag))
                        #
                        tri_approx = TriangularLinOp(
                            mat,
                            step_meas,
                            serrat_meas,
                            lower=lower,
                            with_main_diagonal=with_main_diag,
                        )
                        for i in range(num_tri_tests):
                            v = gaussian_noise(
                                dim,
                                seed=seed + i + 1,
                                dtype=dtype,
                                device=device,
                            )
                            for adjoint in (False, True):
                                if not adjoint:
                                    tri_v = v @ tri
                                    tri_approx_v = v @ tri_approx
                                else:
                                    tri_v = tri @ v
                                    tri_approx_v = tri_approx @ v
                                dist = torch.dist(tri_v, tri_approx_v)

                                print("\n\n!!!!", dist)
                                breakpoint()
                                #
                                # plt.clf(); plt.plot(ww, color="black"); plt.plot(www); plt.show()
                                # plt.clf(); plt.imshow(tril); plt.show()
                                #
                                #
                                # rel_err = (dist / torch.norm(diag)).item()
                                # assert rel_err <= rtol, "Incorrect subdiag recovery!"
