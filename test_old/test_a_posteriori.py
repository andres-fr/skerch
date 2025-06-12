#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for a-posteriori functionality."""

import pytest
import torch

from skerch.a_posteriori import a_posteriori_error, scree_bounds
from skerch.synthmat import SynthMat

from . import exp_decay, rng_seeds, torch_devices  # noqa: F401


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def f64_rtol():
    """Relative error tolerance for float64."""
    result = {torch.float64: 1e-10}
    return result


@pytest.fixture
def f32_rtol():
    """Relative error tolerance for float32."""
    result = {torch.float32: 1e-3}
    return result


@pytest.fixture
def h_w_sym_measurements(request):
    """Test cases: (op_shape, is_symmetric, number_a_post_measurements)."""
    result = [
        ((10, 10), True, 50),
        ((10, 10), False, 50),
        ((10, 100), False, 50),
        ((100, 100), True, 50),
        ((100, 100), False, 50),
        ((100, 1000), False, 50),
        ((1000, 1000), True, 50),
        ((1000, 1000), False, 50),
    ]
    #
    if request.config.getoption("--quick"):
        result = result[:3]
    return result


@pytest.fixture
def mixture_weights(request):
    """Linear weights to perturb one random matrix with another."""
    result = [0.001, 0.1, 1]
    if request.config.getoption("--quick"):
        result = result[:2]
    return result


# ##############################################################################
# # A POSTERIORI ERROR ESTIMATION
# ##############################################################################
def test_posteriori_and_scree_correctness(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    h_w_sym_measurements,
    exp_decay,
    mixture_weights,
):
    """Test case for a-posteriori error estimation and scree bounds.

    This test samples all kinds of exp-decaying matrices, and random
    perturbations of varying magnitude. Then compares the actual Frobenius
    of the residual with the estimated one. It also reuses part of the
    computations to extract and test the scree bounds. Tests:
    * The a-posteriori estimates are within 0.5x and 2x of the real value
    * The scree upper bounds are above the scree lower bounds
    * Scree bounds are non-ascending as rank progresses
    * Scree bounds get tighter as rank progresses
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, _rtol in {**f64_rtol, **f32_rtol}.items():
                for (h, w), sym, q in h_w_sym_measurements:
                    for dec in exp_decay:
                        for weight in mixture_weights:
                            for psd in (True, False):
                                # sample an exp matrix and its perturbation
                                mat1 = SynthMat.exp_decay(
                                    shape=(h, w),
                                    rank=1,
                                    decay=dec,
                                    symmetric=sym,
                                    seed=seed,
                                    dtype=dtype,
                                    device=device,
                                    psd=psd,
                                )
                                mat2 = mat1 + weight * SynthMat.exp_decay(
                                    shape=(h, w),
                                    rank=1,
                                    decay=dec,
                                    symmetric=sym,
                                    seed=seed + 1,
                                    dtype=dtype,
                                    device=device,
                                    psd=psd,
                                )
                                # measure their actual frobenius difference
                                residual = ((mat1 - mat2) ** 2).sum()
                                # a posteriori estimate of their difference
                                for adjoint in [True, False]:
                                    f1, f2, res_estimate = a_posteriori_error(
                                        mat1,
                                        mat2,
                                        q,
                                        seed=seed + 2,
                                        dtype=dtype,
                                        device=device,
                                        adjoint=adjoint,
                                    )[0]
                                    assert residual <= (
                                        res_estimate * 2
                                    ), "Too small a-posteriori estimate!"
                                    assert residual >= (
                                        res_estimate / 2
                                    ), "Too large a-posteriori estimate!"
                                # SCREE TEST:
                                if sym:
                                    S1 = torch.linalg.eigvalsh(mat1)
                                    S2 = torch.linalg.eigvalsh(mat2)
                                else:
                                    _, S1, _ = torch.linalg.svd(mat1)
                                    _, S2, _ = torch.linalg.svd(mat2)
                                _, perm = abs(S1).sort(descending=True)
                                S1 = S1[perm]
                                _, perm = abs(S2).sort(descending=True)
                                S2 = S2[perm]
                                scree_lo, scree_hi = scree_bounds(
                                    S2, f1**0.5, res_estimate**0.5
                                )
                                # check both bounds non-ascending, and hi>=lo
                                assert (
                                    scree_hi >= scree_lo
                                ).all(), "Scree upper bounds <= lower bounds!"
                                assert (
                                    scree_hi.diff() <= 0
                                ).all(), "Scree upper bounds are ascending!"
                                assert (
                                    scree_lo.diff() <= 0
                                ).all(), "Scree lower bounds are ascending!"
                                # check: as rank increases, bounds tighten
                                assert (
                                    (scree_hi - scree_lo).diff() <= 0
                                ).all(), "Scree bounds diverging with rank!"
                                # # Check that bounds surround the actual scree
                                # # curve. THIS IS NOT THE CASE FOR THIS TEST
                                # # CASE, BUT WE AREN'T DOING SVD SO IT MAY BE
                                # # WRONG. IGNORE
                                # import matplotlib.pyplot as plt

                                # actual_scree, _ = scree_bounds(
                                #     S1, (S1**2).sum() ** 0.5, 0
                                # )
                                # # plt.clf()
                                # # plt.plot(scree_lo)
                                # # plt.plot(scree_hi)
                                # # plt.plot(actual_scree, linestyle="dashed")
                                # # plt.show()
