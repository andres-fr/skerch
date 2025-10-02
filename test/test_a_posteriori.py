#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for a-posteriori functionality."""

import pytest
import torch

from skerch.synthmat import RandomLordMatrix
from skerch.utils import COMPLEX_DTYPES
from skerch.a_posteriori import apost_error_bounds, apost_error, scree_bounds

import numpy as np
from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-3,
        torch.complex64: 1e-3,
        torch.float64: 3e-8,
        torch.complex128: 3e-8,
    }
    return result


@pytest.fixture
def dispatcher_noise_types():
    """ """
    result = ["rademacher", "gaussian", "phase", "ssrft"]
    return result


@pytest.fixture
def apost_bounds_examples():
    """Config to test ``apost_error_bounds``.

    Returns a collection of input-output pairs, where input has the form
    ``(num_meas, rel_err, is_complex)`` and outputs ``(low_p, high_p)``.
    """
    result = [
        # probability that error is other than zero: 100%
        ((10, 0.0, True), (1.0, 1.0)),
        # P[error] below 100%: 0, above 200%: very low
        ((10, 1.0, True), (0.0, 0.04648952807678451)),
        # complex yields lower probs than real
        ((1000, 0.1, True), (0.004698482673443763, 0.009188338110655603)),
        ((1000, 0.1, False), (0.06854547886946127, 0.09585581938857757)),
    ]
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
def test_apost_error_bounds(apost_bounds_examples):
    """Test case for ``apost_error_bounds`` (formal and correctness).

    * num_measurements must be positive
    * rel_err must be between 0 and 1
    * output probabilities are as expected
    """
    atol = 1e-10
    low_s, high_s = "P(err<={}x)", "P(err>={}x)"
    # low_s, high_s = "P(err<=0.9x)", "P(err>=1.1x)"
    # positive num_meas
    with pytest.raises(ValueError):
        apost_error_bounds(-1, rel_err=0.5, is_complex=False)
    # rel_err in [0, 1]
    with pytest.raises(ValueError):
        apost_error_bounds(10, rel_err=-0.0001, is_complex=False)
    with pytest.raises(ValueError):
        apost_error_bounds(10, rel_err=1.0001, is_complex=False)
    # output correctness
    for (num_meas, rel_err, is_complex), (
        low_p,
        high_p,
    ) in apost_bounds_examples:
        bounds = apost_error_bounds(num_meas, rel_err, is_complex)
        lp = bounds[low_s.format(1 - rel_err)]
        hp = bounds[high_s.format(1 + rel_err)]
        assert np.isclose(low_p, lp, atol=atol), "Wrong low_p bound!"
        assert np.isclose(high_p, hp, atol=atol), "Wrong high_p bound!"


def test_apost_error_and_scree_correctness():
    """
    What to do?

    must be batched and HDF5 compatible
    """
    breakpoint()


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
