#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for a-posteriori functionality."""

import pytest
import torch
import numpy as np

from skerch.linops import CompositeLinOp
from skerch.synthmat import RandomLordMatrix
from skerch.utils import COMPLEX_DTYPES, gaussian_noise, truncate_decomp
from skerch.algorithms import ssvd
from skerch.a_posteriori import apost_error_bounds, apost_error, scree_bounds

from . import rng_seeds, torch_devices
from . import BasicMatrixLinOp, relerr


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
def apost_noise_types():
    """Collection of tuples ``(noise_type, is_complex_only)``"""
    result = [
        ("rademacher", False),
        ("gaussian", False),
        ("ssrft", False),
        ("phase", True),
    ]
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
def apost_config(request):
    """Config to test a-posteriori error estimation via ``apost_error``.

    Returns tuples in the form
    ``(shape, perturbation, num_meas, num_reps)``, where if two
    random matrices of the given shape/rank/decay are sampled, the two linops
    to be compared are ``A1`` and ``A1 + perturbation * A2``.
    """
    result = [
        #   shape   pert meas reps errtol
        ((800, 800), 0.1, 300, 10, 0.05),
    ]
    # if request.config.getoption("--quick"):
    #     result = result[:1]
    return result


@pytest.fixture
def scree_config(request):
    """Config to test ``scree_bounds``.

    Returns tuples in the form
    ``(shape, rank, decay, svd_meas, trunc, test_meas_reps)``.

    The ``trunc`` field specifies how many dimensions are **left** after
    truncation.
    """
    result = [
        # ((1000, 1000), 20, 0.02, 100, 50, 500),  # nice gap
        ((1000, 1000), 50, 0.1, 50, 40, 100),  # mostly inside asdf
    ]
    # if request.config.getoption("--quick"):
    #     result = result[:1]
    return result


# ##############################################################################
# # A-POSTERIORI ERROR BOUNDS
# ##############################################################################
def test_apost_error_bounds(apost_bounds_examples):
    """Test case for ``apost_error_bounds`` (formal and correctness).

    * num_measurements must be positive
    * rel_err must be between 0 and 1
    * output probabilities are as expected
    """
    atol = 1e-10
    low_s, high_s = "LOWER: P(err<={}x)", "HIGHER: P(err>={}x)"
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


# ##############################################################################
# # A-POSTERIORI ERROR
# ##############################################################################
def test_apost_error_formal():
    """Formal test case for ``apost_error`` and ``scree_bounds``.

    For apost_error:
    * Mismatching shapes raises error
    * Nonpositive num_meas raises error
    """
    m1, m2 = torch.ones(5, 5), torch.ones(4, 4)
    with pytest.raises(ValueError):
        _ = apost_error(m1, m2, "cpu", m1.device, num_meas=5)
    with pytest.raises(ValueError):
        _ = apost_error(m1, m1 * 2, "cpu", m1.device, num_meas=0)


def test_apost_error_correctness(
    rng_seeds, torch_devices, dtypes_tols, apost_config, apost_noise_types
):
    """Correctness test case for ``apost_error``-

    For all configurations, creates a matrix and its slight perturbation, and
    checks that``apost_error`` performs good estimation of Frobenius norms and
    error.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (
                    shape,
                    perturb,
                    nmeas,
                    nreps,
                    errtol,
                ) in apost_config:
                    h, w = shape
                    seed_delta = max(h, w) + nmeas
                    mat1 = gaussian_noise(shape, 0, 1, seed, dtype, device)
                    mat2 = gaussian_noise(
                        shape, 0, 1, seed + seed_delta, dtype, device
                    )
                    mat2 = mat1 + perturb * mat2
                    lop1 = BasicMatrixLinOp(mat1)
                    lop2 = BasicMatrixLinOp(mat2)
                    # ground truth quantities
                    frob1 = mat1.norm()  #  ** 2
                    frob2 = mat2.norm()  # ** 2
                    error = torch.dist(mat1, mat2)  # ** 2
                    svals2 = torch.linalg.svdvals(mat2)
                    #
                    for adj in (True, False):
                        for noise_type, complex_only in apost_noise_types:
                            if dtype not in COMPLEX_DTYPES and complex_only:
                                # this noise type does not support reals,
                                # skip this iteration
                                continue
                            # run apost_error N times and check it is tight
                            for i in range(nreps):
                                (f1, f2, err), _ = apost_error(
                                    lop1,
                                    lop2,
                                    device,
                                    dtype,
                                    nmeas,
                                    seed + seed_delta * (i + 2),
                                    noise_type,
                                    meas_blocksize=max(h, w),
                                    adj_meas=adj,
                                )
                                f1, f2, err = f1**0.5, f2**0.5, err**0.5
                                # Test correctness of apost_error.
                                # if (error - err) / error < eps, then it holds
                                # (1-eps)error <= err <= (1+eps)error.
                                assert (
                                    relerr(frob1, f1, squared=False) < errtol
                                ), "A-posteriori Frob1 estimator is too bad!"
                                assert (
                                    relerr(frob2, f2, squared=False) < errtol
                                ), "A-posteriori Frob2 estimator is too bad!"
                                assert (
                                    relerr(error, err, squared=False) < errtol
                                ), "A-posteriori error estimator is too bad!"


# ##############################################################################
# # SCREE BOUNDS
# ##############################################################################
def test_scree_correctness(
    rng_seeds, torch_devices, dtypes_tols, scree_config, apost_noise_types
):
    """Correctness test case for ``scree_bounds``-

    For all configurations, creates a matrix with a rapidly decaying spectrum,
    and performs a sketched SVD. Then,  checks that the provided
    ``scree_bounds`` are somewhat tight around the true singular values.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (
                    shape,
                    rank,
                    decay,
                    svd_meas,
                    trunc,
                    test_meas,
                ) in scree_config:
                    # sample matrix and compute singular values
                    h, w = shape
                    mat, _ = RandomLordMatrix.exp(
                        shape,
                        rank=rank,
                        decay=decay,
                        diag_ratio=0.0,
                        symmetric=False,
                        psd=False,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    mat *= 123
                    lop1 = BasicMatrixLinOp(mat)
                    frob = mat.norm()
                    svals = torch.linalg.svdvals(mat)
                    gt_scree = (svals**2).flip(0).cumsum(0).flip(0)[
                        :trunc
                    ] / (frob**2)
                    # compute sketched SVD, a-posteriori error estimation
                    # and scree bounds
                    U, S, Vh = ssvd(
                        lop1,
                        device,
                        dtype,
                        outer_dims=svd_meas,
                        seed=seed + max(h, w),
                        recovery_type="singlepass",
                        meas_blocksize=max(h, w),
                    )
                    U, S, Vh = truncate_decomp(trunc, U, S, Vh)
                    lop2 = CompositeLinOp((("US", U * S), ("Vh", Vh)))
                    (f1, f2, err), _ = apost_error(
                        lop1,
                        lop2,
                        device,
                        dtype,
                        test_meas,
                        seed + 10 * max(h, w),
                        "gaussian",
                        meas_blocksize=max(h, w),
                        adj_meas=False,
                    )
                    f1, f2, err = f1**0.5, f2**0.5, err**0.5
                    scree_lo, scree_hi = scree_bounds(S, f2, err)
                    # ground truth quantities

                    """

                    """
                    import matplotlib.pyplot as plt

                    # relerr(mat, (U*S @ Vh))

                    # plt.clf(); plt.plot(scree_lo, label="lo"); plt.plot(gt_scree, label="GT"); plt.plot(scree_hi, label="hi"); plt.legend(); plt.yscale("log"); plt.show()

                    breakpoint()


def old_scr_tst(
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
