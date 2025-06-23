#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for synthetic matrix utilities.


* Formal tests (ValueError for bad inputs, etc)
* Consistency (for seed reproducibility)
* Correctness (NaN, device, dtype, shape, diag_ratio, symmetry, PSD, svals)
"""


import pytest
import torch

from skerch.synthmat import RandomLordMatrix
from skerch.utils import rademacher_flip, complex_dtype_to_real

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def diag_ratios(request):
    """ """
    result = [0.0, 1.0, 0.1, 10.0]
    if request.config.getoption("--quick"):
        result = result[:2]
    return result


@pytest.fixture
def noise_snrs():
    """ """
    result = [1e-4, 1e-2, 1e-1]
    return result


@pytest.fixture
def poly_decays():
    """ """
    result = [2, 1, 0.5]
    return result


@pytest.fixture
def exp_decays():
    """ """
    result = [0.5, 0.1, 0.01]
    return result


@pytest.fixture
def small_shapes_ranks():
    """ """
    result = [
        # square
        ((10, 10), 1),
        ((10, 10), 2),
        ((10, 10), 10),
        # nonsquare (will ignore noise and sym)
        ((11, 10), 1),
        ((10, 11), 2),
        ((11, 10), 10),
        #
        ((50, 50), 10),
    ]
    return result


@pytest.fixture
def bad_shapes():
    """ """
    result = [
        "x",
        0,
        None,
        (1,),
        (-3, -4),
        (),
    ]
    return result


@pytest.fixture
def bad_decay_types():
    """ """
    result = ["x", 0, None, "asdf", "noise"]
    return result


# ##############################################################################
# # FORMAL
# ##############################################################################
def test_lord_formal(torch_devices, dtypes_tols, bad_shapes, bad_decay_types):
    """Various formal tests for synthetic matrices.

    * _decay_helper svals mut be real, and also >=0 if PSD
    * shape must be matrix-compatible
    * ``diag_ratio`` must be >= 0
    * ``rank`` must be >= 0
    * noise matrix must be square
    * unsupported decay types for get_decay_svals
    """
    for device in torch_devices:
        # _decay_helper svals mut be real, and also >=0 if PSD
        svals_neg = torch.zeros(10, dtype=torch.float32) - 1
        svals_complex = torch.zeros(10, dtype=torch.complex64)
        with pytest.raises(ValueError):
            RandomLordMatrix._decay_helper(svals_complex, (10, 10), 3)
        with pytest.raises(ValueError):
            RandomLordMatrix._decay_helper(
                svals_neg, (10, 10), 3, symmetric=True, psd=True
            )
        for dtype in dtypes_tols.keys():
            # shape must be matrix-compatible
            for bad_shape in bad_shapes:
                with pytest.raises((ValueError, TypeError, RuntimeError)):
                    _ = RandomLordMatrix.noise(
                        bad_shape, rank=3, dtype=dtype, device=device
                    )
                with pytest.raises((ValueError, TypeError, RuntimeError)):
                    _ = RandomLordMatrix.poly(
                        bad_shape, rank=3, dtype=dtype, device=device
                    )
                with pytest.raises((ValueError, TypeError, RuntimeError)):
                    _ = RandomLordMatrix.exp(
                        bad_shape, rank=3, dtype=dtype, device=device
                    )
            # diag_ratio must be >= 0
            r = -0.01
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.noise(
                    (10, 10), rank=3, diag_ratio=r, dtype=dtype, device=device
                )
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.poly(
                    (10, 10), rank=3, diag_ratio=r, dtype=dtype, device=device
                )
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.exp(
                    (10, 10), rank=3, diag_ratio=r, dtype=dtype, device=device
                )
            # rank must be >= 0
            rank = -1
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.noise(
                    (10, 10), rank, dtype=dtype, device=device
                )
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.poly(
                    (10, 10), rank, dtype=dtype, device=device
                )
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.exp(
                    (10, 10), rank, dtype=dtype, device=device
                )
            # noise_matrix must be square
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.noise((5, 10), dtype=dtype, device=device)
            with pytest.raises(ValueError):
                _ = RandomLordMatrix.noise((10, 5), dtype=dtype, device=device)
            # unsupported decay types for get_decay_svals
            for decay_type in bad_decay_types:
                with pytest.raises(ValueError):
                    RandomLordMatrix.get_decay_svals(
                        10, 3, decay_type, 0.1, dtype, device
                    )


# ##############################################################################
# # SEED CONSISTENCY
# ##############################################################################
def _helper_seed_consistency(m1, m2, m3, d1, d2, d3, atol):
    """ """
    assert torch.allclose(m1, m2, atol=atol), "Same seed, different matrix?"
    assert not torch.allclose(m1, m3, atol=atol), "Different seed, same mat?"
    assert torch.allclose(d1, d2, atol=atol), "Same seed, different diag?"
    assert not torch.allclose(d1, d3, atol=atol), "Different seed, same diag?"


def test_seed_consistency(rng_seeds, torch_devices, dtypes_tols):
    """

    Tests:
    * Running the same function twice with same seed yields same results
    * Different seeds yield different results
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                # noise
                mat1, diag1 = RandomLordMatrix.noise(
                    (5, 5), 3, 0.1, 1.0, seed, dtype, device
                )
                mat2, diag2 = RandomLordMatrix.noise(
                    (5, 5), 3, 0.1, 1.0, seed, dtype, device
                )
                mat3, diag3 = RandomLordMatrix.noise(
                    (5, 5), 3, 0.1, 1.0, seed + 1, dtype, device
                )
                _helper_seed_consistency(
                    mat1, mat2, mat3, diag1, diag2, diag3, tol
                )
                #
                for sym, psd in ((False, False), (True, False), (True, True)):
                    # poly
                    mat1, diag1 = RandomLordMatrix.poly(
                        (5, 5), 3, 0.1, 1.0, sym, seed, dtype, device, psd
                    )
                    mat2, diag2 = RandomLordMatrix.poly(
                        (5, 5), 3, 0.1, 1.0, sym, seed, dtype, device, psd
                    )
                    mat3, diag3 = RandomLordMatrix.poly(
                        (5, 5), 3, 0.1, 1.0, sym, seed + 1, dtype, device, psd
                    )
                    _helper_seed_consistency(
                        mat1, mat2, mat3, diag1, diag2, diag3, tol
                    )
                    # exp
                    mat1, diag1 = RandomLordMatrix.exp(
                        (5, 5), 3, 0.1, 1.0, sym, seed, dtype, device, psd
                    )
                    mat2, diag2 = RandomLordMatrix.exp(
                        (5, 5), 3, 0.1, 1.0, sym, seed, dtype, device, psd
                    )
                    mat3, diag3 = RandomLordMatrix.exp(
                        (5, 5), 3, 0.1, 1.0, sym, seed + 1, dtype, device, psd
                    )
                    _helper_seed_consistency(
                        mat1, mat2, mat3, diag1, diag2, diag3, tol
                    )


# ##############################################################################
# # MATRIX CORRECTNESS
# ##############################################################################
def _helper_lord_correctness(
    mat,
    diag,
    device,
    dtype,
    tol,
    shape,
    rank,
    diag_ratio,
    sym,
    psd,
    decay,
    mat_type,
):
    """Various tests for low-rank plus diagonal matrices.

    * Matrix and diagonal have no NaNs
    * Matrix and diagonal have right device, dtype and shape
    * Matrix and diagonal follow diag_ratio
    * Symmetry and PSD-ness is respected
    * Absvals of recovered spectrum are correct for exp and poly
    """
    assert mat.norm() > 0, "Zero matrix?"
    assert not mat.isnan().any(), f"NaNs in {mat.shape, mat.dtype, mat.device}"
    assert not diag.isnan().any(), f"NaNs in diagonal"
    #
    assert device == mat.device.type, "Wrong matrix device?"
    assert device == diag.device.type, "Wrong diagonal device?"
    #
    assert dtype == mat.dtype, "Wrong matrix dtype?"
    diag_dtype = (
        complex_dtype_to_real(dtype) if (mat_type == "noise") else dtype
    )
    assert diag_dtype == diag.dtype, "Wrong diagonal dtype?"
    #
    assert mat.shape == shape, "Wrong matrix shape?"
    assert diag.shape == (min(shape),), "Wrong diagonal shape?"
    # from now on, analyze just the low-rank part
    mat = mat - torch.diag(diag)
    #
    mat_vnorm = mat.norm() / max(shape) ** 0.5
    assert torch.isclose(
        mat_vnorm * diag_ratio, diag.norm(), atol=tol
    ), "mat-diagonal norms do not follow diag_ratio?"
    if sym:
        assert torch.allclose(mat, mat.H, atol=tol), "Matrix not symmetric?"
        S = torch.linalg.eigvalsh(mat)
        if mat_type != "noise":
            is_psd = (S > -tol).all()
            assert is_psd == psd, f"Mismatching PSD-ness? should be PSD={psd}"
    else:
        S = torch.linalg.svdvals(mat)

    #
    if mat_type in {"poly", "exp"}:
        S_ref = RandomLordMatrix.get_decay_svals(
            min(shape), rank, mat_type, decay, dtype, device
        )
        assert torch.allclose(
            S.abs().sort()[0], S_ref.abs().sort()[0], atol=tol
        ), "Wrong recovered spectrum for lowrank mat!"
    elif mat_type == "noise":
        pass
    else:
        raise ValueError(f"Unknown mat_type! {mat_type}")


def test_lord_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    small_shapes_ranks,
    diag_ratios,
    noise_snrs,
    poly_decays,
    exp_decays,
):
    """Seed consistency for synthetic random matrices.

    On the following matrices:
    * given seeds and devices
    * all dtypes and small shapes/ranks
    * symmetric+square, nonsym+psd, nonsym+nonpsd
    * poly, exp, and noise if square

    Runs :func:`_helper_lord_correctness`.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (h, w), r in small_shapes_ranks:
                    for diag_ratio in diag_ratios:
                        for sym in (True, False):
                            if sym and h != w:
                                continue
                            # noise: only if sym
                            if sym:
                                for snr in noise_snrs:
                                    mat, diag = RandomLordMatrix.noise(
                                        (h, w),
                                        r,
                                        snr,
                                        diag_ratio,
                                        seed,
                                        dtype,
                                        device,
                                    )
                                    _helper_lord_correctness(
                                        mat,
                                        diag,
                                        device,
                                        dtype,
                                        tol,
                                        (h, w),
                                        r,
                                        diag_ratio,
                                        sym,
                                        None,
                                        snr,
                                        "noise",
                                    )
                            # exp and poly also have PSD-ness
                            for psd in [True] + [False] if sym else []:
                                # at this point we are testing a bunch of mats:
                                # * if sym, they must be square, skip otherwise
                                # * if not sym, they must be PSD
                                #
                                # exp
                                for decay in exp_decays:
                                    mat, diag = RandomLordMatrix.exp(
                                        (h, w),
                                        r,
                                        decay,
                                        diag_ratio,
                                        sym,
                                        seed,
                                        dtype,
                                        device,
                                        psd,
                                    )
                                    _helper_lord_correctness(
                                        mat,
                                        diag,
                                        device,
                                        dtype,
                                        tol,
                                        (h, w),
                                        r,
                                        diag_ratio,
                                        sym,
                                        psd,
                                        decay,
                                        "exp",
                                    )
                                # poly
                                for decay in poly_decays:
                                    mat, diag = RandomLordMatrix.poly(
                                        (h, w),
                                        r,
                                        decay,
                                        diag_ratio,
                                        sym,
                                        seed,
                                        dtype,
                                        device,
                                        psd,
                                    )
                                    _helper_lord_correctness(
                                        mat,
                                        diag,
                                        device,
                                        dtype,
                                        tol,
                                        (h, w),
                                        r,
                                        diag_ratio,
                                        sym,
                                        psd,
                                        decay,
                                        "poly",
                                    )
