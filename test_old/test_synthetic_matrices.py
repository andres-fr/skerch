#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for synthetic matrix utilities."""


import pytest
import torch

from skerch.synthmat import SynthMat
from skerch.utils import rademacher_flip

from . import exp_decay, poly_decay, rng_seeds, snr_lowrank_noise, torch_devices


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
def dims_ranks_square(request):
    """Dimension and rank for square matrices."""
    result = [
        (1, 1),
        (10, 1),
        (100, 10),
        (1_000, 10),
        (1_000, 50),
    ]
    if request.config.getoption("--quick"):
        result = result[:3]
    return result


@pytest.fixture
def heights_widths_ranks_fat(request):
    """Shape and rank for non-square matrices."""
    result = [
        (1, 10, 1),
        (10, 100, 1),
        (100, 1_000, 10),
        (1_000, 10_000, 100),
    ]
    if request.config.getoption("--quick"):
        result = result[:2]
    return result


@pytest.fixture
def decay_ew_atol():
    """Absolute tolerance to determine that eigenvals are strictly decaying."""
    result = {torch.float64: 1e-7, torch.float32: 1e-4}
    return result


# ##############################################################################
# # SEED CONSISTENCY
# ##############################################################################
def test_seed_consistency(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    heights_widths_ranks_fat,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
):
    """Seed consistency for synthetic random matrices.

    Tests that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for h, w, r in heights_widths_ranks_fat:
                    # lowrank+noise
                    for snr in snr_lowrank_noise:
                        mat1 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        mat2 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed + 1,
                            dtype=dtype,
                            device=device,
                        )
                        mat3 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert torch.allclose(
                            mat1, mat3, rtol
                        ), "Same seed different matrix?"
                        assert (
                            mat1 != mat2
                        ).any(), f"Different seed same matrix?, {mat1, mat3}"
                    # exp decay
                    for dec in exp_decay:
                        for psd in (True, False):
                            mat1 = SynthMat.exp_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            mat2 = SynthMat.exp_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed + 1,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            mat3 = SynthMat.exp_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            assert torch.allclose(
                                mat1, mat3, rtol
                            ), "Same seed different matrix?"
                            assert (
                                mat1 != mat2
                            ).any(), (
                                f"Different seed same matrix?, {mat1, mat3}"
                            )
                    # poly decay
                    for dec in poly_decay:
                        for psd in (True, False):
                            mat1 = SynthMat.poly_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            mat2 = SynthMat.poly_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed + 1,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            mat3 = SynthMat.poly_decay(
                                shape=(h, w),
                                rank=r,
                                decay=dec,
                                symmetric=False,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            assert torch.allclose(
                                mat1, mat3, rtol
                            ), "Same seed different matrix?"
                            assert (
                                mat1 != mat2
                            ).any(), (
                                f"Different seed same matrix?, {mat1, mat3}"
                            )


# ##############################################################################
# # SYMMETRIC
# ##############################################################################
def test_symmetric(  # noqa: C901  # ignore "is too complex"
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    dims_ranks_square,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
    decay_ew_atol,
):
    """Tests assumed properties for symmetric synthetic matrices.

    Creates square, symmetric synthetic matrices and tests that:

    * there are no NaNs
    * they are indeed symmetric
    * their diagonals/spectra are correct
    * if non-PSD, spectra have significantly negative entries and recovery is
      still correct
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for dim, r in dims_ranks_square:
                    # lowrank+noise
                    for snr in snr_lowrank_noise:
                        mat = SynthMat.lowrank_noise(
                            shape=(dim, dim),
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert not mat.isnan().any(), f"{mat, device, dtype}"
                        assert torch.allclose(
                            mat, mat.T, rtol=rtol
                        ), "Matrix not symmetric?"
                        assert all(
                            mat[range(r), range(r)] >= 1
                        ), "mat[:rank] is not >=1 for given rank?"
                    # exp decay
                    for dec in exp_decay:
                        for psd in (True, False):
                            mat = SynthMat.exp_decay(
                                shape=(dim, dim),
                                rank=r,
                                decay=dec,
                                symmetric=True,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            ew = torch.linalg.eigvalsh(mat).flip(0)  # desc ord
                            assert (
                                not mat.isnan().any()
                            ), f"{mat, device, dtype}"
                            assert torch.allclose(
                                mat, mat.T, rtol=rtol
                            ), "Matrix not symmetric?"
                            assert torch.allclose(
                                ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                            ), "ew[:rank] should be == 1"
                            ew_dec = 10.0 ** -(
                                dec * torch.arange(1, dim - r + 1, dtype=dtype)
                            ).to(device)
                            if not psd:
                                # apply rademacher to the decay
                                rademacher_flip(
                                    ew_dec, seed=seed + 1, inplace=True
                                )
                                # sort recovered eigenvals by descending mag
                                _, perm = ew.abs().sort(descending=True)
                                ew = ew[perm]
                                # check that we actually have negatives
                                if ew_dec.numel() > 0:
                                    assert (
                                        ew_dec < -abs(decay_ew_atol[dtype])
                                    ).any(), (
                                        "non-PSD should have "
                                        + "negative eigvals!"
                                    )
                            assert torch.allclose(
                                ew[r:],
                                ew_dec,
                                rtol=rtol,
                                # added atol due to eigvalsh
                                atol=decay_ew_atol[dtype],
                            ), "Eigenval decay mismatch!"
                    # poly decay
                    for dec in poly_decay:
                        for psd in (True, False):
                            mat = SynthMat.poly_decay(
                                shape=(dim, dim),
                                rank=r,
                                decay=dec,
                                symmetric=True,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=psd,
                            )
                            ew = torch.linalg.eigvalsh(mat).flip(0)  # desc ord
                            assert (
                                not mat.isnan().any()
                            ), f"{mat, device, dtype}"
                            assert torch.allclose(
                                mat, mat.T, rtol=rtol
                            ), "Matrix not symmetric?"
                            assert torch.allclose(
                                ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                            ), "ew[:rank] should be == 1"
                            ew_dec = (
                                torch.arange(2, dim - r + 2, dtype=dtype)
                                ** (-float(dec))
                            ).to(device)
                            if not psd:
                                # apply rademacher to the decay
                                rademacher_flip(
                                    ew_dec, seed=seed + 1, inplace=True
                                )
                                # sort recovered eigenvals by descending mag
                                _, perm = ew.abs().sort(descending=True)
                                ew = ew[perm]
                                # check that we actually have negatives
                                if ew_dec.numel() > 0:
                                    assert (
                                        ew_dec < -abs(decay_ew_atol[dtype])
                                    ).any(), (
                                        "non-PSD should have "
                                        + "negative eigvals!"
                                    )
                            assert torch.allclose(
                                ew[r:],
                                ew_dec,
                                rtol=rtol,
                                # added atol due to eigvalsh
                                atol=decay_ew_atol[dtype],
                            ), "Eigenval decay mismatch!"


# ##############################################################################
# # ASYMMETRIC
# ##############################################################################
def test_asymmetric_nonsquare(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    heights_widths_ranks_fat,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
    decay_ew_atol,
):
    """Tests assumed properties for asymmetric synthetic matrices.

    Create square, symmetric synthetic matrices and tests that:
    * there are no NaNs
    * their diagonals/spectra are correct

    .. note::

      Since lowrank+noise must be symmetric, it is omitted here.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for h, w, r in heights_widths_ranks_fat:
                    min_dim = min(h, w)
                    # exp decay
                    for dec in exp_decay:
                        mat = SynthMat.exp_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.svdvals(mat)  # desc order
                        assert not mat.isnan().any(), f"{mat, device, dtype}"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), f"ew[:rank] should be == 1, {ew}"
                        ew_dec = 10.0 ** -(
                            dec * torch.arange(1, min_dim - r + 1, dtype=dtype)
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            # added atol due to eigvalsh
                            atol=decay_ew_atol[dtype],
                        ), "Eigenval decay mismatch!"
                    # poly decay
                    for dec in poly_decay:
                        mat = SynthMat.poly_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.svdvals(mat)  # desc order
                        assert not mat.isnan().any(), f"{mat, device, dtype}"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), "ew[:rank] should be == 1"
                        ew_dec = (
                            torch.arange(2, min_dim - r + 2, dtype=dtype)
                            ** (-float(dec))
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            # added atol due to eigvalsh
                            atol=decay_ew_atol[dtype],
                        ), "Eigenval decay mismatch!"
