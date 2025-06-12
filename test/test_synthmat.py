#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for synthetic matrix utilities."""


import pytest
import torch

from skerch.synthmat import RandomLordMatrix
from skerch.utils import rademacher_flip, dtype_to_real

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
def test_seed_consistency():
    """ """
    breakpoint()


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
    diag_dtype = dtype_to_real(dtype) if (mat_type == "noise") else dtype
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


def test_seed_consistency():
    """ """
    pass
    # seed = 12345
    # m1, d1 = RandomLordMatrix.exp(
    #     (100, 100), 10, 0.5, diag_ratio=-1.0, seed=seed, dtype=torch.float64
    # )
    # m2, d2 = RandomLordMatrix.exp(
    #     (100, 100), 10, 0.5, diag_ratio=1.0, seed=seed, dtype=torch.complex128
    # )
    # import matplotlib.pyplot as plt

    # aa, bb, cc = torch.linalg.svd(m1 - 0 * torch.diag(d1))
    # aaa, bbb, ccc = torch.linalg.svd(m2 - 0 * torch.diag(d2))
    # breakpoint()
    # # plt.clf(); plt.plot(bb); plt.show()
    # # plt.clf(); plt.imshow(aa.real); plt.show()
    # # torch.dist(m1, (aa * bb) @ cc)
    # # torch.dist(m2, (aaa * bbb) @ ccc)

    # # torch.dist(m1, m1.H)

    # for snr in snr_lowrank_noise:
    #     mat1 = SynthMat.lowrank_noise(
    #         shape=(h, h),  # L+N must be square
    #         rank=r,
    #         snr=snr,
    #         seed=seed,
    #         dtype=dtype,
    #         device=device,
    #     )
    #     mat2 = SynthMat.lowrank_noise(
    #         shape=(h, h),  # L+N must be square
    #         rank=r,
    #         snr=snr,
    #         seed=seed + 1,
    #         dtype=dtype,
    #         device=device,
    #     )
    #     mat3 = SynthMat.lowrank_noise(
    #         shape=(h, h),  # L+N must be square
    #         rank=r,
    #         snr=snr,
    #         seed=seed,
    #         dtype=dtype,
    #         device=device,
    #     )
    #     assert torch.allclose(
    #         mat1, mat3, rtol
    #     ), "Same seed different matrix?"
    #     assert (
    #         mat1 != mat2
    #     ).any(), f"Different seed same matrix?, {mat1, mat3}"
    # # exp decay
    # for dec in exp_decay:
    #     for psd in (True, False):
    #         mat1 = SynthMat.exp_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         mat2 = SynthMat.exp_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed + 1,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         mat3 = SynthMat.exp_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         assert torch.allclose(
    #             mat1, mat3, rtol
    #         ), "Same seed different matrix?"
    #         assert (
    #             mat1 != mat2
    #         ).any(), (
    #             f"Different seed same matrix?, {mat1, mat3}"
    #         )
    # # poly decay
    # for dec in poly_decay:
    #     for psd in (True, False):
    #         mat1 = SynthMat.poly_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         mat2 = SynthMat.poly_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed + 1,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         mat3 = SynthMat.poly_decay(
    #             shape=(h, w),
    #             rank=r,
    #             decay=dec,
    #             symmetric=False,
    #             seed=seed,
    #             dtype=dtype,
    #             device=device,
    #             psd=psd,
    #         )
    #         assert torch.allclose(
    #             mat1, mat3, rtol
    #         ), "Same seed different matrix?"
    #         assert (
    #             mat1 != mat2
    #         ).any(), (
    #             f"Different seed same matrix?, {mat1, mat3}"
    #         )


# # ##############################################################################
# # # SYMMETRIC
# # ##############################################################################
# def test_symmetric(  # noqa: C901  # ignore "is too complex"
#     rng_seeds,
#     torch_devices,
#     f64_rtol,
#     f32_rtol,
#     dims_ranks_square,
#     snr_lowrank_noise,
#     exp_decay,
#     poly_decay,
#     decay_ew_atol,
# ):
#     """Tests assumed properties for symmetric synthetic matrices.

#     Creates square, symmetric synthetic matrices and tests that:

#     * there are no NaNs
#     * they are indeed symmetric
#     * their diagonals/spectra are correct
#     * if non-PSD, spectra have significantly negative entries and recovery is
#       still correct
#     """
#     for seed in rng_seeds:
#         for device in torch_devices:
#             for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
#                 # symmetric tests
#                 for dim, r in dims_ranks_square:
#                     # lowrank+noise
#                     for snr in snr_lowrank_noise:
#                         mat = SynthMat.lowrank_noise(
#                             shape=(dim, dim),
#                             rank=r,
#                             snr=snr,
#                             seed=seed,
#                             dtype=dtype,
#                             device=device,
#                         )
#                         assert not mat.isnan().any(), f"{mat, device, dtype}"
#                         assert torch.allclose(
#                             mat, mat.T, rtol=rtol
#                         ), "Matrix not symmetric?"
#                         assert all(
#                             mat[range(r), range(r)] >= 1
#                         ), "mat[:rank] is not >=1 for given rank?"
#                     # exp decay
#                     for dec in exp_decay:
#                         for psd in (True, False):
#                             mat = SynthMat.exp_decay(
#                                 shape=(dim, dim),
#                                 rank=r,
#                                 decay=dec,
#                                 symmetric=True,
#                                 seed=seed,
#                                 dtype=dtype,
#                                 device=device,
#                                 psd=psd,
#                             )
#                             ew = torch.linalg.eigvalsh(mat).flip(0)  # desc ord
#                             assert (
#                                 not mat.isnan().any()
#                             ), f"{mat, device, dtype}"
#                             assert torch.allclose(
#                                 mat, mat.T, rtol=rtol
#                             ), "Matrix not symmetric?"
#                             assert torch.allclose(
#                                 ew[:r], torch.ones_like(ew[:r]), rtol=rtol
#                             ), "ew[:rank] should be == 1"
#                             ew_dec = 10.0 ** -(
#                                 dec * torch.arange(1, dim - r + 1, dtype=dtype)
#                             ).to(device)
#                             if not psd:
#                                 # apply rademacher to the decay
#                                 rademacher_flip(
#                                     ew_dec, seed=seed + 1, inplace=True
#                                 )
#                                 # sort recovered eigenvals by descending mag
#                                 _, perm = ew.abs().sort(descending=True)
#                                 ew = ew[perm]
#                                 # check that we actually have negatives
#                                 if ew_dec.numel() > 0:
#                                     assert (
#                                         ew_dec < -abs(decay_ew_atol[dtype])
#                                     ).any(), (
#                                         "non-PSD should have "
#                                         + "negative eigvals!"
#                                     )
#                             assert torch.allclose(
#                                 ew[r:],
#                                 ew_dec,
#                                 rtol=rtol,
#                                 # added atol due to eigvalsh
#                                 atol=decay_ew_atol[dtype],
#                             ), "Eigenval decay mismatch!"
#                     # poly decay
#                     for dec in poly_decay:
#                         for psd in (True, False):
#                             mat = SynthMat.poly_decay(
#                                 shape=(dim, dim),
#                                 rank=r,
#                                 decay=dec,
#                                 symmetric=True,
#                                 seed=seed,
#                                 dtype=dtype,
#                                 device=device,
#                                 psd=psd,
#                             )
#                             ew = torch.linalg.eigvalsh(mat).flip(0)  # desc ord
#                             assert (
#                                 not mat.isnan().any()
#                             ), f"{mat, device, dtype}"
#                             assert torch.allclose(
#                                 mat, mat.T, rtol=rtol
#                             ), "Matrix not symmetric?"
#                             assert torch.allclose(
#                                 ew[:r], torch.ones_like(ew[:r]), rtol=rtol
#                             ), "ew[:rank] should be == 1"
#                             ew_dec = (
#                                 torch.arange(2, dim - r + 2, dtype=dtype)
#                                 ** (-float(dec))
#                             ).to(device)
#                             if not psd:
#                                 # apply rademacher to the decay
#                                 rademacher_flip(
#                                     ew_dec, seed=seed + 1, inplace=True
#                                 )
#                                 # sort recovered eigenvals by descending mag
#                                 _, perm = ew.abs().sort(descending=True)
#                                 ew = ew[perm]
#                                 # check that we actually have negatives
#                                 if ew_dec.numel() > 0:
#                                     assert (
#                                         ew_dec < -abs(decay_ew_atol[dtype])
#                                     ).any(), (
#                                         "non-PSD should have "
#                                         + "negative eigvals!"
#                                     )
#                             assert torch.allclose(
#                                 ew[r:],
#                                 ew_dec,
#                                 rtol=rtol,
#                                 # added atol due to eigvalsh
#                                 atol=decay_ew_atol[dtype],
#                             ), "Eigenval decay mismatch!"


# # ##############################################################################
# # # ASYMMETRIC
# # ##############################################################################
# def test_asymmetric_nonsquare(
#     rng_seeds,
#     torch_devices,
#     f64_rtol,
#     f32_rtol,
#     heights_widths_ranks_fat,
#     snr_lowrank_noise,
#     exp_decay,
#     poly_decay,
#     decay_ew_atol,
# ):
#     """Tests assumed properties for asymmetric synthetic matrices.

#     Create square, symmetric synthetic matrices and tests that:
#     * there are no NaNs
#     * their diagonals/spectra are correct

#     .. note::

#       Since lowrank+noise must be symmetric, it is omitted here.
#     """
#     for seed in rng_seeds:
#         for device in torch_devices:
#             for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
#                 # symmetric tests
#                 for h, w, r in heights_widths_ranks_fat:
#                     min_dim = min(h, w)
#                     # exp decay
#                     for dec in exp_decay:
#                         mat = SynthMat.exp_decay(
#                             shape=(h, w),
#                             rank=r,
#                             decay=dec,
#                             symmetric=False,
#                             seed=seed,
#                             dtype=dtype,
#                             device=device,
#                         )
#                         ew = torch.linalg.svdvals(mat)  # desc order
#                         assert not mat.isnan().any(), f"{mat, device, dtype}"
#                         assert torch.allclose(
#                             ew[:r], torch.ones_like(ew[:r]), rtol=rtol
#                         ), f"ew[:rank] should be == 1, {ew}"
#                         ew_dec = 10.0 ** -(
#                             dec * torch.arange(1, min_dim - r + 1, dtype=dtype)
#                         ).to(device)
#                         assert torch.allclose(
#                             ew[r:],
#                             ew_dec,
#                             rtol=rtol,
#                             # added atol due to eigvalsh
#                             atol=decay_ew_atol[dtype],
#                         ), "Eigenval decay mismatch!"
#                     # poly decay
#                     for dec in poly_decay:
#                         mat = SynthMat.poly_decay(
#                             shape=(h, w),
#                             rank=r,
#                             decay=dec,
#                             symmetric=False,
#                             seed=seed,
#                             dtype=dtype,
#                             device=device,
#                         )
#                         ew = torch.linalg.svdvals(mat)  # desc order
#                         assert not mat.isnan().any(), f"{mat, device, dtype}"
#                         assert torch.allclose(
#                             ew[:r], torch.ones_like(ew[:r]), rtol=rtol
#                         ), "ew[:rank] should be == 1"
#                         ew_dec = (
#                             torch.arange(2, min_dim - r + 2, dtype=dtype)
#                             ** (-float(dec))
#                         ).to(device)
#                         assert torch.allclose(
#                             ew[r:],
#                             ew_dec,
#                             rtol=rtol,
#                             # added atol due to eigvalsh
#                             atol=decay_ew_atol[dtype],
#                         ), "Eigenval decay mismatch!"
