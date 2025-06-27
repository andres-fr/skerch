#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for all noisy measurements

SSRFT

rademacher

gaussian



TODO:
* design a measurement linop that supports shape and @, but can also be
  parallelized if needed (e.g. via get_row or sth). Ideally this is all
  done through byvectorlinop, including ssrft

* once we have the measurement linops in place, test
  - seed consistency
  - (quasi-) orthogonality of randmats
  - formal corner cases




LATER TODO:


* Implement perform_measurement as per below.
  - test correctness and formal (valerr etc)
  - test that parallel versions are equal to inline
* Implement 3 recovery methods
  - test correctness and formal

* Implement sketched algorithms, at least svd and lord


TODO:

* test dct2
* add utest of phase_noise to utils
* create test for phase noise, also conj, check it looks OK
* same for SSRFT
* same for the measurement function, and we are done with meas


"""


import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    TransposedLinOp,
)

from skerch.measurements import (
    perform_measurement,
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
)

from skerch.utils import BadShapeError, BadSeedError

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 3e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def complex_dtypes_tols():
    """Error tolerances for each complex dtype."""
    result = {
        torch.complex64: 1e-5,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def noise_linop_types():
    """Class names for all noise linops to be tested.

    :returns: Collection of pairs ``(lop_type, is_complex_only)``
    """
    result = {
        (GaussianNoiseLinOp, False),
        (RademacherNoiseLinOp, False),
        (PhaseNoiseLinOp, True),
    }
    return result


# ##############################################################################
# # TESTS
# ##############################################################################
def test_measurements_formal(
    rng_seeds, torch_devices, dtypes_tols, noise_linop_types
):
    """Formal test case for measurement linops.

    For every noise linop tests:
    * Repr creates correct strings
    * Register triggers error if overlapping seeds are used if active
    * Get_vector triggers error for invalid index, and returns right dtype and
      device otherwise
    * Invalid index to ``get_vector`` triggers error
    * Deterministic behaviour (fwd and adjoint): running twice is same
    * Seed consistency
    * Output is in requested datatype and device

    For complex_only linops, it also tests:
    * Providing a noncomplex dtype raises a ``ValueError``
    """
    # correct string conversion
    hw = (3, 3)
    lop = RademacherNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = "<RademacherNoiseLinOp(3x3, seed=0, dtype=torch.float32, by_row=False)>"
    assert str(lop) == s, "Unexpected repr for Rademacher noise linop!"
    lop = GaussianNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<GaussianNoiseLinOp(3x3, mean=0.0, std=1.0, seed=0, "
        + "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Gaussian noise linop!"
    lop = PhaseNoiseLinOp(hw, 0, torch.complex64, by_row=False, register=False)
    s = (
        "<PhaseNoiseLinOp(3x3, conj=False, seed=0, "
        + "dtype=torch.complex64, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Phase noise linop!"
    #
    for lop_type, complex_only in noise_linop_types:
        if complex_only:
            dtype1, dtype2 = torch.complex64, torch.complex128
        else:
            dtype1, dtype2 = torch.float32, torch.float64
        # dtype1 = torch.complex64 if lop_type is PhaseNoiseLinOp torch.float32
        lop = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        # register triggers for overlapping seeds regardless of other factors
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((20, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype2, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=True)
        # invalid index triggers error
        with pytest.raises(ValueError):
            lop.get_vector(idx=-1, device="cpu")
        with pytest.raises(ValueError):
            lop.get_vector(idx=5, device="cpu")
        #
        for seed in rng_seeds:
            for device in torch_devices:
                for dtype, tol in dtypes_tols.items():
                    #
                    if complex_only and dtype not in {
                        torch.complex32,
                        torch.complex64,
                        torch.complex128,
                    }:
                        with pytest.raises(ValueError):
                            _ = lop_type(hw, 0, dtype, register=False)
                        continue
                    #
                    hw = (100, 2)
                    lop1 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop2 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop3 = lop_type(
                        hw, seed + 5, dtype, by_row=False, register=False
                    )
                    # deterministic behaviour and seed consistency
                    mat1a = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1b = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1c = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=True
                    )
                    mat2 = linop_to_matrix(
                        lop2, lop1.dtype, device, adjoint=False
                    )
                    mat3 = linop_to_matrix(
                        lop3, lop1.dtype, device, adjoint=False
                    )
                    assert (
                        mat1a == mat1b
                    ).all(), f"Nondeterministic linop? {lop1}"
                    assert (
                        mat1a == mat1c
                    ).all(), f"Different fwd and adjoint? {lop1}"
                    assert (
                        mat1a == mat2
                    ).all(), f"Same seed, differentl linop? {lop1}"
                    #
                    for col in mat1a.H:
                        cosim = abs(col @ mat3) / (col.norm() ** 2)
                        assert (
                            cosim < 0.5
                        ).all(), "Different seeds, similar vectors? {lop1}"
                    # dtype and device check
                    assert mat1a.dtype == dtype, "Mismatching dtype!"
                    assert mat1a.device.type == device, "Mismatching device!"


def test_phasenoise_conj(rng_seeds, torch_devices, complex_dtypes_tols):
    """Test case for conjugation of ``PhaseNoise`` linop.

    For all seeds, devices and complex dtypes, tests that:
    * Conj of linop produces elementwise conjugated entries
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in complex_dtypes_tols.items():
                lop = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=False, register=False
                )
                lop_conj = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=True, register=False
                )
                mat = linop_to_matrix(lop, lop.dtype, device, adjoint=False)
                mat_conj = linop_to_matrix(
                    lop_conj, lop.dtype, device, adjoint=False
                )
                assert (mat.conj() == mat_conj).all(), "Wrong conj linop?"


# ##############################################################################
# # SSRFT
# ##############################################################################
def dct2(x, norm="ortho"):
    """One-dimensional Discrete Cosine Transform - Type 2.

    :param x: Tensor of shape ``(..., n)``, assumed to be real
    """
    len_sig = x.shape[-1]
    len_dct = (len_sig - 1) // 2 + 1
    #
    spiral = ((-0.5j * torch.pi / len_sig) * torch.arange(len_sig)).exp()
    #
    v = torch.empty_like(x)
    v[:len_dct] = x[::2]
    v[len_dct:] = x.flip(-1)[1::2] if (len_sig % 2) else x.flip(-1)[::2]
    #
    result = ((2**0.5) * torch.fft.fft(v, norm=norm) * spiral).real
    return result


def test_ssrft():
    """"""
    import torch_dct as dct
    import matplotlib.pyplot as plt
    from skerch.measurements import SSRFT

    aa = torch.zeros(3, 1000, dtype=torch.float64)
    aa[5] = 1

    bb = SSRFT.ssrft(aa, 1000, seed=12345, dct_norm="ortho")


    TODO:
    * adapt ssrft to also admit complex (phase noise instead of rademacher)
    * integrate ssrft into a linop
    * test it!

    breakpoint()
    # idx=5; aa = torch.zeros(1000); aa[idx] = 1; bb = dct.dct(aa, norm="ortho"); cc=dct2(aa)
    # plt.clf(); plt.plot(cc); plt.show()


# def test_no_nans(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Tests that SSRFT yields no NaNs."""
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for device in torch_devices:
#                 for dtype, _rtol in f64_rtol.items():
#                     x = torch.randn(w, dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = y @ ssrft
#                     #
#                     assert not x.isnan().any(), f"{ssrft, device, dtype}"
#                     assert not y.isnan().any(), f"{ssrft, device, dtype}"
#                     assert not xx.isnan().any(), f"{ssrft, device, dtype}"


# def test_invertible(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Invertibility/orthogonality of quare SSRFT.

#     Tests that, when input and output dimensionality are the same, the SSRFT
#     operator is orthogonal, i.e. we can recover the input exactly via an
#     adjoint operation.

#     Also tests that it works for mat-vec and mat-mat formats.
#     """
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for device in torch_devices:
#                 for dtype, rtol in f64_rtol.items():
#                     # matvec
#                     x = torch.randn(w, dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = y @ ssrft
#                     #
#                     assert torch.allclose(
#                         x, xx, rtol=rtol
#                     ), f"MATVEC: {ssrft, device, dtype}"
#                     # matmat
#                     x = torch.randn((w, 2), dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = (y.T @ ssrft).T
#                     #
#                     assert torch.allclose(
#                         x, xx, rtol=rtol
#                     ), f"MATMAT: {ssrft, device, dtype}"
#                     # matmat-shape tests
#                     assert len(y.shape) == 2
#                     assert len(xx.shape) == 2
#                     assert y.shape[-1] == 2
#                     assert xx.shape[-1] == 2


# def test_seed_consistency(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Seed consistency of SSRFT.

#     Test that same seed and shape lead to same operator with same results,
#     and different otherwise.
#     """
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             ssrft_same = SSRFT((h, w), seed=seed)
#             ssrft_diff = SSRFT((h, w), seed=seed + 1)
#             for device in torch_devices:
#                 for dtype, _rtol in f64_rtol.items():
#                     # matvec
#                     x = torch.randn(w, dtype=dtype).to(device)
#                     assert ((ssrft @ x) == (ssrft_same @ x)).all()
#                     # here, dim=1 may indeed result in same output, since
#                     # there are no permutations or index-pickings, so 50/50.
#                     # therefore we ignore that case.
#                     if x.numel() > 1:
#                         assert ((ssrft @ x) != (ssrft_diff @ x)).any()


# def test_device_consistency(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Seed consistency of SSRFT across different devices.

#     Test that same seed and shape lead to same operator with same results,
#     even when device is different.
#     """
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for dtype, rtol in f64_rtol.items():
#                 # apply SSRFT on given devices and check results are equal
#                 x = torch.randn(w, dtype=dtype)
#                 y = [(ssrft @ x.to(device)).cpu() for device in torch_devices]
#                 for yyy in y:
#                     assert torch.allclose(
#                         yyy, y[0], rtol=rtol
#                     ), "SSRFT inconsistency among devices!"


# def test_unsupported_tall_ssrft(rng_seeds, fat_shapes):
#     """Tail SSRFT linops are not supported."""
#     for seed in rng_seeds:
#         for h, w in fat_shapes:
#             with pytest.raises(BadShapeError):
#                 # If this line throws a BadShapeError, the test passes
#                 SSRFT((w, h), seed=seed)


# def test_input_shape_mismatch(rng_seeds, fat_shapes, torch_devices, f64_rtol):
#     """Test case for SSRFT shape consistency."""
#     for seed in rng_seeds:
#         for h, w in fat_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for device in torch_devices:
#                 for dtype, _rtol in f64_rtol.items():
#                     # forward matmul
#                     x = torch.empty(w + 1, dtype=dtype).to(device)
#                     with pytest.raises(BadShapeError):
#                         ssrft @ x
#                     # adjoint matmul
#                     x = torch.empty(h + 1, dtype=dtype).to(device)
#                     with pytest.raises(BadShapeError):
#                         x @ ssrft
