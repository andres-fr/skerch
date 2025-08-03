#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`recovery`."""


import pytest
import torch
import numpy as np

from skerch.utils import gaussian_noise, svd
from skerch.synthmat import RandomLordMatrix
from skerch.recovery import singlepass, nystrom, oversampled
from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 3e-5,
        torch.complex64: 3e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
def uv_test_helper(mat, U, Vh, atol):
    """UV test helper.

    Given the decomposition ``mat = U @ Vh``, tests that:

    * The equality actually holds
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(mat, torch.Tensor) else np.allclose
    # correctness of result
    assert allclose(mat, U @ Vh, atol=atol), "Incorrect recovery!"
    # matching device and type
    assert U.device == mat.device, "Incorrect U device!"
    assert Vh.device == mat.device, "Incorrect V device!"
    assert U.dtype == mat.dtype, "Incorrect U dtype!"
    assert Vh.dtype == mat.dtype, "Incorrect V dtype!"


def svd_test_helper(mat, svals, I, U, S, Vh, atol):
    """SVD test helper.

    Given the produced SVD of ``mat``, tests that:

    * The SVD is actually close to the matrix
    * The recovered ``S`` are close to the corresponding ``svals``
    * ``U, V`` have orthonormal columns
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(I, torch.Tensor) else np.allclose
    # correctness of result
    assert allclose(mat, (U * S) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered U, V
    assert allclose(I, U.conj().T @ U, atol=atol), "U not orthogonal?"
    assert allclose(I, Vh @ Vh.conj().T, atol=atol), "V not orthogonal?"
    # correctness of recovered svals
    assert allclose(svals[: len(S)], S, atol=atol), "Incorrect svals!"
    # matching device and type
    assert U.device == mat.device, "Incorrect U device!"
    assert S.device == mat.device, "Incorrect S device!"
    assert Vh.device == mat.device, "Incorrect V device!"
    assert U.dtype == mat.dtype, "Incorrect U dtype!"
    assert S.dtype == mat.real.dtype, "Incorrect S dtype!"
    assert Vh.dtype == mat.dtype, "Incorrect V dtype!"


# ##############################################################################
# # PERFORM MEASUREMENTS
# ##############################################################################
def test_recovery_formal(rng_seeds, torch_devices, dtypes_tols):
    """Test case for recovery (formal and correctness).

    For torch/numpy inputs, and for all recovery methods, runs
    :func:`svd_test_helper`.
    """
    hw, rank, meas = (50, 50), 5, 25
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                # torch identity, low-rank matrix and sketches
                I1 = torch.eye(meas, dtype=dtype, device=device)
                mat1, _ = RandomLordMatrix.exp(
                    hw,
                    rank,
                    decay=100,
                    diag_ratio=0.0,
                    symmetric=False,
                    psd=False,
                    seed=seed,
                    dtype=dtype,
                    device=device,
                )
                U1, S1, Vh1 = svd(mat1)
                right1 = gaussian_noise(
                    (hw[1], meas), seed=seed + 100, device=device, dtype=dtype
                )
                left1 = gaussian_noise(
                    (meas, hw[0]), seed=seed + 101, device=device, dtype=dtype
                )
                Y1 = mat1 @ right1  # tall
                Z1 = left1 @ mat1  # fat
                # numpy corresponding objects
                I2 = I1.cpu().numpy()
                mat2 = mat1.cpu().numpy()
                S2 = S1.cpu().numpy()
                Y2, Z2 = Y1.cpu().numpy(), Z1.cpu().numpy()
                right2 = right1.cpu().numpy()
                #
                for modality, I, mat, S, Y, Z, right in (
                    ("torch", I1, mat1, S1, Y1, Z1, right1),
                    ("numpy", I2, mat2, S2, Y2, Z2, right2),
                ):
                    # singlepass
                    Urec, Srec, Vhrec = singlepass(Y, Z, right)
                    try:
                        svd_test_helper(mat, S, I, Urec, Srec, Vhrec, tol)
                    except AssertionError as ae:
                        errmsg = f"Singlepass {modality} error!"
                        raise AssertionError(errmsg) from ae
                    # nystrom - UV
                    Urec, Vhrec = nystrom(Y, Z, right, as_svd=False)
                    try:
                        uv_test_helper(mat, Urec, Vhrec, tol)
                    except AssertionError as ae:
                        errmsg = f"Nystrom-UV {modality} error!"
                        raise AssertionError(errmsg) from ae
                    # nystrom - SVD
                    Urec, Srec, Vhrec = nystrom(Y, Z, right, as_svd=True)
                    try:
                        svd_test_helper(mat, S, I, Urec, Srec, Vhrec, tol)
                    except AssertionError as ae:
                        errmsg = f"Nystrom-SVD {modality} error!"
                        raise AssertionError(errmsg) from ae
                    # oversampled
