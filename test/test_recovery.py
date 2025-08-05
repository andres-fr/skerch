#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`recovery`."""


import pytest
import torch
import numpy as np

from skerch.utils import gaussian_noise, svd, eigh
from skerch.synthmat import RandomLordMatrix
from skerch.recovery import singlepass, nystrom, oversampled
from skerch.recovery import singlepass_h, nystrom_h, oversampled_h
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
def general_recovery_shapes():
    """Tuples in the form ``((height, width), rank, outermeas, innermeas)."""
    result = [
        ((50, 50), 5, 25, 35),
        ((40, 50), 5, 25, 35),
        ((50, 40), 5, 25, 35),
    ]
    return result


@pytest.fixture
def hermitian_recovery_shapes():
    """Tuples in the form ``(heigth, rank, outermeas, innermeas)."""
    result = [
        (50, 5, 25, 35),
    ]
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
    diff = torch.diff if isinstance(I, torch.Tensor) else np.diff
    # correctness of result
    assert allclose(mat, (U * S) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered U, V
    assert allclose(I, U.conj().T @ U, atol=atol), "U not orthogonal?"
    assert allclose(I, Vh @ Vh.conj().T, atol=atol), "V not orthogonal?"
    # correctness of recovered svals
    assert allclose(svals[: len(S)], S, atol=atol), "Incorrect svals!"
    # svals nonnegative and by descending magnitude
    assert (S >= 0).all(), "Negative svals!"
    assert (diff(S) <= 0).all(), "Ascending svals?"
    # matching device and type
    assert U.device == mat.device, "Incorrect U device!"
    assert S.device == mat.device, "Incorrect S device!"
    assert Vh.device == mat.device, "Incorrect V device!"
    assert U.dtype == mat.dtype, "Incorrect U dtype!"
    assert S.dtype == mat.real.dtype, "Incorrect S dtype!"
    assert Vh.dtype == mat.dtype, "Incorrect V dtype!"


def qc_test_helper(mat, I, core_rec, q_rec, atol):
    """ """
    allclose = torch.allclose if isinstance(I, torch.Tensor) else np.allclose
    diff = torch.diff if isinstance(I, torch.Tensor) else np.diff
    C, Q, Qh = core_rec, q_rec, q_rec.conj().T
    # correctness of result
    assert allclose(mat, Q @ C @ Qh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered Q
    assert allclose(I, Qh @ Q, atol=atol), "Eigvecs not orthogonal?"
    # simmetry of recovered core
    assert allclose(C, C.conj().T, atol=atol), "Core not hermitian?"
    # matching device and type
    assert Q.device == mat.device, "Incorrect Q device!"
    assert C.device == mat.device, "Incorrect core device!"
    assert Q.dtype == mat.dtype, "Incorrect Q dtype!"
    assert C.dtype == mat.dtype, "Incorrect core dtype!"


def eigh_test_helper(mat, ews, I, ews_rec, evs_rec, atol):
    """Hermitian eigendecomposition test helper.

    Given the produced EIGH of ``mat``, tests that:

    * The EIGH is actually close to the matrix
    * The recovered eigvals are close to true ones
    * Eigenvectors are orthonormal
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(I, torch.Tensor) else np.allclose
    diff = torch.diff if isinstance(I, torch.Tensor) else np.diff
    V, Lbd, Vh = evs_rec, ews_rec, evs_rec.conj().T
    # correctness of result
    assert allclose(mat, (V * Lbd) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered V
    assert allclose(I, Vh @ V, atol=atol), "Eigvecs not orthogonal?"
    # correctness of recovered eigvals
    sorted_Lbd = Lbd[Lbd.argsort()]
    sorted_ews = ews[ews[: len(Lbd)].argsort()]
    assert allclose(sorted_ews, sorted_Lbd, atol=atol), "Incorrect eigvals!"
    # Eigvals by descending magnitude
    assert (diff(abs(Lbd)) <= 0).all(), "Ascending eigval magnitudes?"
    # matching device and type
    assert V.device == mat.device, "Incorrect eigvecs device!"
    assert Lbd.device == mat.device, "Incorrect eigvals device!"
    assert V.dtype == mat.dtype, "Incorrect eigvecs dtype!"
    assert Lbd.dtype == mat.real.dtype, "Incorrect eigvals dtype!"


# ##############################################################################
# # RECOVERY FOR GENERAL MATRICES
# ##############################################################################
def test_recovery_general(
    rng_seeds, torch_devices, dtypes_tols, general_recovery_shapes
):
    """Test case for recovery of general matrices (formal and correctness).

    For torch/numpy inputs, and for all recovery methods (in UV/SVD mode),
    tests that:

    *
    :func:`svd_test_helper`. This tests that provided outputs are correct,
    have the expected properties and are in the matching device/dtype.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for hw, rank, outermeas, innermeas in general_recovery_shapes:
                    # torch: low-rank matrix, svals and identity
                    I1 = torch.eye(outermeas, dtype=dtype, device=device)
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
                    _, S1, _ = svd(mat1)
                    # torch: outer/inner, left/right random measurement linops
                    rlop1 = gaussian_noise(
                        (hw[1], outermeas),
                        seed=seed + 100,
                        device=device,
                        dtype=dtype,
                    )
                    llop1 = gaussian_noise(
                        (outermeas, hw[0]),
                        seed=seed + 101,
                        device=device,
                        dtype=dtype,
                    )
                    rilop1 = gaussian_noise(
                        (hw[1], innermeas),
                        seed=seed + 102,
                        device=device,
                        dtype=dtype,
                    )
                    lilop1 = gaussian_noise(
                        (innermeas, hw[0]),
                        seed=seed + 103,
                        device=device,
                        dtype=dtype,
                    )
                    # torch: outer and core random measurements
                    Y1 = mat1 @ rlop1  # tall
                    Z1 = llop1 @ mat1  # fat
                    C1 = lilop1 @ (mat1 @ rilop1)  # core
                    # numpy: all corresponding objects
                    I2 = I1.cpu().numpy()
                    mat2 = mat1.cpu().numpy()
                    S2 = S1.cpu().numpy()
                    Y2, Z2 = Y1.cpu().numpy(), Z1.cpu().numpy()
                    C2 = C1.cpu().numpy()
                    rlop2 = rlop1.cpu().numpy()
                    rilop2 = rilop1.cpu().numpy()
                    lilop2 = lilop1.cpu().numpy()
                    # LOOP OVER CONFIGS
                    conf1 = (I1, mat1, S1, Y1, Z1, C1, rlop1, rilop1, lilop1)
                    conf2 = (I2, mat2, S2, Y2, Z2, C2, rlop2, rilop2, lilop2)
                    for mode, I, mat, S, Y, Z, C, right, rilop, lilop in (
                        ("torch", *conf1),
                        ("numpy", *conf2),
                    ):
                        # singlepass - UV
                        Urec, Vhrec = singlepass(Y, Z, right, as_svd=False)
                        try:
                            uv_test_helper(mat, Urec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Singlepass-UV {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # singlepass - SVD
                        Urec, Srec, Vhrec = singlepass(
                            Y, Z, right, as_svd=True
                        )
                        try:
                            svd_test_helper(mat, S, I, Urec, Srec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Singlepass-SVD {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # nystrom - UV
                        Urec, Vhrec = nystrom(Y, Z, right, as_svd=False)
                        try:
                            uv_test_helper(mat, Urec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Nystrom-UV {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # nystrom - SVD
                        Urec, Srec, Vhrec = nystrom(Y, Z, right, as_svd=True)
                        try:
                            svd_test_helper(mat, S, I, Urec, Srec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Nystrom-SVD {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # oversampled - UV
                        Urec, Vhrec = oversampled(
                            Y, Z, C, lilop, rilop, as_svd=False
                        )
                        try:
                            uv_test_helper(mat, Urec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Oversampled-UV {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # oversampled - SVD
                        Urec, Srec, Vhrec = oversampled(
                            Y, Z, C, lilop, rilop, as_svd=True
                        )
                        try:
                            svd_test_helper(mat, S, I, Urec, Srec, Vhrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Oversampled-SVD {mode} error!"
                            raise AssertionError(errmsg) from ae


# ##############################################################################
# # RECOVERY FOR HERMITIAN MATRICES
# ##############################################################################
def test_recovery_hermitian(
    rng_seeds, torch_devices, dtypes_tols, hermitian_recovery_shapes
):
    """Test case for recovery of Hermitian matrices (formal and correctness).

    For torch/numpy inputs, and for all recovery methods (in UV/SVD mode),
    tests that:

    *
    :func:`svd_test_helper`. This tests that provided outputs are correct,
    have the expected properties and are in the matching device/dtype.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (
                    dims,
                    rank,
                    outermeas,
                    innermeas,
                ) in hermitian_recovery_shapes:
                    hw = (dims, dims)
                    # torch: low-rank matrix, svals and identity
                    I1 = torch.eye(outermeas, dtype=dtype, device=device)
                    mat1, _ = RandomLordMatrix.exp(
                        hw,
                        rank,
                        decay=100,
                        diag_ratio=0.0,
                        symmetric=True,
                        psd=False,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    ews1, _ = eigh(mat1)
                    # torch: outer/inner, left/right random measurement linops
                    rlop1 = gaussian_noise(
                        (hw[1], outermeas),
                        seed=seed + 100,
                        device=device,
                        dtype=dtype,
                    )
                    rilop1 = gaussian_noise(
                        (hw[1], innermeas),
                        seed=seed + 102,
                        device=device,
                        dtype=dtype,
                    )
                    lilop1 = gaussian_noise(
                        (innermeas, hw[0]),
                        seed=seed + 103,
                        device=device,
                        dtype=dtype,
                    )
                    # torch: outer and core random measurements
                    Y1 = mat1 @ rlop1  # tall
                    C1 = lilop1 @ (mat1 @ rilop1)  # core
                    # numpy: all corresponding objects
                    I2 = I1.cpu().numpy()
                    mat2 = mat1.cpu().numpy()
                    ews2 = ews1.cpu().numpy()
                    Y2, C2 = Y1.cpu().numpy(), C1.cpu().numpy()
                    rlop2 = rlop1.cpu().numpy()
                    rilop2 = rilop1.cpu().numpy()
                    lilop2 = lilop1.cpu().numpy()
                    # LOOP OVER CONFIGS
                    conf1 = (I1, mat1, ews1, Y1, C1, rlop1, rilop1, lilop1)
                    conf2 = (I2, mat2, ews2, Y2, C2, rlop2, rilop2, lilop2)
                    for mode, I, mat, ews, Y, C, right, rilop, lilop in (
                        ("torch", *conf1),
                        ("numpy", *conf2),
                    ):
                        # singlepass_h - QCQh
                        Crec, Qrec = singlepass_h(Y, right, as_eigh=False)
                        try:
                            qc_test_helper(mat, I, Crec, Qrec, tol)
                        except AssertionError as ae:
                            errmsg = f"Singlepass_h-QCQh {mode} error!"
                            raise AssertionError(errmsg) from ae
                        # singlepass_h - EIGH
                        ews_rec, evs_rec = singlepass_h(Y, right, as_eigh=True)
                        try:
                            eigh_test_helper(
                                mat, ews, I, ews_rec, evs_rec, tol
                            )
                        except AssertionError as ae:
                            errmsg = f"Singlepass_h-EIGH {mode} error!"
                            raise AssertionError(errmsg) from ae
