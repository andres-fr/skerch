#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for QR decomposition."""


import pytest
import torch

from skerch.utils import gaussian_noise

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def heights_widths():
    """Shapes of matrices to QR decompose."""
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (1_000, 1_000),
    ]
    result += [
        (10, 1),
        (100, 10),
        (1_000, 100),
        (10_000, 1_000),
    ]
    return result


@pytest.fixture
def entry_atol():
    """Absolute, entry-wise error tolerance for ``diff(Q.T @ Q, I)``."""
    result = 1e-5
    return result


# ##############################################################################
# #
# ##############################################################################
def test_orth_q(rng_seeds, torch_devices, heights_widths, entry_atol):
    """Test case for orthogonality of Q in QR decomposition.

    Tests that ``Q.T @ Q`` is close to identity.
    """
    for h, w in heights_widths:
        assert h >= w, "This test doesn't need/admit fat matrices!"
    #
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype in (torch.float64, torch.float32):
                for h, w in heights_widths:
                    mat = gaussian_noise(
                        (h, w),
                        mean=0.0,
                        std=1.0,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    Q = torch.linalg.qr(mat)[0]
                    I_residual = Q.T @ Q
                    I_residual[range(w), range(w)] -= 1
                    worst = I_residual.abs().max().item()
                    assert worst <= abs(entry_atol), "Q matrix not orthogonal?"
