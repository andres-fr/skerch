#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""General ``pytest`` utilities, like fixtures.

.. note::

  See `docs <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_.
"""


import pytest
import torch


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def torch_devices():
    """Available PyTorch devices for the unit tests."""
    result = ["cpu"]
    if torch.cuda.is_available():
        result.append("cuda")
    return result


@pytest.fixture
def rng_seeds(request):
    """Random seeds for the unit tests.

    .. note::
     This implicitly uses code from :mod:`.conftest`.
    """
    result = request.config.getoption("--seeds")
    return result


@pytest.fixture
def snr_lowrank_noise():
    """SNR values for Lowrank+noise matrix. The larger, the more noise.

    .. note::
      See `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_
    """
    result = [1e-3, 1e-2, 1e-1, 1]
    return result


@pytest.fixture
def exp_decay():
    """Exponential decay values. The larger, the faster decay.

    .. note::
      See `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_
    """
    result = [0.5, 0.1, 0.01]
    return result


@pytest.fixture
def poly_decay():
    """Polynomial decay values. The larger, the faster decay.

    .. note::
      See `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_
    """
    result = [2, 1, 0.5]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
def linop_to_matrix(lop, dtype=torch.float32, device="cpu", adjoint=False):
    """Convert a linop to a matrix via one-hot matrix-vector products."""
    h, w = lop.shape
    result = torch.zeros(lop.shape, dtype=dtype, device=device)
    if adjoint:
        oh = torch.zeros(h, dtype=dtype, device=device)
        for i in range(h):
            oh *= 0
            oh[i] = 1
            result[i, :] = oh @ lop
    else:
        oh = torch.zeros(w, dtype=dtype, device=device)
        for i in range(w):
            oh *= 0
            oh[i] = 1
            result[:, i] = lop @ oh
    #
    return result
