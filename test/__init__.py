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


# ##############################################################################
# # HELPERS
# ##############################################################################
def autocorrelation_test_helper(vec, delta_at_least=0.8, nondelta_at_most=0.1):
    """If vec is iid noise, its autocorrelation resembles a delta."""
    # normalize and compute unit-norm autocorrelation
    vec_mean = vec.mean()
    vec_norm = (vec - vec_mean).norm()
    vec = (vec - vec_mean) / vec_norm
    autocorr = torch.fft.fft(vec, norm="ortho")
    autocorr = (autocorr * autocorr.conj()) ** 0.5
    autocorr = abs(torch.fft.ifft(autocorr, norm="ortho"))
    # check that autocorrelation is close to standard unit delta
    assert abs(autocorr.norm() - 1) < 1e-5, "Autocorr should have unit norm"
    assert (
        autocorr[0] >= delta_at_least
    ), "Noise autocorr does not have a strong delta"
    assert (
        autocorr[1:] <= nondelta_at_most
    ).all(), "Noise autocorr has strong non-delta!"
