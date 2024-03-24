#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""General utilities for the ``skerch`` library."""


import torch


# ##############################################################################
# # STR
# ##############################################################################
def torch_dtype_as_str(dtype):
    """Torch dtype to string.

    Given a PyTorch datatype object, like ``torch.float32``, returns the
    corresponding string, in this case 'float32'.
    """
    full_str = str(dtype)
    dot_idx = full_str.index(".")
    result = full_str[dot_idx + 1 :]
    return result


# ##############################################################################
# # ERRORS
# ##############################################################################
class NoFlatError(Exception):
    """Error to be thrown when a tensor is not flat (i.e. not a vector)."""

    pass


class BadShapeError(Exception):
    """Error to be thrown when a shape is not as it should."""

    pass


# ##############################################################################
# # REPRODUCIBLE NOISE
# ##############################################################################
def uniform_noise(shape, seed=None, dtype=torch.float64, device="cpu"):
    """Reproducible ``torch.rand`` uniform noise.

    :returns: A tensor of given shape, dtype and device, containing uniform
      random noise between 0 and 1 (analogous to ``torch.rand``), but with
      reproducible behaviour fixed to given random seed.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    noise = torch.rand(shape, generator=rng, dtype=dtype, device=device)
    return noise


def gaussian_noise(
    shape, mean=0.0, std=1.0, seed=None, dtype=torch.float64, device="cpu"
):
    """Reproducible ``torch.normal`` Gaussian noise.

    :returns: A tensor of given shape, dtype and device, containing gaussian
      noise with given mean and std (analogous to ``torch.normal``), but with
      reproducible behaviour fixed to given random seed.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    #
    noise = torch.zeros(shape, dtype=dtype, device=device)
    noise.normal_(mean=mean, std=std, generator=rng)
    return noise


def randperm(n, seed=None, device="cpu", inverse=False):
    """Reproducible randperm of ``n`` integers from  0 to (n-1) (both included).

    :param bool inverse: If False, a random permutation ``P`` is provided. If
      true, an inverse permutation ``Q`` is provided, such that both
      permutations are inverse to each other, i.e. ``v == v[P][Q] == v[Q][P]``.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    #
    perm = torch.randperm(n, generator=rng, device=device)
    if inverse:
        # we take the O(N) approach since we anticipate large N
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        perm = inv
    return perm


def rademacher(x, seed=None, inplace=True, rng_device="cpu"):
    """Reproducible sign-flipping via Rademacher noise.

    .. note::
      This function makes use of :func:`uniform_noise` to sample the Rademacher
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.

    .. warning::
      PyTorch does not ensure RNG reproducibility across
      devices. This parameter determines the device to generate the noise from.
      If you want cross-device reproducibility, make sure that the noise is
      always generated from the same device.

    :param rng_device: While result will be returned on the same device as
      ``x``, the Rademacher noise used to flip entries will be sampled on
      this device (see reproducibility note).
    """
    mask = (
        uniform_noise(
            x.shape, seed=seed, dtype=torch.float32, device=rng_device
        )
        > 0.5
    ).to(x.device) * 2 - 1
    if inplace:
        x *= mask
        return x, mask
    else:
        return x * mask, mask
