#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for synthetic (random) matrices.

This module implements a few families of synthetic random matrices, inspired by
`[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_, but also expanded:

* Low rank + noise: dampened Gaussian noise matrix in the form ``G @ G.T``,
  plus a (low-rank) subset of the identity matrix.
* Exponential: Singular values have a low-rank unit section followed by
  exponential decay.
* Polynomial: Singular values have a low-rank unit section followed by
  polynomial decay.

In `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_, the exponential and
polynomial matrices are diagonal. here, we extend them as follows:

* Random orthogonal matrices for the left and right singular spaces
* Optionally, the matrix is square and symmetric so left and right vectors are
  adjoint
* Optionally, the decaying spectrum is multiplied by Rademacher noise, to test
  also with non-PSD matrices
"""


import torch

from .utils import gaussian_noise, rademacher


# ##############################################################################
# # SYNTH MATRIX FACTORY
# ##############################################################################
class SynthMat:
    """Static class to produce different families of synthetic matrices."""

    @staticmethod
    def lowrank_noise(
        shape=(100, 100),
        rank=10,
        snr=1e-4,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """Low rank + noise random matrix.

        Produces a lowrank+noise matrix, as follows: First, A Gaussian IID noise
        matrix ``G`` is sampled for the provided (square) shape. Then, the
        noisy matrix ``(snr / dims) * G@G.T`` is computed. Finally, the first
        ``rank`` diagonal entries are incremented by 1.
        As a result, we have a symmetric noisy matrix with strong diagonal
        for the first ``rank`` entries, and with a signal-to-noise ratio of
        ``snr``.

        :param int rank: How many diagonal entries should be reinforced.
        :param float snr: Signal-to-noise ratio. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 1e-4 for low noise, 1e-2 for mid noise, and 1e-1 for
          high noise.
        :returns: A matrix of given shape and properties.
        """
        h, w = shape
        assert h == w, "lowrank_noise must be square! (and symmetric)"
        # create matrix as a scaled outer product of Gaussian noise
        result = gaussian_noise(
            shape,
            mean=0.0,
            std=1.0,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        result = (snr / shape[0]) * (result @ result.T)
        # add 1 to the first "rank" diagonal entries
        result[range(rank), range(rank)] += 1
        return result

    @staticmethod
    def _decay_helper(
        svals,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """Given singular values, produces a random matrix.

        Helper method for the polynomial and exp-decay matrices. Given the
        singular values, it samples the left- and right- singular spaces,
        and returns the final matrix.
        """
        min_shape = min(shape)
        # build singular bases using QR subgroup algorithm (Diaconis). QR is not
        # fastest, but these are test matrices so speed is not crucial.
        G_left = gaussian_noise(
            (shape[0], min_shape),
            mean=0.0,
            std=1.0,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        U, _ = torch.linalg.qr(G_left)
        del G_left
        #
        if symmetric:
            result = U @ torch.diag(svals) @ U.T
        else:
            G_right = gaussian_noise(
                (shape[1], min_shape),
                mean=0.0,
                std=1.0,
                seed=seed + 1,
                dtype=dtype,
                device=device,
            )
            V, _ = torch.linalg.qr(G_right)
            result = U @ torch.diag(svals) @ V.T
        #
        return result

    @classmethod
    def poly_decay(
        cls,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
        psd=True,
    ):
        r"""Random matrix with polynomial singular value decay.

        Produces a matrix in the form ``U @ S @ V.T``, where ``U`` and ``V`` are
        random orthogonal matrices, and ``S`` has entries with polynomially
        decaying magnitude, analogous to the ones described in
        `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_.

        :param int rank: Entries in ``S[:rank]`` will have a magnitude of 1.
        :param float decay: Parameter determining how quickly magnitudes in
          ``S[rank:]`` decay to zero, following :math:`2^d, 3^d, 4^d, \dots`
          for decay :math:`d`. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 0.5 for slow decay, 1 for medium, 2 for fast.
        :param bool symmetric: If true, ``V == U``.
        :param psd: If true, ``S`` will be multiplied with Rademacher noise
          to create a non-PSD matrix (assuming symmetric).
        """
        min_shape = min(shape)
        # a few ones, followed by a poly decay
        svals = torch.zeros(min_shape, dtype=dtype).to(device)
        svals[:rank] = 1
        svals[rank:] = torch.arange(2, min_shape - rank + 2) ** (-float(decay))
        if not psd:
            rademacher(svals[rank:], seed=seed + 1, inplace=True)
        #
        result = cls._decay_helper(
            svals, shape, rank, decay, symmetric, seed, dtype, device
        )
        return result

    @classmethod
    def exp_decay(
        cls,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
        psd=True,
    ):
        r"""Random matrix with exponential singular value decay.

        Produces a matrix in the form ``U @ S @ V``, where ``U`` and ``V`` are
        random orthogonal matrices, and ``S`` has entries with exponentially
        decaying magnitude, analogous to the ones described in
        `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_.

        :param int rank: Entries in ``S[:rank]`` will have a magnitude of 1.
        :param float decay: Parameter determining how quickly magnitudes in
          ``S[rank:]`` decay to zero, following :math:`10^{-d}, 10^{-2d}, \dots`
          for decay :math:`d`. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 0.01 for slow decay, 0.1 for medium, 0.5 for fast.
        :param bool symmetric: If true, ``V == U``.
        :param psd: If true, ``S`` will be multiplied with Rademacher noise
          to create a non-PSD matrix (assuming symmetric).
        """
        min_shape = min(shape)
        # a few ones, followed by exp decay
        svals = torch.zeros(min_shape, dtype=dtype).to(device)
        svals[:rank] = 1
        svals[rank:] = 10 ** -(decay * torch.arange(1, min_shape - rank + 1))
        if not psd:
            rademacher(svals[rank:], seed=seed + 1, inplace=True)
        #
        result = cls._decay_helper(
            svals, shape, rank, decay, symmetric, seed, dtype, device
        )
        return result
