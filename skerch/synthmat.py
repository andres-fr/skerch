#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for synthetic (random) matrices.

This module implements a few families of synthetic matrices, inspired by
`[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_, but also expanded:

* Low rank + noise: dampened Gaussian noise matrix in the form ``G @ G.H``,
  plus a (low-rank) subset of the identity matrix.
* Exponential: Singular values have a low-rank unit section followed by
  exponential decay.
* Polynomial: Singular values have a low-rank unit section followed by
  polynomial decay.
* Low-rank + diagonal: Optionally, a random diagonal can be added.

In `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_, the exponential and
polynomial matrices are symmetric. here, we extend them as follows:

* Random orthogonal matrices for the left and right singular spaces
* Optionally, the matrix is square and symmetric so left and right vectors are
  adjoint
* Optionally, the decaying spectrum is multiplied by Rademacher noise, to test
  also with non-PSD matrices in the Symmetric case.
"""


import torch

from .utils import gaussian_noise, rademacher_flip, complex_dtype_to_real


# ##############################################################################
# # APPROXIMATELY LOW-RANK MATRICES
# ##############################################################################
class RandomLordMatrix:
    """Static class to produce different random lowrank + diagonal matrices."""

    @staticmethod
    def mix_matrix_and_diag(
        mat, diag, diag_ratio=1.0, inplace=True, diag_dtype=torch.float64
    ):
        """
        :param diag_ratio: The ratio ``diag.norm() / v_norm``, where ``v_norm``
          is a mean-vector norm of the matrix across the smaller dimension,
          i.e. if the matrix is tall, the mean is computed among rows, and
          otherwise among columns. Since the diagonal has as many entries as
          these vectors, a ratio of 1.0 means that the diagonal is as large
          as the average corresponding vector from the matrix.
        """
        if not inplace:
            mat = mat.clone()
            diag = diag.clone()
        #
        if diag_ratio < 0:
            raise ValueError("diag_ratio must be >= 0")
        elif diag_ratio == 0:
            ratio = 0
            diag *= ratio
        else:
            h, w = mat.shape
            diag_dim = len(diag)
            v_norm = mat.norm() / (max(h, w) ** 0.5)
            if v_norm == 0:
                raise ValueError("Low-rank matrix cannot be zero!")
            d_norm = diag.norm()
            # if we now multiply diag by (v_norm / d_norm), it will have same
            # norm as the average mat vector. Further multiply by desired ratio
            ratio = diag_ratio * (v_norm / d_norm)
            diag *= ratio
            mat[range(diag_dim), range(diag_dim)] += diag
        #
        return mat, diag, ratio

    @classmethod
    def noise(
        cls,
        shape=(100, 100),
        rank=10,
        snr=1e-4,
        diag_ratio=0.0,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """Low rank + symmetric noise + real noisy diagonal.

        If diag_ratio is 0, produces a lowrank+noise matrix, as follows: First,
        A Gaussian IID noise matrix ``G`` is sampled for the provided (square)
        shape. Then, the noisy matrix ``(snr / dims) * G@G.H`` is computed.
        Finally, the first ``rank`` diagonal entries are incremented by 1.
        As a result, we have a symmetric (resp. Hermitian) noisy matrix with
        strong diagonal for the first ``rank`` entries, and with a
        signal-to-noise ratio of ``snr``.

        If ``diag_ratio != 0``, a real-valued Gaussian IID diagonal is added
        via :meth:`mix_matrix_and_diag`.

        :param int rank: How many diagonal entries should be reinforced.
        :param float snr: Signal-to-noise ratio for the symmetric noise. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 1e-4 for low noise, 1e-2 for mid noise, and 1e-1 for
          high noise.
        :returns: The pair ``(mat, diag)`` where ``mat = lowrank+noise+diag``.
        """
        h, w = shape
        if h != w:
            raise ValueError("noise matrix must be square! (and symmetric)")
        if rank <= 0:
            raise ValueError("Rank must be positive!")
        # create matrix as a scaled outer product of Gaussian noise
        result = gaussian_noise(
            shape,
            mean=0.0,
            std=1.0,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        result = (snr / shape[0]) * (result @ result.H)
        # add 1 to the first "rank" diagonal entries
        result[range(rank), range(rank)] += 1
        #
        diag_dim = min(h, w)
        diag_dtype = complex_dtype_to_real(dtype)
        diag = gaussian_noise(
            diag_dim,
            mean=0,
            seed=seed - 1234,
            dtype=diag_dtype,
            device=device,
        )
        #
        cls.mix_matrix_and_diag(result, diag, diag_ratio, inplace=True)
        return result, diag

    @staticmethod
    def _decay_helper(
        svals,
        shape=(100, 100),
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
        psd=True,
    ):
        """Given singular values, produces a random matrix.

        Helper method for the polynomial and exp-decay matrices. Given the
        singular values, it samples the left- and right- singular spaces,
        and returns the final matrix.
        """
        # symmetric must be square
        h, w = shape
        if (h != w) and symmetric:
            raise ValueError("Symmetric matrices must be square!")
        # check that svals are nonnegative real
        if svals.dtype != complex_dtype_to_real(svals.dtype):
            raise ValueError("Singular/eigenvalues must be real!")
        try:
            if psd and (svals < 0).any():
                raise ValueError("Negative eigenvalues not allowed for PSD!")
        except Exception as e:
            raise ValueError("Invalid eigenvalues!") from e
        #
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
            result = (U * svals) @ U.H
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
            result = (U * svals) @ V.H
        #
        return result

    @staticmethod
    def get_decay_svals(dims, rank, decay_type, decay, dtype, device):
        """ """
        svals = torch.zeros(dims, dtype=dtype, device=device)
        svals[:rank] = 1
        if decay_type == "poly":
            svals[rank:] = torch.arange(2, dims - rank + 2) ** (-float(decay))
        elif decay_type == "exp":
            svals[rank:] = 10 ** -(decay * torch.arange(1, dims - rank + 1))
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")
        return svals

    @classmethod
    def poly(
        cls,
        shape=(100, 100),
        rank=10,
        decay=2.0,
        diag_ratio=0.0,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
        psd=True,
    ):
        r"""Random matrix with polynomial singular value decay plus diagonal.

        Produces a matrix in the form ``U @ S @ V.H + D``, where ``U`` and ``V``
        are random orthogonal matrices, ``S`` has entries with polynomially
        decaying magnitude, analogous to the ones described in
        `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_, and ``D``
        has random Gaussian IID entries with intensity given by ``diag_ratio``.

        :param int rank: Entries in ``S[:rank]`` will have a magnitude of 1.
        :param float decay: Parameter determining how quickly magnitudes in
          ``S[rank:]`` decay to zero, following :math:`2^d, 3^d, 4^d, \dots`
          for decay :math:`d`. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 0.5 for slow decay, 1 for medium, 2 for fast.
        :param diag_ratio: Intensity of ``D``. See :meth:`mix_matrix_and_diag`.
        :param bool symmetric: If true, ``V == U``.
        :param psd: If false, and matrix is symmetric, the singular values will
          be multiplied with Rademacher noise to create a non-PSD matrix.
        """
        if rank <= 0:
            raise ValueError("Rank must be positive!")
        h, w = shape
        svals_dtype = complex_dtype_to_real(dtype)
        min_shape = min(shape)
        # a few ones, followed by a poly decay
        svals = cls.get_decay_svals(
            min_shape, rank, "poly", decay, svals_dtype, device
        )
        i = 1
        if (not psd) and symmetric:
            # ensure at least one val is negative
            while (svals >= 0).all():
                rademacher_flip(svals, seed=seed + i, inplace=True)
                i += 1
        #
        result = cls._decay_helper(
            svals, shape, decay, symmetric, seed, dtype, device, psd
        )
        #
        diag_dim = min(h, w)
        diag = gaussian_noise(
            diag_dim,
            mean=0,
            seed=seed - 1234,
            dtype=dtype,
            device=device,
        )
        #
        cls.mix_matrix_and_diag(result, diag, diag_ratio, inplace=True)
        return result, diag

    @classmethod
    def exp(
        cls,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        diag_ratio=0.0,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
        psd=True,
    ):
        r"""Random matrix with exponential singular value decay plus diagonal.

        Like :meth:`poly`, but the SV decay of the low-rank component follows
        :math:`10^{-d}, 10^{-2d}, \dots` for decay :math:`d`. In
          `[TYUC2019, 7.3.1] <https://arxiv.org/abs/1902.08651>`_ the following
          values are used: 0.01 for slow decay, 0.1 for medium, 0.5 for fast.
        """
        if rank <= 0:
            raise ValueError("Rank must be positive!")
        h, w = shape
        svals_dtype = complex_dtype_to_real(dtype)
        min_shape = min(shape)
        # a few ones, followed by exp decay
        svals = cls.get_decay_svals(
            min_shape, rank, "exp", decay, svals_dtype, device
        )
        i = 1
        if (not psd) and symmetric:
            # ensure at least one val is negative
            while (svals >= 0).all():
                rademacher_flip(svals, seed=seed + i, inplace=True)
                i += 1
        #
        result = cls._decay_helper(
            svals, shape, decay, symmetric, seed, dtype, device, psd
        )
        #
        diag_dim = min(h, w)
        diag = gaussian_noise(
            diag_dim,
            mean=0,
            seed=seed - 1234,
            dtype=dtype,
            device=device,
        )
        #
        cls.mix_matrix_and_diag(result, diag, diag_ratio, inplace=True)
        return result, diag
