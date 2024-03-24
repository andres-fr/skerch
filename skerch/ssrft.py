#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Scrambled Subsampled Randomized Fourier Transform (SSRFT) utilities.

PyTorch-compatible implementation of the SSRFT, from
`[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_.
"""


import torch
import torch_dct as dct

from .linops import BaseRandomLinOp
from .utils import BadShapeError, NoFlatError, rademacher, randperm


# ##############################################################################
# # SSRFT
# ##############################################################################
def ssrft(x, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    r"""Right (forward) matrix multiplication of the SSRFT.

    This function implements a matrix-free, right-matmul operator of the
    Scrambled Subsampled Randomized Fourier Transform (SSRFT) for real-valued
    signals, from `[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_.

    .. math::

      \text{SSRFT} = R\,\mathcal{F}\,\Pi\,\mathcal{F}\,\Pi'

    Where :math:`R` is a random index-picker, \mathcal{F} is a Discrete Cosine
    Transform, and :math:`\Pi, \Pi'` are random permutations.

    :param x: Vector to be projected, such that ``y = SSRFT @ x``
    :param out_dims: Dimensions of output ``y``, must be less than ``dim(x)``
    :param seed: Random seed
    """
    # make sure all sources of randomness are CPU, to ensure cross-device
    # consistency of the operator
    if len(x.shape) != 1:
        raise NoFlatError("Only flat tensors supported!")
    x_len = len(x)
    assert out_dims <= x_len, "Projection to larger dimensions not supported!"
    seeds = [seed + i for i in range(5)]
    # first scramble: permute, rademacher, and DCT
    perm1 = randperm(x_len, seed=seeds[0], device="cpu")
    x, rad1 = rademacher(x[perm1], seed=seeds[1], inplace=False)
    del perm1, rad1
    x = dct.dct(x, norm=dct_norm)
    # second scramble: permute, rademacher and DCT
    perm2 = randperm(x_len, seed=seeds[2], device="cpu")
    x, rad2 = rademacher(x[perm2], seeds[3], inplace=False)
    del perm2, rad2
    x = dct.dct(x, norm=dct_norm)
    # extract random indices and return
    out_idxs = randperm(x_len, seed=seeds[4], device="cpu")[:out_dims]
    x = x[out_idxs]
    return x


def ssrft_adjoint(x, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    r"""Left (adjoint) matrix multiplication of the SSRFT.

    Adjoint operator of SSRFT, such that ``x @ SSRFT = y``. See :func:`.ssrft`
    for more details. Note the following implementation detail:

    * Permutations are orthogonal transforms
    * Rademacher transforms are also orthogonal (also diagonal and self-inverse)
    * DCT/DFT are also orthogonal transforms
    * The index-picker :math:`R` is a subset of rows of I.

    With orthogonal operators, transform and inverse are the same. Therefore,
    this adjoint operator takes the following form:

    .. math::

       \text{SSRFT}^T =& (R\,\mathcal{F}\,\Pi\,\mathcal{F}\,\Pi')^T \\
       =& \Pi'^T \, \mathcal{F}^T \, \Pi^T \, \mathcal{F}^T \, R^T \\
       =& \Pi'^{-1} \, \mathcal{F}^{-1} \, \Pi^{-1} \, \mathcal{F}^{-1} \, R^T

    So we can make use of the inverses, except for :math:`R^T`, which is a
    column-truncated identity, so we embed the entries picked by :math:`R` into
    the corresponding indices, and leave the rest as zeros.
    """
    # make sure all sources of randomness are CPU, to ensure cross-device
    # consistency of the operator
    if len(x.shape) != 1:
        raise NoFlatError("Only flat tensors supported!")
    x_len = len(x)
    assert (
        out_dims >= x_len
    ), "Backprojection into smaller dimensions not supported!"
    #
    seeds = [seed + i for i in range(5)]
    result = torch.zeros(
        out_dims,
        dtype=x.dtype,
    ).to(x.device)
    # first embed signal into original indices
    out_idxs = randperm(out_dims, seed=seeds[4], device="cpu")[:x_len]
    result[out_idxs] = x
    del x
    # then do the idct, followed by rademacher and inverse permutation
    result = dct.idct(result, norm=dct_norm)
    rademacher(result, seeds[3], inplace=True)
    perm2_inv = randperm(out_dims, seed=seeds[2], device="cpu", inverse=True)
    result = result[perm2_inv]
    del perm2_inv
    # second inverse pass
    result = dct.idct(result, norm=dct_norm)
    rademacher(result, seeds[1], inplace=True)
    perm1_inv = randperm(out_dims, seed=seeds[0], device="cpu", inverse=True)
    result = result[perm1_inv]
    #
    return result


class SSRFT(BaseRandomLinOp):
    """Scrambled Subsampled Randomized Fourier Transform (SSRFT).

    This class encapsulates the left- and right-SSRFT transforms into a single
    linear operator, which is deterministic for the same shape and seed
    (particularly, also across different torch devices).
    """

    def __init__(self, shape, seed=0b1110101001010101011):
        """:param shape: ``(height, width)`` of this linear operator."""
        super().__init__(shape, seed)
        h, w = shape
        if h > w:
            raise BadShapeError("Height > width not supported!")
        # :param scale: Ideally, ``1/l``, where ``l`` is the average diagonal
        #   value of the covmat ``A.T @ A``, where ``A`` is a FastJLT operator,
        #   so that ``l2norm(x)`` approximates ``l2norm(Ax)``.
        self.scale = NotImplemented

    def matmul(self, x):
        """Forward (right) matrix-vector multiplication ``SSRFT @ x``.

        See parent class for more details.
        """
        return ssrft(x, self.shape[0], seed=self.seed, dct_norm="ortho")

    def rmatmul(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ SSRFT``.

        See parent class for more details.
        """
        return ssrft_adjoint(x, self.shape[1], seed=self.seed, dct_norm="ortho")
