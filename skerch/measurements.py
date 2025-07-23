#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""

from collections import defaultdict
import warnings
import torch
import torch_dct as dct
from .linops import ByVectorLinOp
from .utils import (
    COMPLEX_DTYPES,
    BadShapeError,
    BadSeedError,
    rademacher_noise,
    phase_noise,
    rademacher_flip,
    phase_shift,
    gaussian_noise,
    phase_noise,
    randperm,
)


# ##############################################################################
# # HELPERS
# ##############################################################################
def perform_measurement(lop, meas_lop, adjoint=False, parallel_mode=None):
    """
    :param lop: must satisfy shape and @
    :param meas_lop: must satisfy shape, get_vector and also dtype
    """
    h1, w1 = lop.shape
    h2, w2 = meas_lop.shape
    dtype = meas_lop.dtype
    if (adjoint and w2 != h1) or ((not adjoint) and w1 != h2):
        raise ValueError(
            f"Incompatible shapes! {lop.shape}, {meas_lop.shape}, "
            + f"adjoint={adjoint}"
        )
    #
    if parallel_mode is None:
        warnings.warn("measurements can be parallelized", RuntimeWarning)
        breakpoint()
    #
    elif parallel_mode == "mp":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown parallel_mode! {parallel_mode}")


# ##############################################################################
# # IID NOISE LINOPS
# ##############################################################################
class RademacherNoiseLinOp(ByVectorLinOp):
    """Random linear operator with i.i.d. Rademacher entries.

    :param shape: Shape of the linop as ``(h, w)``.
    :param seed: Random seed used in ``get_vector`` to deterministically sample
      random vectors. Each vector with ``idx`` is sampled from ``seed + idx``,
      for this reason is important that two different linops are instantiated
      with sufficiently distant seeds, to prevent overlaps.
    :param dtype: Dtype of the generated noise.
    :param by_row: See :class:`ByVectorLinOp`.
    :param register: If true, when the linop is created, its seed range is
      added to a global register, which checks if there are any overlapping
      ranges, in which case a ``BadSeedError`` is triggered. If false, this
      behaviour is disabled.
    """

    REGISTER = defaultdict(list)

    @classmethod
    def check_register(cls):
        """Checks if two different-seeded linops have overlapping seeds."""
        for reg_type, reg in cls.REGISTER.items():
            sorted_reg = sorted(reg, key=lambda x: x[0])
            for (beg1, end1), (beg2, end2) in zip(
                sorted_reg[:-1], sorted_reg[1:]
            ):
                if end1 >= beg2:
                    clsname = cls.__name__
                    msg = (
                        f"Overlapping seeds when creating {clsname}! "
                        f"({reg_type}, {sorted_reg}). This is not necessarily "
                        "an issue, but may lead to different-seeded random "
                        "linops generating the same rows or columns. To "
                        "prevent this, ensure that the random seeds of "
                        "different noise linops are separated by more than the "
                        "number of rows/columns. To disable this behaviour, "
                        "initialize with register=False."
                    )
                    raise BadSeedError(msg)

    def __init__(self, shape, seed, dtype, by_row=False, register=True):
        """Initializer. See class docstring."""
        super().__init__(shape, by_row)
        self.seed = seed
        self.dtype = dtype
        #
        if register:
            seed_beg = self.seed
            seed_end = seed_beg + (
                self.shape[0] if self.by_row else self.shape[1]
            )
            self.__class__.REGISTER["default"].append((seed_beg, seed_end))
            self.check_register()

    def get_vector(self, idx, device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        dims = w if self.by_row else h
        if idx < 0 or idx >= (h if self.by_row else w):
            raise ValueError(f"Invalid index {idx} for shape {self.shape}!")
        result = (
            rademacher_noise(  # device always CPU to ensure determinism
                dims, seed=self.seed + idx, device="cpu"
            )
            .to(self.dtype)
            .to(device)
        )
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"seed={self.seed}, dtype={self.dtype}, by_row={self.by_row})>"
        )
        return s


class GaussianNoiseLinOp(RademacherNoiseLinOp):
    """Random linear operator with i.i.d. Gaussian entries.

    See superclass docstring for more details.
    """

    REGISTER = defaultdict(list)

    def __init__(
        self,
        shape,
        seed,
        dtype,
        by_row=False,
        register=True,
        mean=0.0,
        std=1.0,
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, seed, dtype, by_row, register)
        self.mean = mean
        self.std = std

    def get_vector(self, idx, device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        dims = w if self.by_row else h
        if idx < 0 or idx >= (h if self.by_row else w):
            raise ValueError(f"Invalid index {idx} for shape {self.shape}!")
        result = gaussian_noise(  # device always CPU to ensure determinism
            dims,
            self.mean,
            self.std,
            seed=self.seed + idx,
            dtype=self.dtype,
            device="cpu",
        ).to(device)
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"mean={self.mean}, std={self.std}, "
            + f"seed={self.seed}, dtype={self.dtype}, by_row={self.by_row})>"
        )
        return s


class PhaseNoiseLinOp(RademacherNoiseLinOp):
    """Random linear operator with i.i.d. complex entries in the unit circle.

    :param conj: For the same seed, the linear operators with true and false
      ``conj`` values are complex conjugates of each other.

    See superclass docstring for more details.
    """

    REGISTER = defaultdict(list)

    def __init__(
        self, shape, seed, dtype, by_row=False, register=True, conj=False
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, seed, dtype, by_row, register)
        if dtype not in COMPLEX_DTYPES:
            raise ValueError(f"Dtype must be complex! was {dtype}")
        self.conj = conj

    def get_vector(self, idx, device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        dims = w if self.by_row else h
        if idx < 0 or idx >= (h if self.by_row else w):
            raise ValueError(f"Invalid index {idx} for shape {self.shape}!")
        result = phase_noise(
            dims, self.seed + idx, self.dtype, device="cpu"
        ).to(device)
        if self.conj:
            result = result.conj()
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"conj={self.conj}, "
            + f"seed={self.seed}, dtype={self.dtype}, by_row={self.by_row})>"
        )
        return s


# ##############################################################################
# # SSRFT
# ##############################################################################
class SSRFT:
    @staticmethod
    def ssrft(x, out_dims, seed=0b1110101001010101011, norm="ortho"):
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

        if len(x.shape) != 1 or x.numel() <= 0:
            raise BadShapeError(f"Input must be a nonempty vector! {x.shape}")
        x_len = len(x)
        if out_dims > x_len:
            raise ValueError("out_dims can't be larger than last dimension!")
        # make sure all sources of randomness are CPU, to ensure cross-device
        # consistency of the operator
        seeds = [seed + i for i in range(5)]
        if x.dtype in COMPLEX_DTYPES:
            xx = x.clone()  #################################################
            # first scramble: permute, phase noise, and FFT
            x = x[randperm(x_len, seed=seeds[0], device="cpu")]
            phase_shift(
                x, seed=seeds[1], inplace=True, rng_device="cpu", conj=False
            )
            x = torch.fft.fft(x, norm=norm)
            # second scramble: permute, phase noise, and FFT
            x = x[randperm(x_len, seed=seeds[2], device="cpu")]
            phase_shift(
                x, seed=seeds[3], inplace=True, rng_device="cpu", conj=False
            )
            x = torch.fft.fft(x, norm=norm)

        else:
            # first scramble: permute, rademacher, and DCT
            x = x[randperm(x_len, seed=seeds[0], device="cpu")]
            rademacher_flip(x, seed=seeds[1], inplace=True, rng_device="cpu")
            x = dct.dct(x, norm=norm)
            # second scramble: permute, rademacher and DCT
            x = x[randperm(x_len, seed=seeds[2], device="cpu")]
            rademacher_flip(x, seed=seeds[3], inplace=True, rng_device="cpu")
            x = dct.dct(x, norm=norm)

            # """
            # INVERSE
            # """
            # # create output and embed random indices
            # y = torch.zeros(x_len, dtype=x.dtype, device=x.device)
            # y[randperm(out_dims, seed=seeds[4], device="cpu")[:x_len]] = x
            # # invert second scramble: iDCT, rademacher, and inverse permutation
            # y = dct.idct(y, norm=norm)
            # rademacher_flip(y, seed=seeds[3], inplace=True)
            # y = y[randperm(x_len, seed=seeds[2], device="cpu", inverse=True)]
            # # invert first scramble: iDCT, rademacher, and inverse permutation
            # y = dct.idct(y, norm=norm)
            # rademacher_flip(y, seed=seeds[1], inplace=True)
            # y = y[randperm(x_len, seed=seeds[0], device="cpu", inverse=True)]

        # extract random indices and return
        x = x[randperm(x_len, seed=seeds[4], device="cpu")[:out_dims]]

        """
        INVERSE
        """

        # create output and embed random indices
        y = torch.zeros(x_len, dtype=x.dtype, device=x.device)
        y[randperm(out_dims, seed=seeds[4], device="cpu")[:x_len]] = x
        # invert second scramble: iFFT, rademacher, and inverse permutation
        y = torch.fft.ifft(y, norm=norm)
        phase_shift(y, seed=seeds[3], inplace=True, conj=True)

        import matplotlib.pyplot as plt

        breakpoint()
        # plt.clf(); plt.plot(x); plt.show()

        return x

        # plt.clf(); plt.plot(x); plt.show()

        # extract random indices and return


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
    rademacher_flip(result, seeds[3], inplace=True)
    perm2_inv = randperm(out_dims, seed=seeds[2], device="cpu", inverse=True)
    result = result[perm2_inv]
    del perm2_inv
    # second inverse pass
    result = dct.idct(result, norm=dct_norm)
    rademacher_flip(result, seeds[1], inplace=True)
    perm1_inv = randperm(out_dims, seed=seeds[0], device="cpu", inverse=True)
    result = result[perm1_inv]
    #
    return result


# class SSRFT(BaseRandomLinOp):
#     """Scrambled Subsampled Randomized Fourier Transform (SSRFT).

#     This class encapsulates the left- and right-SSRFT transforms into a single
#     linear operator, which is deterministic for the same shape and seed
#     (particularly, also across different torch devices).
#     """

#     def __init__(self, shape, seed=0b1110101001010101011):
#         """:param shape: ``(height, width)`` of this linear operator."""
#         super().__init__(shape, seed)
#         h, w = shape
#         if h > w:
#             raise BadShapeError("Height > width not supported!")
#         # :param scale: Ideally, ``1/l``, where ``l`` is the average diagonal
#         #   value of the covmat ``A.T @ A``, where ``A`` is a FastJLT operator,
#         #   so that ``l2norm(x)`` approximates ``l2norm(Ax)``.
#         self.scale = NotImplemented

#     def matmul(self, x):
#         """Forward (right) matrix-vector multiplication ``SSRFT @ x``.

#         See parent class for more details.
#         """
#         return ssrft(x, self.shape[0], seed=self.seed, dct_norm="ortho")

#     def rmatmul(self, x):
#         """Adjoint (left) matrix-vector multiplication ``x @ SSRFT``.

#         See parent class for more details.
#         """
#         return ssrft_adjoint(x, self.shape[1], seed=self.seed, dct_norm="ortho")

#     def get_row(self, idx, dtype, device):
#         """Returns SSRFT[idx, :] via left-matmul with a one-hot vector."""
#         in_buff = torch.zeros(self.shape[0], dtype=dtype, device=device)
#         in_buff[idx] = 1
#         return in_buff @ self
