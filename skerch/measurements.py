#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to perform sketched measurements."""

from collections import defaultdict
import warnings
import torch
import torch_dct as dct
from .linops import BaseLinOp, ByBlockLinOp
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
# # IID NOISE LINOPS
# ##############################################################################
class RademacherNoiseLinOp(ByBlockLinOp):
    """Random linear operator with i.i.d. Rademacher entries.

    .. warning::

      Since this linop uses random generators and seeds to fetch the blocks,
      it is important that two different instances do not overlap in seeds,
      to prevent correlated noise. Use sufficiently far away seeds and
      ``register=True`` to test this behaviour.

    :param shape: Shape of the linop as ``(h, w)``.
    :param seed: Random seed used in :meth:`get_block` to sample random blocks.
    :param by_row: See :class:`skerch.linops.ByBlockLinOp`.
    :param register: If true, when the linop is created, its seed range
      (going from ``seed`` to ``seed + max(h, w)``) is added to a class-wide
      register, which raises a :class:`skerch.utils.BadSeedError` if there
      are any other instances of this class with overlapping ranges. If
      false, this behaviour is disabled.
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

    def __init__(
        self,
        shape,
        seed,
        by_row=False,
        batch=None,
        blocksize=1,
        register=True,
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, by_row, batch, blocksize)
        self.seed = seed
        #
        if register:
            seed_end = seed + (self.shape[0] if self.by_row else self.shape[1])
            self.__class__.REGISTER["default"].append((seed, seed_end))
            self.check_register()

    def get_block(self, block_idx, input_dtype, input_device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        idxs = self.get_vector_idxs(block_idx)
        h, w = self.shape
        bsize = len(idxs)
        #
        out_shape = (bsize, w) if self.by_row else (h, bsize)
        result = (  # device always CPU to ensure determinism across devices
            rademacher_noise(
                out_shape, seed=self.seed + idxs.start, device="cpu"
            )
            .to(input_dtype)
            .to(input_device)
        )
        return result

    def __repr__(self):
        """Returns a string in the form <classname(shape), attr=value, ...>."""
        clsname = self.__class__.__name__
        byrow_s = ", by row" if self.by_row else ", by col"
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        block_s = f", blocksize={self.blocksize}"
        seed_s = f", seed={self.seed}"
        #
        feats = f"{byrow_s}{batch_s}{block_s}{seed_s}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){feats}>"
        return s


class GaussianNoiseLinOp(RademacherNoiseLinOp):
    """Random linear operator with i.i.d. Gaussian entries.

    Like :class:`RademacherNoiseLinOp`, but with Gaussian noise.
    """

    REGISTER = defaultdict(list)

    def __init__(
        self,
        shape,
        seed,
        by_row=False,
        batch=None,
        blocksize=1,
        register=True,
        mean=0.0,
        std=1.0,
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, seed, by_row, batch, blocksize, register)
        self.mean = mean
        self.std = std

    def get_block(self, block_idx, input_dtype, input_device):
        """Samples a vector with Gaussian i.i.d. noise.

        See base class for details.
        """
        idxs = self.get_vector_idxs(block_idx)
        h, w = self.shape
        bsize = len(idxs)
        #
        out_shape = (bsize, w) if self.by_row else (h, bsize)
        result = gaussian_noise(  # device always CPU to ensure determinism
            out_shape,
            self.mean,
            self.std,
            seed=self.seed + idxs.start,
            dtype=input_dtype,
            device="cpu",
        ).to(input_device)
        return result

    def __repr__(self):
        """Returns a string in the form <classname(shape), attr=value, ...>."""
        clsname = self.__class__.__name__
        byrow_s = ", by row" if self.by_row else ", by col"
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        block_s = f", blocksize={self.blocksize}"
        seed_s = f", seed={self.seed}"
        stats_s = f", mean={self.mean}, std={self.std}"
        #
        feats = f"{byrow_s}{batch_s}{block_s}{seed_s}{stats_s}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){feats}>"
        return s


class PhaseNoiseLinOp(RademacherNoiseLinOp):
    """Random linear operator with i.i.d. complex entries in the unit circle.

    Like :class:`RademacherNoiseLinOp`, but with phase noise. Must be of
    complex datatype.

    :param conj: For the same seed, the linear operators with true and false
      ``conj`` values are complex conjugates of each other.
    """

    REGISTER = defaultdict(list)

    def __init__(
        self,
        shape,
        seed,
        by_row=False,
        batch=None,
        blocksize=1,
        register=True,
        conj=False,
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, seed, by_row, batch, blocksize, register)
        self.conj = conj

    def get_block(self, block_idx, input_dtype, input_device):
        """Samples a vector with i.i.d. phase noise.

        See base class definition for details.
        """
        idxs = self.get_vector_idxs(block_idx)
        h, w = self.shape
        bsize = len(idxs)
        #
        out_shape = (w, bsize) if self.by_row else (h, bsize)
        result = phase_noise(  # device always CPU to ensure determinism
            out_shape, self.seed + idxs.start, input_dtype, device="cpu"
        ).to(input_device)
        #
        if self.conj:
            result = result.conj()
        return result

    def __repr__(self):
        """Returns a string in the form <classname(shape), attr=value, ...>."""
        clsname = self.__class__.__name__
        byrow_s = ", by row" if self.by_row else ", by col"
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        block_s = f", blocksize={self.blocksize}"
        seed_s = f", seed={self.seed}"
        conj_s = f", conj={self.conj}"
        #
        feats = f"{byrow_s}{batch_s}{block_s}{seed_s}{conj_s}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){feats}>"
        return s


# ##############################################################################
# # SSRFT
# ##############################################################################
class SSRFT:
    """Scrambled Subsampled Randomized Fourier Transform (SSRFT).

    This static class implements the forward and adjoint SSRFT, as described
    in `[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_:

    .. math::

      \text{SSRFT} = R\,\mathcal{F}\,\Pi\,\mathcal{F}\,\Pi'

    Where :math:`R` is a random index-picker, \mathcal{F} is either a
    DCT or a FFT (if ``x`` is complex), and :math:`\Pi, \Pi'` are
    random permutations which also multiply entries by Rademacher or
    phase noise (if ``x`` is complex).
    """

    @staticmethod
    def ssrft(x, out_dims, seed=0b1110101001010101011, norm="ortho"):
        r"""Forward SSRFT (see class docstring for definition).

        :param x: Matrix to be projected, such that ``y = SSRFT @ x``
        :param out_dims: Number of rows in ``y`` with ``rows(y) <= rows(x)``
        :param seed: Random seed for the SSRFT.
        :param norm: Norm for the FFT and DCT. Currently only ``ortho`` is
          supported to ensure orthogonality.
        """
        if norm != "ortho":
            raise NotImplementedError("Unsupported norm! use ortho")
        #
        n = x.shape[-1]
        if out_dims > n or out_dims <= 0:
            raise ValueError(
                "out_dims can't be larger than input dimension or <=0!"
            )
        # make sure all sources of randomness are CPU, to ensure cross-device
        # consistency of the operator
        seeds = [seed + i for i in range(5)]
        if x.dtype in COMPLEX_DTYPES:
            # first scramble: permute, phase noise, and FFT
            x = x[..., randperm(n, seed=seeds[0], device="cpu")]
            x = x * phase_noise(
                x.shape[-1],
                seed=seeds[1],
                dtype=x.dtype,
                device="cpu",
                conj=False,
            ).to(x.device)
            x = torch.fft.fft(x, norm=norm)
            # second scramble: permute, phase noise, and FFT
            x = x[..., randperm(n, seed=seeds[2], device="cpu")]
            x = x * phase_noise(
                x.shape[-1],
                seed=seeds[3],
                dtype=x.dtype,
                device="cpu",
                conj=False,
            ).to(x.device)

            x = torch.fft.fft(x, norm=norm)
        else:
            # first scramble: permute, rademacher, and DCT
            x = x[..., randperm(n, seed=seeds[0], device="cpu")]
            x = x * rademacher_noise(x.shape[-1], seeds[1], device="cpu").to(
                x.device
            )
            x = dct.dct(x, norm=norm)
            # second scramble: permute, rademacher and DCT
            x = x[..., randperm(n, seed=seeds[2], device="cpu")]
            x = x * rademacher_noise(x.shape[-1], seeds[3], device="cpu").to(
                x.device
            )
            x = dct.dct(x, norm=norm)
        # extract random indices and return
        x = x[..., randperm(n, seed=seeds[4], device="cpu")[:out_dims]]
        return x

    @staticmethod
    def issrft(x, out_dims, seed=0b1110101001010101011, norm="ortho"):
        r"""Adjoint SSRFT (see class docstring for definition).

        Inversion of the SSRFT, such that for a square SSRFT,
        ``x == issrft(ssrft(x))`` holds.

        Note that this means that, for complex ``x``, the adjoint operation
        involves complex conjugation as well. See class docstring and
        :meth:`ssrft` for more details.

        :param out_dims: In this case, instead of random index-picker, which
          reduces dimension, we have an index embedding, which increases
          dimension by placing the ``x`` entries in the corresponding indices
          (and leaving the rest to zeros). For this reason,
          ``out_dims >= len(x)`` is required.
        """
        if norm != "ortho":
            raise NotImplementedError("Unsupported norm! use ortho")
        n = x.shape[-1]
        if out_dims < n:
            raise ValueError("out_dims can't be smaller than input dimension!")
        # make sure all sources of randomness are CPU, to ensure cross-device
        # consistency of the operator
        seeds = [seed + i for i in range(5)]
        # create output and embed random indices
        out = torch.zeros(
            x.shape[:-1] + (out_dims,), dtype=x.dtype, device=x.device
        )
        out[..., randperm(out_dims, seed=seeds[4], device="cpu")[:n]] = x
        #
        if x.dtype in COMPLEX_DTYPES:
            # invert second scramble: iFFT, rademacher, and inverse permutation
            out = torch.fft.ifft(out, norm=norm)
            out = out * phase_noise(
                out.shape[-1],
                seed=seeds[3],
                dtype=x.dtype,
                device="cpu",
                conj=True,
            ).to(x.device)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[2], device="cpu", inverse=True),
            ]
            # invert first scramble: iFFT, rademacher, and inverse permutation
            out = torch.fft.ifft(out, norm=norm)
            out = out * phase_noise(
                out.shape[-1],
                seed=seeds[1],
                dtype=x.dtype,
                device="cpu",
                conj=True,
            ).to(x.device)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[0], device="cpu", inverse=True),
            ]
        else:
            # invert second scramble: iDCT, rademacher, and inverse permutation
            out = dct.idct(out, norm=norm)
            out = out * rademacher_noise(
                out.shape[-1], seeds[3], device="cpu"
            ).to(x.device)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[2], device="cpu", inverse=True),
            ]
            # invert first scramble: iDCT, rademacher, and inverse permutation
            out = dct.idct(out, norm=norm)
            out = out * rademacher_noise(
                out.shape[-1], seeds[1], device="cpu"
            ).to(x.device)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[0], device="cpu", inverse=True),
            ]
        #
        return out


class SsrftNoiseLinOp(ByBlockLinOp):
    """Linop for the Scrambled Subsampled Randomized Fourier Transform (SSRFT).

    This class encapsulates the forward and adjoint SSRFT transforms into a
    single linear operator with fixed shape and orthonormal columns, which is
    deterministic for the same dtype, shape and seed (also across different
    torch devices).

    See :class:`SSRFT` for more details.


    .. note::

      This linop can either be square or tall, but never fat (i.e. width must
      be less or equal than height). Since the SSRFT cannot increase the
      dimensionality of its input, the forward matmul of this linop is actually
      the inverse SSRFT, and the adjoint matmul is the forward SSRFT.
      This slight change in format that doesn't really affect the semantics of
      the SSRFT, and it makes it more compatible with other noise linops, which
      are typically also tall instead of fat. It is also more common to think
      about orthogonal columns than rows. To make it fat,
      :class:`skerch.linops.TransposedLinOp` can still be used.

    .. note::

      Unlike classes extending :class:`ByBlockLinOp`, in this case it is not
      efficient to apply this operator by row/column. Instead, this
      implementation applies the SSRFT directly to the input, by vector,
      but it also provides ``get_vector`` functionality via one-hot vecmul to
      facilitate parallel measurements via :func:`perform_measurements`.
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

    def __init__(
        self, shape, seed, batch=None, blocksize=1, norm="ortho", register=True
    ):
        """Initializer. See class docstring."""
        by_row = False
        super().__init__(shape, by_row, batch, blocksize)
        h, w = shape
        if w > h:
            raise BadShapeError(
                "Width > height not supported in SSRFT! use transposition or "
                "adjoint matmul."
            )
        self.seed = seed
        self.norm = norm
        #
        if register:
            seed_end = seed + (self.shape[0] if self.by_row else self.shape[1])
            self.__class__.REGISTER["default"].append((seed, seed_end))
            self.check_register()

    def get_block(self, block_idx, input_dtype, input_device):
        """Samples a SSRFT block.

        See base class definition for details.
        """
        idxs = self.get_vector_idxs(block_idx)
        h, w = self.shape
        bsize = len(idxs)
        #
        onehot_mat = torch.zeros(
            (bsize, w), dtype=input_dtype, device=input_device
        )
        onehot_mat[range(bsize), idxs] = 1
        #
        result = SSRFT.issrft(
            onehot_mat,
            self.shape[0],
            seed=self.seed,
            norm=self.norm,
        ).transpose(0, 1)
        return result

    def __repr__(self):
        """Returns a string in the form <classname(shape), attr=value, ...>."""
        clsname = self.__class__.__name__
        byrow_s = ", by row" if self.by_row else ", by col"
        batch_s = "" if self.batch is None else f", batch={self.batch}"
        block_s = f", blocksize={self.blocksize}"
        seed_s = f", seed={self.seed}"
        norm_s = f", norm={self.norm}"
        #
        feats = f"{byrow_s}{batch_s}{block_s}{seed_s}{norm_s}"
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}){feats}>"
        return s
