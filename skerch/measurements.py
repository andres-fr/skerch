#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

* Adapt SSRFT linop to by-block paradigm
* adapt the measurement API to this new paradigm
  - get rid of MP and the meas_fn
  -


this section is strongly biased towards matvec, since I was thinking about
multimachine parallelization.

Currently, users provide max_mp_workers. If none, it is matvec, and if given,
the for loop is distributed across machines.

Before anything:
* show that MP is reliable and useful in an integration test case
  - create a smal NN and problem, where HVP takes like 1 sec
  - 20 measurements should take 20 seconds
  - distribute across 5 cores, now it should take around 4 sec.

Even if we keep MP, the meas_fn pattern is ugly AF and prevents batching.
How to enable batching while being MP compatible?


---
ok, so given lop could be anything! so we need to support that.
* use cases:
  - just run everything in baselinop mode (give batch=None or int)
    useful if the linop supports batching and is just a plain @ call really
  - user specifies which measurement idxs should be run (i.e. @ is not
    precise enough). In this case we also have batch, which means that
    a bunch of vectors are gathered at once.

---
SO:

*

* add batch to by_vector get_measvec


"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
# # CONVENIENCE WRAPPERS
# ##############################################################################
def get_measvec(idx, meas_lop, device=None, dtype=None):
    """ """
    if isinstance(meas_lop, torch.Tensor):
        measvec = meas_lop[:, idx]
    elif isinstance(meas_lop, SsrftNoiseLinOp):
        if device is None or dtype is None:
            raise ValueError("SsrftNoiseLinop requires device and dtype!")
        measvec = meas_lop.get_vector(idx, device, dtype, by_row=False)
    else:
        measvec = meas_lop.get_vector(idx, device)
    #
    return measvec


def lop_measurement(idx, adjoint, lop, meas_lop, device=None, dtype=None):
    """ """
    measvec = get_measvec(idx, meas_lop, device, dtype)
    result = (measvec.conj() @ lop) if adjoint else (lop @ measvec)
    return idx, result


def perform_measurements(
    meas_fn,
    meas_idxs,
    adjoint=False,
    parallel_mode=None,
    compact=False,
    max_mp_workers=None,
):
    """
    :param meas_fn: Function callable with ``meas_fn(idx)`` that returns
      the pair ``idx, v_idx``, where ``v_idx`` is a vector corresponding
      to the desired measurement at given index.
    """
    result = {}
    meas_fn = partial(meas_fn, adjoint=adjoint)
    #
    if parallel_mode is None:
        warnings.warn(
            "CPU measurements could be parallelized with parallel_mode=mp",
            RuntimeWarning,
        )
        for idx in meas_idxs:
            result[idx] = meas_fn(idx)[1]
    #
    elif parallel_mode == "mp":
        # keep an eye on this, could hang
        with ProcessPoolExecutor(max_workers=max_mp_workers) as pool:
            result = dict(pool.map(meas_fn, meas_idxs))
    #
    else:
        raise ValueError(f"Unknown parallel_mode! {parallel_mode}")
    #
    if compact:
        sorted_idxs = sorted(result)
        result = torch.stack(
            [result[idx] for idx in sorted_idxs], dim=0 if adjoint else 1
        )
        result = (sorted_idxs, result)
    return result


# ##############################################################################
# # IID NOISE LINOPS
# ##############################################################################
class RademacherNoiseLinOp(ByBlockLinOp):
    """Random linear operator with i.i.d. Rademacher entries.

    :param shape: Shape of the linop as ``(h, w)``.
    :param seed: Random seed used in ``get_vector`` to deterministically sample
      random vectors. Each vector with ``idx`` is sampled from ``seed + idx``,
      for this reason is important that two different linops are instantiated
      with sufficiently distant seeds, to prevent overlaps.
    :param dtype: Dtype of the generated noise.
    :param by_row: See :class:`ByBlockLinOp`.
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

    def get_block(self, idxs, input_dtype, input_device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        blocksize = len(idxs)
        if idxs.start < 0 or idxs.stop > (h if self.by_row else w):
            raise ValueError("Invalid range {idxs} for shape {self.shape}!")
        out_shape = (blocksize, w) if self.by_row else (h, blocksize)
        result = (  # device always CPU to ensure determinism across devices
            rademacher_noise(
                out_shape, seed=self.seed + idxs.start, device="cpu"
            )
            .to(input_dtype)
            .to(input_device)
        )
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"seed={self.seed}, by_row={self.by_row})>"
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

    def get_block(self, idxs, input_dtype, input_device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        blocksize = len(idxs)
        if idxs.start < 0 or idxs.stop > (h if self.by_row else w):
            raise ValueError("Invalid range {idxs} for shape {self.shape}!")
        out_shape = (blocksize, w) if self.by_row else (h, blocksize)
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
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"mean={self.mean}, std={self.std}, "
            + f"seed={self.seed}, by_row={self.by_row})>"
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

    def get_block(self, idxs, input_dtype, input_device):
        """Samples a vector with Rademacher i.i.d. noise.

        See base class definition for details.
        """
        if input_dtype not in COMPLEX_DTYPES:
            raise ValueError(f"Input dtype must be complex! was {input_dtype}")
        #
        h, w = self.shape
        blocksize = len(idxs)
        if idxs.start < 0 or idxs.stop > (h if self.by_row else w):
            raise ValueError("Invalid range {idxs} for shape {self.shape}!")
        out_shape = (blocksize, w) if self.by_row else (h, blocksize)
        result = phase_noise(  # device always CPU to ensure determinism
            out_shape, self.seed + idxs.start, input_dtype, device="cpu"
        ).to(input_device)
        #
        if self.conj:
            result = result.conj()
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"conj={self.conj}, "
            + f"seed={self.seed}, by_row={self.by_row})>"
        )
        return s


# ##############################################################################
# # SSRFT
# ##############################################################################
class SSRFT:
    """ """

    @staticmethod
    def ssrft(x, out_dims, seed=0b1110101001010101011, norm="ortho"):
        r"""Forward SSRFT.

        This function implements a matrix-free, right-matmul operator of the
        Scrambled Subsampled Randomized Fourier Transform (SSRFT), see e.g.
        `[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_.

        .. math::

          \text{SSRFT} = R\,\mathcal{F}\,\Pi\,\mathcal{F}\,\Pi'

        Where :math:`R` is a random index-picker, \mathcal{F} is either a
        DCT or a FFT (if ``x`` is complex), and :math:`\Pi, \Pi'` are
        random permutations which also multiply entries by Rademacher or
        phase noise (if ``x`` is complex).

        :param x: Matrix to be projected, such that ``y = SSRFT @ x``
        :param out_dims: Number of rows in ``y`` with ``rows(y) <= rows(x)``
        :param seed: Random seed
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
            phase_shift(
                x, seed=seeds[1], inplace=True, rng_device="cpu", conj=False
            )
            x = torch.fft.fft(x, norm=norm)
            # second scramble: permute, phase noise, and FFT
            x = x[..., randperm(n, seed=seeds[2], device="cpu")]
            phase_shift(
                x, seed=seeds[3], inplace=True, rng_device="cpu", conj=False
            )
            x = torch.fft.fft(x, norm=norm)
        else:
            # first scramble: permute, rademacher, and DCT
            x = x[..., randperm(n, seed=seeds[0], device="cpu")]
            rademacher_flip(x, seed=seeds[1], inplace=True, rng_device="cpu")
            x = dct.dct(x, norm=norm)
            # second scramble: permute, rademacher and DCT
            x = x[..., randperm(n, seed=seeds[2], device="cpu")]
            rademacher_flip(x, seed=seeds[3], inplace=True, rng_device="cpu")
            x = dct.dct(x, norm=norm)
        # extract random indices and return
        x = x[..., randperm(n, seed=seeds[4], device="cpu")[:out_dims]]
        return x

    @staticmethod
    def issrft(x, out_dims, seed=0b1110101001010101011, norm="ortho"):
        r"""Inverse SSRFT.

        Inversion of the SSRFT, such that for a square ssrft,
        ``x == issrft(ssrft(x))`` holds.
        Note that this means that, for complex ``x``, the adjoint operation
        involves complex conjugation as well.
        See :meth:`.ssrft` for more details.

        :param out_dims: In this case, instead of random index-picker, which
          reduces dimension, we have an index embedding, which increases
          dimension by placing the ``x`` entries in the corresponding indices
          (and leaving the rest to zeros). For this reason,
          ``out_dims >= len(x)`` is required.
        """
        if norm != "ortho":
            raise NotImplementedError("Unsupported norm! use ortho")
        # if len(x.shape) != 1 or x.numel() <= 0:
        #     raise BadShapeError(f"Input must be a nonempty vector! {x.shape}")
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
            phase_shift(out, seed=seeds[3], inplace=True, conj=True)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[2], device="cpu", inverse=True),
            ]
            # invert first scramble: iFFT, rademacher, and inverse permutation
            out = torch.fft.ifft(out, norm=norm)
            phase_shift(out, seed=seeds[1], inplace=True, conj=True)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[0], device="cpu", inverse=True),
            ]
        else:
            # invert second scramble: iDCT, rademacher, and inverse permutation
            out = dct.idct(out, norm=norm)
            rademacher_flip(out, seed=seeds[3], inplace=True)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[2], device="cpu", inverse=True),
            ]
            # invert first scramble: iDCT, rademacher, and inverse permutation
            out = dct.idct(out, norm=norm)
            rademacher_flip(out, seed=seeds[1], inplace=True)
            out = out[
                ...,
                randperm(out_dims, seed=seeds[0], device="cpu", inverse=True),
            ]
        #
        return out


class SsrftNoiseLinOp(ByBlockLinOp):
    """Scrambled Subsampled Randomized Fourier Transform (SSRFT).

    This class encapsulates the forward and adjoint SSRFT transforms into a
    single linear operator with orthogonal columns, which is deterministic for
    the same shape and seed (also across different torch devices).


    .. note::

      This linop can either be square or tall, but never fat (i.e. width must
      be less or equal than height). Since the SSRFT cannot increase the
      dimensionality of its input, the forward matmul of this linop is actually
      the inverse SSRFT, and the adjoint matmul is the forward SSRFT.
      This slight change in format that doesn't really affect the semantics of
      the SSRFT, and it makes it more compatible with other noise linops, which
      are typically also tall instead of fat. It is also more common to think
      about orthogonal columns than rows.

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

    # def matmul(self, x):
    #     """Forward (right) matrix-vector multiplication ``self @ x``.

    #     See class docstring and parent class for more details.
    #     """
    #     # note that the issrft acts like the Hermitian transpose of A, but here
    #     # we don't want A.H@x, but x@A. We achieve this via (A.H @ x.H).H,
    #     # which equals (x @ A).H.H = x@A.
    #     x = x.transpose(0, -1)
    #     result = SSRFT.issrft(
    #         x.conj(), self.shape[0], seed=self.seed, norm=self.norm
    #     ).conj()
    #     result = result.transpose(0, -1)
    #     return result

    # def rmatmul(self, x):
    #     """Left matrix-vector multiplication ``x @ self``.

    #     See class docstring and parent class for more details.
    #     """
    #     return SSRFT.ssrft(x, self.shape[1], seed=self.seed, norm=self.norm)

    # def get_vector(self, idx, device, dtype, by_row=False):
    #     """Samples a SSRFT row or column.

    #     :param idx: Number between 0 and below number of columns (resp. rows),
    #       indicating the corresponding vector to be sampled.
    #     :param by_row: If false, the ``idx`` column (zero-indexed) will be
    #       sampled. Otherwise the column.
    #     """
    #     h, w = self.shape
    #     in_dims = h if by_row else w
    #     out_dims = w if by_row else h
    #     if idx < 0 or idx >= (h if by_row else w):
    #         raise ValueError(
    #             f"Invalid index {idx} for shape {self.shape} "
    #             f"and by_row={by_row}!"
    #         )
    #     oh = torch.zeros(in_dims, dtype=dtype, device=device)
    #     oh[idx] = 1
    #     result = oh @ self if by_row else self @ oh
    #     return result

    def get_block(self, idxs, input_dtype, input_device):
        """Samples a SSRFT block.

        See base class definition for details.
        """
        h, w = self.shape
        blocksize = len(idxs)
        if idxs.start < 0 or idxs.stop > (h if self.by_row else w):
            raise ValueError("Invalid range {idxs} for shape {self.shape}!")
        #
        onehot_mat = torch.zeros(
            (blocksize, w), dtype=input_dtype, device=input_device
        )
        onehot_mat[range(blocksize), idxs] = 1
        #
        result = (
            SSRFT.issrft(
                onehot_mat,
                self.shape[0],
                seed=self.seed + idxs.start,
                norm=self.norm,
            ).transpose(0, 1)
            # .conj()
        )
        return result

    def matmul(self, x):
        """ """
        ### return self._bb_matmul_helper(x, adjoint=False)
        return self._bb_matmul_helper(x.conj(), adjoint=False).conj()

        #     x = x.transpose(0, -1)
        #     result = SSRFT.issrft(
        #         x.conj(), self.shape[0], seed=self.seed, norm=self.norm
        #     ).conj()
        #     result = result.transpose(0, -1)
        #     return result

    def rmatmul(self, x):
        """ """
        return self._bb_matmul_helper(x, adjoint=True)

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}, seed={self.seed})>"
        return s

    def __repr__(self):
        """Returns a string: <classname(shape, seed=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"seed={self.seed})>"
        )
        return s
