#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""


TEST STRAT:


* seed consistency: loop over all linop types and check this
"""

import warnings
import torch
from .linops import ByVectorLinOp
from .utils import BadShapeError, BadSeedError, gaussian_noise


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
class GaussianNoiseLinOp(ByVectorLinOp):
    """Random linear operator with i.i.d. Gaussian entries.

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

    REGISTER = set()

    @classmethod
    def check_register(cls):
        """Checks if two different-seeded linops have overlapping seeds."""
        sorted_reg = sorted(cls.REGISTER, key=lambda x: x[0])
        for (beg1, end1), (beg2, end2) in zip(sorted_reg[:-1], sorted_reg[1:]):
            if end1 >= beg2:
                clsname = cls.__name__
                msg = (
                    f"Overlapping seeds when creating {clsname}! "
                    f"({sorted_reg}). This is not necessarily an issue, but "
                    "may lead to different-seeded random linops generating the "
                    "same rows or columns. To prevent this, ensure that the "
                    "random seeds of different noise linops are separated "
                    "by more than the number of rows/columns. To disable "
                    "this behaviour, initialize with register=False."
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
            self.REGISTER.add((seed_beg, seed_end))
            self.check_register()

    def get_vector(self, idx, device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        h, w = self.shape
        dims = w if self.by_row else h
        if idx < 0 or idx >= (h if self.by_row else w):
            raise ValueError(f"Invalid index {idx} for shape {self.shape}!")
        result = gaussian_noise(  # device always CPU to ensure determinism
            dims, seed=self.seed + idx, dtype=self.dtype, device="cpu"
        ).to(device)
        return result

    def __repr__(self):
        """Returns a string: <classname(shape, seed=..., by_row=...)>."""
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}, "
            + f"seed={self.seed}, dtype={self.dtype} by_row={self.by_row})>"
        )
        return s


# ##############################################################################
# # SSRFT
# ##############################################################################
