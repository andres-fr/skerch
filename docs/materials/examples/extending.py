# -*- coding: utf-8 -*-

r"""Extending With Custom Functionality
=======================================

In this example we see how to extend ``skerch`` with custom functionality.
Specifically:

* Adding a new recovery method for low-rank sketched algorithms
* Adding a new noise source

This showcases the practicality of ``skerch``: Not only it works on linops
that satisfy very simple interfaces with fair speed and accuracy,
but it can also be easily extended and modified to facilitate new applications
and research directions.
"""

from collections import defaultdict
from time import time
import matplotlib.pyplot as plt
import torch

from skerch.synthmat import RandomLordMatrix
from skerch.measurements import GaussianNoiseLinOp
from skerch.algorithms import SketchedAlgorithmDispatcher, ssvd


# %%
#
# ##############################################################################
#
# Creation of test data
# ---------------------
#
# We start by sampling an (approximately) low-rank matrix using
# :class:`skerch.synthmat.RandomLordMatrix`, and then running the built-in
# :func:`skerch.algorithms.ssvd` via Nystrom recovery with Rademacher noise,
# yielding good accuracy:

SEED = 124816315799
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.complex128
SHAPE, RANK, DECAY = (100, 200), 10, 0.1
SKETCH_MEAS, TEST_MEAS = 50, 30

mat = RandomLordMatrix.exp(
    SHAPE, RANK, DECAY, symmetric=False, device=DEVICE, dtype=DTYPE, psd=False
)[0]

sU, sS, sVh = ssvd(
    mat,
    DEVICE,
    DTYPE,
    SKETCH_MEAS,
    seed=SEED,
    noise_type="rademacher",
    recovery_type="nystrom",
    lstsq_rcond=1e-10,
)
smat = (sU * sS) @ sVh
print(
    "Relative error (Rademacher+Nystrom):",
    (torch.dist(mat, smat) / mat.norm()).item(),
)


# %%
#
# ##############################################################################
#
# Testing a new recovery method
# -----------------------------
#
# Let's now test a wild theory: since random matrices are so cool, maybe a
# random sample is also a good recovery? With ``skerch``, all we need to do is:
# 1. Define our new recovery method
# 2. Extend the dispatcher to provide the recovery as needed
# 3. Feed the requested string and dispatcher to the existing SVD algorithm


def bogo_recovery(sketch_right, sketch_left, *args, **kwargs):
    """Just guess the output. How bad could it be?"""
    U = torch.linalg.qr(torch.randn_like(sketch_right))[0]
    Vh = torch.linalg.qr(torch.randn_like(sketch_left.H))[0].H
    S = torch.randn_like(U[0]).abs().sort(descending=True)[0]
    if kwargs["as_svd"]:
        return U, S, Vh
    else:
        return U * S, Vh


class BogoDispatcher(SketchedAlgorithmDispatcher):
    """A custom dispatcher that provides ``bogo_recovery``."""

    @staticmethod
    def recovery(recovery_type, hermitian=False):
        """Returns recovery funtion with given specs."""
        if recovery_type == "bogo":
            return bogo_recovery, None
        else:
            raise ValueError(f"Unknown recovery! {recovery_type}")


bU, bS, bVh = ssvd(
    mat,
    DEVICE,
    DTYPE,
    SKETCH_MEAS,
    seed=SEED,
    noise_type="rademacher",
    recovery_type="bogo",  # changed!
    lstsq_rcond=1e-10,
    dispatcher=BogoDispatcher,  # changed!
)
bmat = (bU * bS) @ bVh
print(
    "Relative error (Rademacher+Bogo):",
    (torch.dist(mat, bmat) / mat.norm()).item(),
)


# %%
#
# Oops! It seems that BogoRecovery is not a good method, and we should
# stick to the big guns. Good to know, and all in a couple dozen lines
# of code!
#
# .. note::
#   Currently, recovery methods and dispatcher must fulfill particular
#   interfaces (see :mod:`skerch.recovery` for examples). To try methods
#   that deviate from those, the best practice is probably to copypaste
#   the ``ssvd`` function and adjust the parts that break compatibility.


# %%
#
# ##############################################################################
#
# Testing a new measurement distribution
# --------------------------------------
#
# OK but hear me out: since random matrices are so cool, maybe some other
# arbitrary form of noise also provides a good recovery? Or maybe you
# suspect that a particular type of noise is best suited for a particular
# family of linear operators? With ``skerch``, this can be easily tested:
# 1. Define our new measurement linop by extending
#   :class:`skerch.measurements.ByBlockLinOp`
# 2. Extend the dispatcher to provide the measurement linop as needed
# 3. Feed the requested string and dispatcher to the existing SVD algorithm


class GaussemacherNoiseLinOp(GaussianNoiseLinOp):
    """Gaussian noise with a hard lower bound on the magnitude."""

    REGISTER = defaultdict(list)

    THRESHOLD = 0.5

    def __init__(
        self, shape, seed, by_row=False, batch=None, blocksize=1, register=True
    ):
        super().__init__(
            shape, seed, by_row, batch, blocksize, register, 0.0, 1.0
        )

    def get_block(self, block_idx, input_dtype, input_device):
        result = super().get_block(block_idx, input_dtype, input_device)
        mag = result.abs()
        scale = torch.where(
            mag < self.THRESHOLD,
            self.THRESHOLD / (mag + 1e-7),
            torch.ones_like(mag),
        )
        return result * scale


class GaussemacherDispatcher(SketchedAlgorithmDispatcher):
    @staticmethod
    def mop(noise_type, hw, seed, dtype, blocksize=1, register=False):
        """ """
        if "gaussemacher" in noise_type:
            mop = GaussemacherNoiseLinOp(
                hw, seed, blocksize=blocksize, register=register
            )
        else:
            raise ValueError(f"Unknown noise type! {noise_type}")
        return mop


gU, gS, gVh = ssvd(
    mat,
    DEVICE,
    DTYPE,
    SKETCH_MEAS,
    seed=SEED,
    noise_type="gaussemacher",  # changed!
    recovery_type="nystrom",
    lstsq_rcond=1e-10,
    dispatcher=GaussemacherDispatcher,  # changed!
)
gmat = (gU * gS) @ gVh
print(
    "Relative error (Gaussemacher(0.5)+Nystrom):",
    (torch.dist(mat, gmat) / mat.norm()).item(),
)


# %%
#
# So this actually works? Maybe random matrices aren't that bad after
# all...


# %%
#
# ##############################################################################
#
#
# In Summary:
# -----------
#
# * We have seen how to extend ``skerch`` with new low-rank recovery methods
#   with just a few lines of code.
# * Similarly, we can also add new noise sources with little effort.
# * Still, some interfaces must be satisfied to run built-in code. Whenever
#   your interfaces collide (e.g. you require a new type of input), best
#   advice is to copypaste and modify the algorithm, which thanks to the
#   modularity of ``skerch`` is also fairly low-effort.
