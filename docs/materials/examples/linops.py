# -*- coding: utf-8 -*-

r"""Linear Operators and Matrix-Freedom
=======================================

In this example we explore the rich linear operator API available in
``skerch``, specifically:

* Creation of a ``skerch``-compatible linear operator
* Matrix-free transposition
* Wrapping numpy-only linear operators into PyTorch
* Matrix-free composition and addition of linear operators
* Matrix-free diagonal linear operators
* Matrix-free noisy linear operators for sketching

This functionality allows us to work with linear operators at scale.
"""

from time import time
import matplotlib.pyplot as plt
import torch
from skerch.utils import gaussian_noise
from skerch.linops import linop_to_matrix, TransposedLinOp, TorchLinOpWrapper
from skerch.linops import ByBlockLinOp, CompositeLinOp, SumLinOp
from skerch.linops import DiagonalLinOp, BandedLinOp
from skerch.measurements import (
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SsrftNoiseLinOp,
)
from skerch.algorithms import snorm


# %%
#
# ##############################################################################
#
# Creating a ``skerch``-compatible linop
# -------------------------------------
#
# To work with ``skerch``, linear operators must satisfy only 2 requirements:
# * They must support left- and right-matrix multiplication via ``@``
# * They must feature a ``.shape = (height, width)`` attribute
#
# This is satisfied by regular matrices like the following ``mat``:

SHAPE = (50, 50)
NUM_MEAS = 10
SEED = 12345
DTYPE = torch.complex64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mat = gaussian_noise(SHAPE, 0, 10, seed=SEED, dtype=DTYPE, device=DEVICE)
ramp = torch.arange(mat.shape[1], dtype=mat.real.dtype, device=mat.device)
mat += ramp + 1j * ramp

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(mat.real.cpu())
ax2.imshow(mat.imag.cpu())
fig.suptitle("Some test matrix")
fig.tight_layout()


# %%
# But often we don't have full matrices. Instead, we have some matrix-free
# linear operator at a potentially very large scale, defined by its shape
# and matmul functionality. As an example, the following matrix-free operator
# will be compatible with all numerical routines provided in ``skerch``,
# because it provides ``.shape`` and ``@``:


class SomeLinOp:
    """Some matrix-free linop to exemplify skerch-compatibility."""

    def __init__(self, dims):
        self.shape = (dims, dims)

    def __matmul__(self, x):
        result = x * 0.5
        result[0, ...] += x.sum(dim=0)
        return result

    def __rmatmul__(self, x):
        result = x * 0.5
        result = (result.transpose(0, -1) + x[..., 0]).transpose(0, -1)
        return result


lop = SomeLinOp(SHAPE[0])
lopmat = linop_to_matrix(lop, dtype=ramp.dtype, device="cpu")
lop_S = torch.linalg.svdvals(lopmat)
op_norm = snorm(
    lop,
    DEVICE,
    DTYPE,
    num_meas=NUM_MEAS,
    seed=SEED + 123,
    noise_type="gaussian",
    norm_types=["op"],
)[0]["op"]
print("Lop norm:", torch.linalg.svdvals(lopmat).max().item())
print("Sketched norm:", op_norm.item())

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.plot(ramp.cpu(), label="original")
ax1.plot(lop @ ramp.cpu(), label="scaled and flipped")
ax2.imshow(lopmat)
ax1.legend()
fig.suptitle("The matrix-free linop and its action")
fig.tight_layout()


# %%
#
# ##############################################################################
#
# Matrix-free transposition
# -------------------------
#
# We can also use :class:`skerch.linops.TransposedLinOp` to transpose linear
# operators in a matrix-free fashion:

lopT = TransposedLinOp(lop)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(linop_to_matrix(lop, dtype=ramp.dtype, device="cpu"))
ax2.imshow(linop_to_matrix(lopT, dtype=ramp.dtype, device="cpu"))
fig.suptitle("The linear operator and its transpose")
fig.tight_layout()

# %%
#
# ##############################################################################
#
# PyTorch wrapper
# ---------------
#
# Since ``skerch`` is built on PyTorch, it won't directly work on numpy
# linops that don't accept PyTorch inputs. This can be solved with the wrapper:


class NumpyFlipLinOp:
    """Numpy-only antidiagonal linop. Doesn't work with torch tensors."""

    def __init__(self, shape):
        self.shape = shape

    def __matmul__(self, x):
        return x[::-1].copy()

    def __rmatmul__(self, x):
        return self.__matmul__(x)


class TorchFlipLinOp(TorchLinOpWrapper, NumpyFlipLinOp):
    """This wrapper works with tensors and numpy arrays."""


np_lop = NumpyFlipLinOp(SHAPE)
torch_lop = TorchFlipLinOp(SHAPE)
arr = ramp.cpu().numpy()


print("Numpy linop on numpy data:", np_lop @ arr)
print("Wrapped linop on numpy data:", torch_lop @ arr)
print("Wrapped linop on torch data:", torch_lop @ ramp)


# %%
#
# ##############################################################################
#
# Other linear operators
# ----------------------
#
# We can perform matrix-free compositions and additions of linear operators.
# Other matrix-free structured linops, such as diagonal and banded, are also
# available (see :module:`skerch.linops`):

k = 2
U, S, Vh = torch.linalg.svd(mat.cpu())

lop_k = CompositeLinOp(
    [("U_k", U[:, :k]), ("S_k", DiagonalLinOp(S[:k])), ("Vh_k", Vh[:k, :])]
)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(mat.real.cpu())
ax2.imshow(linop_to_matrix(lop_k, dtype=mat.dtype, device="cpu").real)
fig.suptitle(f"Matrix and its top-{k} matrix-free approximation [{lop_k}]")
fig.tight_layout()

# %%
#
# ##############################################################################
#
# Matrix-free noisy linear operators for sketching
# ------------------------------------------------
#
# In order to run the sketches, ``skerch`` provides built-in support for noisy
# measurements in the form of matrix-free linear operators. In order to
# facilitate parallelized measurements, these linops are a bit more
# restrictive: besides the ``.shape`` and ``@`` properties required by all
# ``skerch`` linops, they must also implement a ``get_blocks`` iterator, that
# yields blocks of columns and their indices.
#
# A good way to add new types of noise into existing ``skerch`` algorithms is
# then to extend :class:`skerch.linops.ByBlockLinOp` with ``get_blocks``,
# and then extending :class:`skerch.algorithms.SketchedAlgorithmDispatcher`
# adding the newly created type of noise to the register (see e.g. the
# source code for :class:`skerch.measurements.RademacherNoiseLinOp` and
# :func:`skerch.algorithms.snorm` for examples).
#
# The figure below illustrates some of the already supported types of noise:

blocksize = 5
mop1 = RademacherNoiseLinOp(SHAPE, SEED, blocksize=blocksize)
mop2 = GaussianNoiseLinOp(SHAPE, SEED, blocksize=blocksize)
mop3 = PhaseNoiseLinOp(SHAPE, SEED, blocksize=blocksize)
mop4 = SsrftNoiseLinOp(SHAPE, SEED, blocksize=blocksize)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
ax1.imshow(mop1.to_matrix(DTYPE, "cpu").real)
ax2.imshow(mop2.to_matrix(DTYPE, "cpu").real)
ax3.imshow(mop3.to_matrix(DTYPE, "cpu").real)
ax4.imshow(mop4.to_matrix(DTYPE, "cpu").real)
fig.suptitle(f"Different types of noise matrices")
fig.tight_layout()


# %%
# And the line below illustrates the behaviour of ``get_blocks``:

mop1_blocks_info = [(b.shape, idxs) for b, idxs in mop1.get_blocks(DTYPE)]

# %%
# To illustrate the necessity of blockwise measurements, consider the
# following example, where a larger block size results in substantially faster
# computations:

mop_shape = (mat.shape[1], 100)
mop_slow = RademacherNoiseLinOp(mop_shape, SEED, blocksize=1, register=False)
mop_fast = RademacherNoiseLinOp(mop_shape, SEED, blocksize=100, register=False)
times = [[], []]
for i in range(20):
    t0 = time()
    mat @ mop_slow
    times[0].append(time() - t0)
    #
    t0 = time()
    mat @ mop_fast
    times[1].append(time() - t0)

fig, ax = plt.subplots()
ax.boxplot(times, label=["blocksize=1", "blocksize=100"])
ax.set_yscale("log")
fig.suptitle(f"Speedup resulting from blockwise measurements")
fig.tight_layout()


# %%
#
# ##############################################################################
#
# This concludes the ``skerch`` tour of linear operators! Please refer to the
# API docs and other examples. for more details. In summary:
#
# * We have seen how to create simple matrix-free linops, so
#   that they are compatible with the ``skerch`` routines.
# * We also saw how to manipulate said linops, by transposing them, converting
#   them to matrices and wrapping them to ensure NumPy compatibility
#   in-core sketched methods for a broad class of low-rank matrices
# * We explored other available linear operators, including compositions and
#   noisy measurement linops, emphasizing the benefit of blockwise
#   measurements.
