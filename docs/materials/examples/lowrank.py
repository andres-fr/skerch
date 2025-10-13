# -*- coding: utf-8 -*-

r"""Sketched Low-Rank Decompositions
====================================

In this example we create noisy, numerically low-rank matrices and compare
their ``skerch`` low-rank SVD with the built-in ``torch`` counterparts in
terms of  interface, accuracy and speed (see
`Full PyTorch SVD <https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html>`_ and `Sketched PyTorch SVD <https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_lowrank.py>`_).

A key observation here is that ``skerch`` **only requires a bare-minimum
interface**: compatible linear operators must simply feature a
``.shape = (height, width)`` attribute and implement the ``@`` matmul
operator from the left and right handside. Trying to run such a
simple object using the ``torch`` implementations does not work, because
they have more interface requirements.

We then observe that, for low-rank matrices, the ``skerch`` implementation
shows good runtime and accuracy: while being nominally worse than either of
the alternatives, it is still competitive.

This showcases a tradeoff: for this small cost in accuracy and runtime,
``skerch`` allows us to work in much broader settings, also providing
substantial flexibility and extendability (it is simple to
incorporate custom noise sources and recovery methods, see
:ref:`Extending With Custom Functionality`).

Finally, we showcase the a-posteriori functionality, which also works on the
bare-minimum interface and can be used to assess rank and quality of the
recovery, even when the original matrix is still unknown or intractable.
"""

from time import time
import matplotlib.pyplot as plt
import torch

from skerch.synthmat import RandomLordMatrix
from skerch.a_posteriori import apost_error, apost_error_bounds, scree_bounds
from skerch.utils import truncate_decomp
from skerch.algorithms import seigh, ssvd
from skerch.linops import CompositeLinOp, DiagonalLinOp


# %%
#
# ##############################################################################
#
# Creation of test data
# ---------------------
#
# We start by sampling an (approximately) low-rank matrix using
# :class:`skerch.synthmat.RandomLordMatrix`. This is very convenient since
# it allows us fine control over the rank, spectral decay, symmetry and
# diagonal strength (see :ref:`Synthetic Matrices`).
#
# Then we create a bare-bones linear operator from that matrix, to illustrate
# that most existing routines have more restrictive interfaces:

SEED = 54321
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.complex64
SHAPE, RANK, DECAY = (1500, 1500), 100, 0.05
SKETCH_MEAS, TEST_MEAS = 200, 30


class MatLinOp:
    """Bare-bones linear operator, equivalent to the given matrix."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __matmul__(self, x):
        return self.matrix @ x

    def __rmatmul__(self, x):
        return x @ self.matrix


mat = RandomLordMatrix.exp(
    SHAPE, RANK, DECAY, symmetric=False, device=DEVICE, dtype=DTYPE, psd=False
)[0]
lop = MatLinOp(mat)


# %%
#
# ##############################################################################
#
# Compatibility with bare-bones linear operators
# ----------------------------------------------
#
# We now observe that both ``torch.linalg.svd`` and ``torch.svd_lowrank``
# crash when we try to run them on our bare-bones linear operator:

try:
    _ = torch.linalg.svd(lop)
except TypeError as te:
    print("torch.linalg.svd error on linop:", te)

try:
    _ = torch.svd_lowrank(lop)
except AttributeError as te:
    print("torch.svd_lowrank error on linop:", te)


# %%
#
# ##############################################################################
#
# Sketched low-rank approximations
# --------------------------------
#
# We now compute the in-core sketched SVD via :func:`skerch.algorithms.ssvd`.
# Since the ``torch`` alternatives don't run on the bare-bones ``lop``
# interface, we run them using the explicit ``mat``.
# Note that, as already established, this is not a fair comparison in terms
# of scope (plus they may also be optimized for tensors, e.g. via JIT
# compilation).

t0 = time()
U1, S1, Vh1 = torch.linalg.svd(mat)
t1 = time() - t0
U1, S1, Vh1 = U1[:, :SKETCH_MEAS], S1[:SKETCH_MEAS], Vh1[:SKETCH_MEAS]
#
t0 = time()
U2, S2, Vh2 = torch.svd_lowrank(mat, q=SKETCH_MEAS, niter=1)
t2 = time() - t0
#
t0 = time()
U3, S3, Vh3 = ssvd(
    lop,  # runs on lop!
    DEVICE,
    DTYPE,
    SKETCH_MEAS,
    seed=SEED,
    noise_type="rademacher",
    recovery_type="nystrom",
    lstsq_rcond=1e-10,
)
t3 = time() - t0
times = (t1, t2, t3)

err1 = torch.dist(mat, (U1 * S1) @ Vh1).item()
err2 = torch.dist(mat, (U2 * S2) @ Vh2.H).item()
err3 = torch.dist(mat, (U3 * S3) @ Vh3).item()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.bar(["torch SVD (mat)", "torch sSVD (mat)", "skerch sSVD (lop)"], times)
ax2.bar(
    ["torch SVD (mat)", "torch sSVD (mat)", "skerch sSVD (lop)"],
    (err1, err2, err3),
)
fig.suptitle(
    f"Wallclock times and Frobenius errors for rank={SKETCH_MEAS}. "
    "Note that PyTorch implementations run on tensors."
)
fig.tight_layout()


# %%
#
# Despite this not being a fair comparison, we see that ``skerch`` yields
# competitive results in terms of high speed and low error.


# %%
#
# ##############################################################################
#
# Approximate error analysis
# --------------------------
#
# In the code above, we used ``torch.dist`` to measure the error between
# original and recovery. This required us to express them as matrices.
#
# When the linear operator is too large or matrix-free, which is the main
# point of ``skerch`` here, this kind of exact error analysis is not always
# possible, and we resort to approximate *a posteriori* methods
# (see :mod:`skerch.a_posteriori` for more details).
# Crucially, this also works on the bare-bones interface: we only require
# that the input components implement the ``.shape = (height, width)``
# attribute and the ``@`` matmul operator:

lop1 = CompositeLinOp([("U", U1), ("S", DiagonalLinOp(S1)), ("Vh_k", Vh1)])
lop2 = CompositeLinOp([("U", U2), ("S", DiagonalLinOp(S2)), ("Vh_k", Vh2.H)])
lop3 = CompositeLinOp([("U", U3), ("S", DiagonalLinOp(S3)), ("Vh_k", Vh3)])

(_, _, err1sq), _ = apost_error(
    mat, lop1, DEVICE, DTYPE, num_meas=TEST_MEAS, seed=SEED + max(SHAPE) * 2
)
(_, _, err2sq), _ = apost_error(
    mat, lop2, DEVICE, DTYPE, num_meas=TEST_MEAS, seed=SEED + max(SHAPE) * 2
)
(_, f3sq, err3sq), _ = apost_error(
    mat, lop3, DEVICE, DTYPE, num_meas=TEST_MEAS, seed=SEED + max(SHAPE) * 2
)


width = 0.2
fig, ax = plt.subplots(figsize=(8, 4))

# Bars for each tuple
ax.bar(torch.arange(3) - width / 2, (err1, err2, err3), width, label="Exact")
ax.bar(
    torch.arange(3) + width / 2,
    (err1sq.item() ** 0.5, err2sq.item() ** 0.5, err3sq.item() ** 0.5),
    width,
    label="Approximate",
)
ax.set_xticks(torch.arange(3))
ax.set_xticklabels(["torch SVD", "torch sSVD", "skerch sSVD"])
ax.legend()
fig.suptitle(
    f"Exact vs approximate error estimation for {TEST_MEAS} test measurements"
)
fig.tight_layout()

# %%
#
# We see that the approximate errors are very close to the previously computed
# exact ones. The probability of this not happening, for any given number
# of *a posteriori* measurements and error tolerance (in this case 50%), can
# be obtained as follows (see also :ref:`Command Line Interface`):

apost_error_bounds(TEST_MEAS, 0.5)


# %%
#
# ##############################################################################
#
# *A posteriori* rank estimation
# ------------------------------
#
# We can also use the sketched decomposition and the a-posteriori computations
# to obtain the *scree* bounds plotted below. For each `k` (x-axis), the
# scree plot indicates the relative error of our sketched approximation, if
# it was truncated to rank-`k`.
#
# These upper and lower bounds can be used to estimate the effective rank of
# the original operator, e.g. by looking for "elbows" in the scree curve.
# In this case, we have a simple synthetic matrix and a good sketched
# apporximation, so we can see how the scree bounds
# accurately and clearly reflect the given ``RANK`` of the original operator
# (signaled with a vertical line):

scree_lo, scree_hi = scree_bounds(S3, f3sq**0.5, err3sq**0.5)
svals = torch.linalg.svdvals(mat)
scree_true = (svals**2).flip(0).cumsum(0).flip(0)[: len(S3)] / (
    svals**2
).sum()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(scree_lo.cpu(), label="lower", ls="--", linewidth=2)
ax.plot(scree_true.cpu(), label="actual", linewidth=2)
ax.plot(scree_hi.cpu(), label="higher", ls="--", linewidth=2)
ax.axvline(RANK, color="red", linewidth=2)
ax.set_yscale("log")
ax.legend()


# %%
#
# ##############################################################################
#
# And we are done!
#
# * We have seen how to perform sketched SVDs with ``skerch`` on linear
#   operators that support a bare-minimum interface. This is not directly
#   possible with other ``torch`` implementations
# * We observed that the ``skerch`` implementation has bare-minimum
#   requirements on the interface of the linear operators provided, working
#   where other implementations don't, with competitive runtime and accuracy
# * We also demonstrated matrix-free, scalable methods to verify the quality
#   of the sketched approximation and to estimate the rank of the original
#   operator
