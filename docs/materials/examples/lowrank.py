# -*- coding: utf-8 -*-

r"""Sketched Decompositions
===========================

In this example we create noisy, numerically low-rank matrices
and compare their ``skerch`` low-rank SVD with the built-in
 PyTorch ones in terms of accuracy and speed:
* `Full PyTorch SVD <https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html>`
* `Sketched PyTorch SVD <https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_lowrank.py>`

We observe that, for low-rank matrices, both sketched methods are faster and
more accurate than the full SVD. We also notice that ``skerch`` is a bit slower
and worse than the sketched PyTorch SVD. This may seem discouraging, but we
note three mitigating factors:
* At larger scales, the difference between ``skerch`` and the sketched PyTorch
  implementation vanishes: due to its modularity, ``skerch`` likely has more
  overhead and less optimizations (e.g. JIT) to run the algorithm.
* At large scales, very small errors in sketched methods are rare to obtain
  since we typically cannot afford enough measurements to cover the spectrum.
* One main strength of sketched methods is their modularity and flexibility.
  For a small overhead, ``skerch`` allows to easliy extend and swap different
  components. Optimized implementations tend to be more rigid.

To conclude, we showcase the a-posteriori functionality, which can be used
to assess rank and quality of the recovery when the original matrix is not
known.
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
# diagonal strength:

SEED = 54321
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.complex64
SHAPE, RANK, DECAY = (1500, 1500), 100, 0.05
SKETCH_MEAS, TEST_MEAS = 200, 30

mat = RandomLordMatrix.exp(
    SHAPE, RANK, DECAY, symmetric=False, device=DEVICE, dtype=DTYPE, psd=False
)[0]


# %%
#
# ##############################################################################
#
# Sketched low-rank approximations
# --------------------------------
#
# We can now conveniently compute our in-core sketched decomposition via
# :func:`skerch.algorithms.ssvd` (sketched SVD), and compare it with the
# built-in PyTorch alternatives:

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
    mat,
    DEVICE,
    DTYPE,
    SKETCH_MEAS,
    seed=SEED,
    noise_type="rademacher",
    recovery_type="singlepass",
)
t3 = time() - t0
times = (t1, t2, t3)

err1 = torch.dist(mat, (U1 * S1) @ Vh1).item()
err2 = torch.dist(mat, (U2 * S2) @ Vh2.H).item()
err3 = torch.dist(mat, (U3 * S3) @ Vh3).item()

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.bar(["torch SVD", "torch sSVD", "skerch sSVD"], times)
ax2.bar(["torch SVD", "torch sSVD", "skerch sSVD"], (err1, err2, err3))
fig.suptitle(f"Wallclock times and Frobenius errors for rank={SKETCH_MEAS}")
fig.tight_layout()


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
# When the linear operator is too large or matrix-free, this kind of exact
# error analysis is not always possible, and we resort to approximate
# *a posteriori* methods (see :mod:`skerch.a_posteriori` for more details).
# As usual with ``skerch``, we only require that the input components
# implement the ``.shape = (height, width)`` attribute and the ``@`` matmul
# operator:

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
fig, ax = plt.subplots()

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
# case we have a simple synthetic matrix, we can see how the scree bounds
# accurately and clearly reflect the given ``RANK`` of the original operator
# (signaled with a vertical line):

scree_lo, scree_hi = scree_bounds(S3, f3sq**0.5, err3sq**0.5)
svals = torch.linalg.svdvals(mat)
scree_true = (svals**2).flip(0).cumsum(0).flip(0)[: len(S3)] / (
    svals**2
).sum()

fig, ax = plt.subplots()
ax.plot(scree_lo.cpu(), label="lower", ls="--", linewidth=2)
ax.plot(scree_true.cpu(), label="ACTUAL", linewidth=2)
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
# * We have seen how to perform sketched SVDs with ``skerch``, observing that
#   sketched decompositions are fast and accurate
# * We demonstrated matrix-free, scalable methods to verify the quality
#   of the approximation and to estimate the rank of the original operators
