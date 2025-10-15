# -*- coding: utf-8 -*-

r"""Traces, Diagonals, Triangles
================================

In this example we explore the following ``skerch`` functionality:

* Trace and diagonal estimations
* Triangular matrix-vector multiplications

Given a linear operator :math:`A`, we first perform the trace and diagonal
estimation using variations of the Girard-Hutchinson sketched method. The
computations needed for both estimations are very similar and can be mostly
recycled to compute both quantities at once.
To illustrate the benefits of low-rank deflation for diagonal estimation, we
run these methods on a full-rank and a low-rank matrix.

Then, we move onto triangular mat-vec estimation, i.e. :math:`tril(A) v` and
:math:`triu(A) v`, which also makes use of a modification of Girard-Hutchinson
combined with deterministic measurements.

We verify the accuracy of the sketched approximations by comparing them to the
actual quantities.

.. note::
  One core feature of Girard-Hutchinson is its rather slow convergence rate:
  in general, doing just a few noisy measurements can introduce large amounts
  of error and be worse that not doing it at all (especially if entries in
  the measurement vectors span multiple orders of magnitude).
  To obtain reliable estimates at scale, typically measurements must be
  in the order of thousands (see Table 1 in
  `[BN2022] <https://arxiv.org/abs/2201.10684>`_ for bounds). This is still
  fine nonetheless for linear operators with large ambient dimension.
"""

from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from skerch.utils import gaussian_noise
from skerch.synthmat import RandomLordMatrix
from skerch.algorithms import hutchpp, TriangularLinOp


# %%
#
# ##############################################################################
#
# Creation of test matrices
# -------------------------
#
# We create two matrices, with smooth and fast decaying spectrum:

SEED = 392781
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DIMS, RANK = 3000, 100
DEFL_DIMS, GH_MEAS = 75, 2000

shape = (DIMS, DIMS)
mat = RandomLordMatrix.exp(
    shape, RANK, 0.001, seed=SEED + 1, device=DEVICE, dtype=DTYPE
)[0]
lomat = RandomLordMatrix.exp(
    shape, RANK, 100, seed=SEED, device=DEVICE, dtype=DTYPE
)[0]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
ax1.plot(torch.linalg.svdvals(mat.cpu()))
ax2.plot(torch.linalg.svdvals(lomat.cpu()))
ax1.set_title("Smooth decay")
ax2.set_title("Steep decay")
fig.suptitle("Singular values")
fig.tight_layout()

# %%
#
# ##############################################################################
#
# Trace and diagonal estimation via Hutch++
# -----------------------------------------
#
# Hutch++ (implemented in :func:`skerch.algorithms.hutchpp`) combines
# Girard-Hutchinson with low-rank deflation to estimate the trace and/or
# the diagonal. In ``skerch`` we can customize many aspects, including
# how many measurements are performed in each part. Note that the
# low-rank deflation requires ``2 * DEFL_DIMS`` measurements:

hutch1 = hutchpp(
    mat,
    DEVICE,
    DTYPE,
    DEFL_DIMS,
    GH_MEAS,
    seed=SEED + 2 * DIMS,
    noise_type="rademacher",
    meas_blocksize=None,
    return_diag=True,
)
hutch2 = hutchpp(
    lomat,
    DEVICE,
    DTYPE,
    DEFL_DIMS,
    GH_MEAS,
    seed=SEED + 3 * DIMS,
    noise_type="rademacher",
    meas_blocksize=None,
    return_diag=True,
)

tr1, diag1 = hutch1["tr"][0], hutch1["diag"][0]
tr2, diag2 = hutch2["tr"][0], hutch2["diag"][0]


# %%
# We now assess output quality by visually inspecting the diagonals and
# measuring relative errors, observing that both are well below 5%:


def relerr(ori, rec, squared=True):
    """Relative error in the form ``(frob(ori - rec) / frob(ori))**2``."""
    result = (ori - rec).norm() / ori.norm()
    if squared:
        result = result**2
    return result


def relsumerr(ori_sum, rec_sum, ori_vec, squared=True):
    """Relative error of a sum of estimators.

    The error for adding N estimators is bounded by ``sqrt(N)`` times the
    norm of said estimators, because:
    ``(1^T ori) - (1^T rec) = 1^T (ori - rec)``, and the norm of this, by
    Applying Cauchy-Schwarz:
    ``norm(1^T (ori - rec)) <= norm(1)*norm(ori-rec) = sqrt(N)*norm(ori-rec)``.

    So, for the sum of entries, we apply ``relerr``, but divided by ``sqrt(N)``
    to account for this factor:

    ``| ori_sum - rec_sum |`` / (sqrt(N) * norm(ori_vec))``.

    This is consistent in the sense that, if rec_vec is close to ori_vec by
    0.001, this metric will also output at most 0.001 for the estimated sum.
    """
    result = abs(ori_sum - rec_sum) / (len(ori_vec) ** 0.5 * ori_vec.norm())
    if squared:
        result = result**2
    return result


# ground-truth values
mat_diag, lomat_diag = mat.diag(), lomat.diag()
mat_tr, lomat_tr = mat_diag.sum(), lomat_diag.sum()
# relative errors
tr1_err = relsumerr(mat_tr, tr1, mat_diag, squared=False).item()
tr2_err = relsumerr(lomat_tr, tr2, lomat_diag, squared=False).item()
diag1_err = relerr(mat_diag, diag1, squared=False).item()
diag2_err = relerr(lomat_diag, diag2, squared=False).item()

beg, end = 0, 80
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
ax1.plot(mat_diag[beg:end].cpu(), color="black", label="original")
ax1.plot(diag1[beg:end].cpu(), color="pink", linestyle="--", label="approx")
ax1.set_title("Smooth spectral decay")
ax1.legend()
ax2.plot(lomat_diag[beg:end].cpu(), color="black", label="original")
ax2.plot(diag2[beg:end].cpu(), color="pink", linestyle="--", label="approx")
ax2.legend()
ax2.set_title("Steep spectral decay")
fig.suptitle("Hutch++ diagonal approximations for unitary and low-rank linops")
fig.tight_layout()

print("Trace relative error (smooth):", tr1_err)
print("Trace relative error (steep):", tr2_err)
print("Diagonal relative error (smooth):", diag1_err)
print("Diagonal relative error (steep):", diag2_err)


# %%
#
# ##############################################################################
#
# Triangular matrix-vector estimation
# -----------------------------------
#
# Similar in spirit to :func:`skerch.algorithms.hutchpp`,
# :class:`skerch.algorithms.TriangularLinOp` wraps any given linear operator
# (as long as it implements the ``.shape = (height, width)`` attribute and
# the ``@`` matmul operation), and combines deterministic staircase-shaped
# measurements with a modification of Girard-Hutchinson in order to estimate
# triangular matrix-vector products in the form ``tri(lop) @ v``.
# Here we can also customize many aspects, including how many measurements
# are performed in each part:

ltri = TriangularLinOp(
    mat,
    stair_width=max(1, DIMS // 20),
    num_gh_meas=GH_MEAS,
    lower=True,
    with_main_diagonal=False,
    seed=SEED + 4 * DIMS,
    noise_type="rademacher",
)

# ground truth values for triangular matrix product
v = gaussian_noise(DIMS, 0, 1, seed=SEED - 1, dtype=DTYPE, device=DEVICE)
mat_tril = mat.tril(-1)
w1 = mat_tril @ v
w2 = v @ mat_tril

# sketched approximations
ltri_w1 = ltri @ v
ltri_w2 = v @ ltri

# relative errors
w1_err = relerr(w1, ltri_w1, squared=False).item()
w2_err = relerr(w2, ltri_w2, squared=False).item()


beg, end = 0, 100
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
ax1.plot(w1[beg:end].cpu(), color="black", label="original")
ax1.plot(ltri_w1[beg:end].cpu(), color="pink", linestyle="--", label="approx")
ax1.set_title("$tril(A) v$")
ax1.legend()
ax2.plot(w2[beg:end].cpu(), color="black", label="original")
ax2.plot(ltri_w2[beg:end].cpu(), color="pink", linestyle="--", label="approx")
ax2.set_title("$v^T tril(A) $")
ax2.legend()
fig.tight_layout()


print("Lower-triangular relative error:", w1_err)
print("Lower-triangular relative error (adjoint):", w2_err)


# %%
#
# ##############################################################################
#
# And we are done!
#
# * We have seen how to estimate traces, diagonals and triangular matrix
#   multiplications using ``skerch``, and only requiring the *bare-minimum*
#   interface for linear operators
# * We illustrated the effectiveness of low-rank deflation as well as the
#   tendency of Girard-Hutchinson to need more measurements
