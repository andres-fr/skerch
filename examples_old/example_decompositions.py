# -*- coding: utf-8 -*-

r"""Sketched Decompositions
===========================

In this tutorial we create noisy, numerically low-rank matrices,
and compare their exact decomposition with the sketched one.

We do this for arbitrary matrices (SVD), as well as symmetric
ones (Hermitian EIGH), both PSD and non-PSD.

We also show how to assess rank and quality of the recovery in more difficult
scenarios. For that, a-posteriori error estimation methods are used.
"""

import matplotlib.pyplot as plt
import torch

from skerch.a_posteriori import (
    a_posteriori_error,
    a_posteriori_error_bounds,
    scree_bounds,
)
from skerch.decompositions import seigh, ssvd, truncate_core
from skerch.linops import CompositeLinOp, DiagonalLinOp
from skerch.synthmat import SynthMat

# %%
#
# ##############################################################################
#
# Global variables and hyperparameters
# ------------------------------------
#
# ``OUTER`` and ``INNER`` reflect the number of sketches to take. Typically,
# ``INNER`` is twice ``OUTER``, which is itself larger than ``RANK`` by a
# constant (see `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_).
#
# Despite the availability of *a priori* methods to estimate the number of
# measurements, (see :mod:`skerch.a_priori`), ``RANK`` is typically unknown
# and we resort to *a posteriori* methods, which estimate the quality of
# the approximation as a function of the recovered rank using
# ``NUM_A_POSTERIORI`` measurements.
#
# Last but not least, recovered singular vectors can get noisy when the
# associated singular values get closer to zero. Generally, this is tackled
# via truncation of the core matrices down to ``TRUNCATE`` dimensions.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
SHAPE, RANK = (200, 200), 10
OUTER, INNER, TRUNCATE = 30, 60, 20
NUM_A_POSTERIORI = 50
EXP_DECAY = 0.2

# %%
#
# ##############################################################################
#
# Creation of test data
# ---------------------
#
# Now we create three random, rank-defficient matrices, symmetric
# PSD and non-PSD (see :mod:`skerch.synthmat`). Then compute their exact
# decompositions using ``torch``:

linop_asym = SynthMat.exp_decay(
    SHAPE, RANK, EXP_DECAY, symmetric=False, device=DEVICE, dtype=DTYPE
)
linop_sym = SynthMat.exp_decay(
    SHAPE,
    RANK,
    EXP_DECAY,
    symmetric=True,
    device=DEVICE,
    dtype=DTYPE,
    psd=False,
)
linop_psd = SynthMat.exp_decay(
    SHAPE, RANK, EXP_DECAY, symmetric=True, device=DEVICE, dtype=DTYPE, psd=True
)

U_asym, S_asym, Vt_asym = torch.linalg.svd(linop_asym)
S_sym, U_sym = torch.linalg.eigh(linop_sym)
S_psd, U_psd = torch.linalg.eigh(linop_psd)

# %%
#
# ##############################################################################
#
# Computation of sketched approximations
# --------------------------------------
#
# We can now conveniently compute and truncate in-core sketched decompositions
# using functionality from :mod:`skerch.decompositions`. We can also efficiently
# compose the recovered thin matrices into the original shape (see documentation
# of :class:`skerch.linops.CompositeLinOp` for more details).

q_asym, u_asym, s_asym, vt_asym, pt_asym = ssvd(
    linop_asym,
    op_device=DEVICE,
    op_dtype=DTYPE,
    outer_dim=OUTER,
    inner_dim=INNER,
)
u_asym, s_asym, vt_asym = truncate_core(TRUNCATE, u_asym, s_asym, vt_asym)
appr_asym = CompositeLinOp(
    (
        ("Q", q_asym),
        ("U", u_asym),
        ("S", DiagonalLinOp(s_asym)),
        ("Vt", vt_asym),
        ("Pt", pt_asym),
    )
)

q_sym, u_sym, s_sym = seigh(
    linop_sym,
    op_device=DEVICE,
    op_dtype=DTYPE,
    outer_dim=OUTER,
    inner_dim=INNER,
)
u_sym, s_sym = truncate_core(TRUNCATE, u_sym, s_sym)
appr_sym = CompositeLinOp(
    (
        ("Q", q_sym),
        ("U", u_sym),
        ("S", DiagonalLinOp(s_sym)),
        ("Ut", u_sym.T),
        ("Qt", q_sym.T),
    )
)

q_psd, u_psd, s_psd = seigh(
    linop_psd,
    op_device=DEVICE,
    op_dtype=DTYPE,
    outer_dim=OUTER,
    inner_dim=INNER,
)
u_psd, s_psd = truncate_core(TRUNCATE, u_psd, s_psd)
appr_psd = CompositeLinOp(
    (
        ("Q", q_psd),
        ("U", u_psd),
        ("S", DiagonalLinOp(s_psd)),
        ("Ut", u_psd.T),
        ("Qt", q_psd.T),
    )
)

# %%
#
# ##############################################################################
#
# Exact error analysis
# -------------------->
#
# Since we are working with small matrices, we can here naïvely compare
# original matrix and reconstruction in terms of their Frobenius error, and
# even plot them:

naive_asym = q_asym @ u_asym @ DiagonalLinOp(s_asym) @ vt_asym @ pt_asym
naive_sym = q_sym @ u_sym @ DiagonalLinOp(s_sym) @ u_sym.T @ q_sym.T
naive_psd = q_psd @ u_psd @ DiagonalLinOp(s_psd) @ u_psd.T @ q_psd.T

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(linop_asym.cpu()[:50, :50], cmap="bwr")
ax2.imshow(naive_asym.cpu()[:50, :50], cmap="bwr")
fig.tight_layout()

print("Frobenius error (asym):", torch.dist(linop_asym, naive_asym).item())
print("Frobenius error (sym):", torch.dist(linop_sym, naive_sym).item())
print("Frobenius error (psd):", torch.dist(linop_psd, naive_psd).item())
print(
    "Singular value error (asym):",
    torch.dist(S_asym[:TRUNCATE], s_asym[:TRUNCATE]).item(),
)
# Note that torch.eigh provides eigenpairs in ascending value, but we want
# them by descending magnitude
S_sym_descmag = S_sym[abs(S_sym).argsort(descending=True)]
print(
    "Eigenvalue error (sym):",
    torch.dist(S_sym_descmag[:TRUNCATE], s_sym[:TRUNCATE]).item(),
)
S_psd_descmag = S_psd[abs(S_psd).argsort(descending=True)]
print(
    "Eigenvalue error (psd):",
    torch.dist(S_psd_descmag[:TRUNCATE], s_psd[:TRUNCATE]).item(),
)

# %%
#
# ##############################################################################
#
# Approximate error analysis
# --------------------------
#
# When the linear operator is too large or matrix-free, an exact error analysis
# is not always possible, and we resort to approximate *a posteriori* methods
# (see :mod:`skerch.a_posteriori` for more details).
#
# First, we estimate the Frobenius error between the orginal and the sketched
# operators. Note how this does not require an explicit reconstruction.

(f1_asym, f2_asym, frob_err_asym) = a_posteriori_error(
    linop_asym, appr_asym, NUM_A_POSTERIORI, dtype=DTYPE, device=DEVICE
)[0]
print("Estimated Frobenius Error (asym):", frob_err_asym**0.5)

(f1_sym, f2_sym, frob_err_sym) = a_posteriori_error(
    linop_sym, appr_sym, NUM_A_POSTERIORI, dtype=DTYPE, device=DEVICE
)[0]
print("Estimated Frobenius Error (sym):", frob_err_sym**0.5)

(f1_psd, f2_psd, frob_err_psd) = a_posteriori_error(
    linop_psd, appr_psd, NUM_A_POSTERIORI, dtype=DTYPE, device=DEVICE
)[0]
print("Estimated Frobenius Error (psd):", frob_err_psd**0.5)


# %%
#
# We see that the approximate errors are very close to the previously computed
# exact ones. The probability of this not happening, for any given number
# of *a posteriori* measurements and error tolerance (in this case 50%), can
# be obtained as follows:

a_posteriori_error_bounds(NUM_A_POSTERIORI, 0.5)


# %%
#
# ##############################################################################
#
# *A posteriori* rank estimation
# ------------------------------
#
# With the quantities obtained so far, we can compute the *scree* bounds, which
# tell us how much of the Frobenius norm of the original operator are we
# actually recovering, as a function of increasing number of dimensions.
#
# This information can be used to estimate the effective rank of the original
# operator, e.g. by looking for "elbows" in the scree curve. Since in this
# case we have a simple synthetic matrix, we can see how the scree bounds
# accurately and clearly reflect the given ``RANK`` of the original operator.
#
# .. warning::
#
#   This functionality has not been unit-tested, use with care. Contributions
#   are welcome :)

scree_lo, scree_hi = scree_bounds(s_asym, f1_asym**0.5, frob_err_asym**0.5)
plt.plot(scree_lo.cpu())
plt.plot(scree_hi.cpu())


# %%
#
# ##############################################################################
#
# And we are done!
#
# * We have seen how to approximate singular (eigen-) decompositions via
#   in-core sketched methods for a broad class of low-rank matrices
# * We verified that said approximations are actually close to the original
#   ones
# * We demonstrated matrix-free, scalable methods to also verify the quality
#   of the approximation and the rank of the original operators
