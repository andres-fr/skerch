# -*- coding: utf-8 -*-

r"""Sketched Triangular Operators
=================================

In this tutorial we create a noisy matrix and perform sketched estimations of
matrix-vector multiplications with its lower and upper triangles.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from skerch.synthmat import SynthMat
from skerch.triangles import TriangularLinOp
from skerch.utils import gaussian_noise

# %%
#
# ##############################################################################
#
# Globals and creation of test matrix
# -----------------------------------
#
# We will create a random matrix of shape ``(ORDER, ORDER)``, and a test vector
# with Gaussian i.i.d. noise (the other global parameters can be ignored).

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
ORDER, RANK = 1000, 100
EXP_DECAY = 0.2
SEED = 12345

v = gaussian_noise(ORDER, seed=SEED - 1, dtype=DTYPE, device=DEVICE)

linop = SynthMat.exp_decay(
    shape=(ORDER, ORDER),
    rank=RANK,
    decay=EXP_DECAY,
    symmetric=False,
    seed=SEED,
    dtype=DTYPE,
    device=DEVICE,
    psd=False,
)

tril = torch.tril(linop, diagonal=0)
triu = torch.triu(linop, diagonal=0)

# test measurements
lo_v = tril @ v
v_lo = v @ tril
up_v = triu @ v
v_up = v @ triu


# %%
#
# ##############################################################################
#
# Sketched triangular operators
# -----------------------------
#
# To estimate triangular matrix-vector products, these linear operators perform
# multiple measurements (as opposed to just one if we knew the exact triangular
# operator). These measurements have two very distinct natures:
#
# * The "step-wise" measurements are exact, and measure blocks that are fully
#   inside of the triangle
# * The Hutchinson measurements are a modification of the Hutchinson diagonal
#   estimator, to estimate the product with the block-triangular patterns that
#   were ignored by the exact step-wise measurements.
#
# For this reason, we want to have as many exact measurements as we can afford.
# The Hutchinson measurements, much more inefficient, will be necessary if a
# relevant part of the linop is concentrated near the main diagonal.

tril_est = TriangularLinOp(
    linop,
    stair_width=(ORDER // 20),  # we do approx 20 exact measurements
    num_hutch_measurements=(ORDER // 5),  # Hutch works better for larger order
    lower=True,
    with_main_diagonal=True,
)

triu_est = TriangularLinOp(
    linop,
    stair_width=(ORDER // 20),
    num_hutch_measurements=(ORDER // 5),
    lower=False,
    with_main_diagonal=True,
)


lo_v_est = tril_est @ v
v_lo_est = v @ tril_est
up_v_est = triu_est @ v
v_up_est = v @ triu_est


fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(
    triu.cpu()[:50, :50],
    cmap="bwr",
    aspect="auto",
    norm=mpl.colors.CenteredNorm(),
)
ax2.plot(up_v.cpu()[:50], color="black", label="original")
ax2.plot(up_v_est.cpu()[:50], color="pink", label="approximation")
ax2.legend()
fig.suptitle("Upper Triangular and $triu(A)~v$ (detail)")
fig.tight_layout()
#

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(
    tril.cpu()[:, :], cmap="bwr", aspect="auto", norm=mpl.colors.CenteredNorm()
)
ax2.plot(lo_v.cpu()[:], color="black", label="original")
ax2.plot(lo_v_est.cpu()[:], color="pink", label="approximation")
ax2.legend()
fig.suptitle("Lower Triangular and $tril(A)~v$")
fig.tight_layout()
#
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(
    tril.cpu()[:, :], cmap="bwr", aspect="auto", norm=mpl.colors.CenteredNorm()
)
ax2.plot(v_lo.cpu()[:], color="black", label="original")
ax2.plot(v_lo_est.cpu()[:], color="pink", label="approximation")
ax2.legend()
fig.suptitle("Lower Triangular and $v~tril(A)$")
fig.tight_layout()
#
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(
    triu.cpu()[:, :], cmap="bwr", aspect="auto", norm=mpl.colors.CenteredNorm()
)
ax2.plot(up_v.cpu()[:], color="black", label="original")
ax2.plot(up_v_est.cpu()[:], color="pink", label="approximation")
ax2.legend()
fig.suptitle("Upper Triangular and $triu(A)~v$")
fig.tight_layout()
#
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(
    triu.cpu()[:, :], cmap="bwr", aspect="auto", norm=mpl.colors.CenteredNorm()
)
ax2.plot(v_up.cpu()[:], color="black", label="original")
ax2.plot(v_up_est.cpu()[:], color="pink", label="approximation")
ax2.legend()
fig.suptitle("Upper Triangular and $v~triu(A)$")
fig.tight_layout()
