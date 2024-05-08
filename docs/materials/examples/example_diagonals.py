# -*- coding: utf-8 -*-

r"""Sketched (Sub-)Diagonals
============================

In this tutorial we create a noisy, numerically low-rank matrix,
and perform sketched estimations of its main diagonal and subdiagonals.
"""

import matplotlib.pyplot as plt
import torch

from skerch.subdiagonals import subdiagpp
from skerch.synthmat import SynthMat

# %%
#
# ##############################################################################
#
# Globals and creation of test matrix
# -----------------------------------
#
# We will create a random matrix of shape ``(ORDER, ORDER)``, with an effective
# ``RANK`` of unit singular values followed by exponential decay with rate
# ``EXP_DECAY``.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
ORDER, RANK = 1000, 100
EXP_DECAY = 0.2
SEED = 12345


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

# %%
#
# ##############################################################################
#
# Exact (sub-)diagonals
# ---------------------
#
# For testing purposes, we extract the exact main diagonal, as well as the
# sub-diagonals right above and below

main_diag = torch.diag(linop, diagonal=0)
sub_diag = torch.diag(linop, diagonal=-1)
super_diag = torch.diag(linop, diagonal=1)

print("Norm of main diagonal:", main_diag.norm().item())
print("Norm of lower subdiagonal:", sub_diag.norm().item())
print("Norm of upper subdiagonal:", super_diag.norm().item())

# %%
#
# ##############################################################################
#
# Sketched approximation of main diagonal
# ---------------------------------------
#
# for this initial approximation we do not deflate, and just rely on the
# Hutchinson estimator:

main_diag_est, _, (top_norm, bottom_norm) = subdiagpp(
    linop,
    int(ORDER * 0.5),
    DTYPE,
    DEVICE,
    SEED + 1,
    0,
    0,
)

fig, ax = plt.subplots()
ax.plot(main_diag.cpu()[:100], color="black", label="original")
ax.plot(main_diag_est.cpu()[:100], color="pink", label="approximation")
ax.legend()
fig.tight_layout()

print(
    "Norm of residual (main diagonal):",
    torch.dist(main_diag, main_diag_est).item(),
)

# %%
#
# Since we are not deflating, we confirm that all of the obtained diagonal
# comes from the deflated component

print("Norm of top-rank component:", top_norm)
print("Norm of deflated component:", bottom_norm)

# %%
#
# ##############################################################################
#
# Sketched approximation of subdiagonals
# --------------------------------------
#
# Now we do the same for the neighboring subdiagonals

sub_diag_est, _, (top_norm, bottom_norm) = subdiagpp(
    linop,
    int(ORDER * 0.5),
    DTYPE,
    DEVICE,
    SEED + 1,
    0,
    -1,
)

super_diag_est, _, (top_norm, bottom_norm) = subdiagpp(
    linop,
    int(ORDER * 0.5),
    DTYPE,
    DEVICE,
    SEED + 1,
    0,
    1,
)

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(super_diag.cpu()[:100], color="black")
ax1.plot(super_diag_est.cpu()[:100], color="pink", alpha=0.8)
ax2.plot(sub_diag.cpu()[:100], color="black")
ax2.plot(sub_diag_est.cpu()[:100], color="pink", alpha=0.8)
fig.tight_layout()

print(
    "Norm of residual (upper subdiagonal):",
    torch.dist(super_diag, super_diag_est).item(),
)
print(
    "Norm of residual (lower subdiagonal):",
    torch.dist(sub_diag, sub_diag_est).item(),
)

# %%
#
# ##############################################################################
#
# Deflation
# ---------
#
# Since our test matrix can be well-approximated with low rank, deflation can
# help a lot. Here we combine deflation with Hutchinson measurements, and show
# that the recovery is much better, and that most of it is due to the deflation.

main_diag_est, defl_mat, (top_norm, bottom_norm) = subdiagpp(
    linop,
    int(ORDER * 0.5),
    DTYPE,
    DEVICE,
    SEED + 1,
    RANK,
    0,
)

fig, ax = plt.subplots()
ax.plot(main_diag.cpu()[:100], color="black")
ax.plot(main_diag_est.cpu()[:100], color="pink")
fig.tight_layout()

print(
    "Norm of residual (main diagonal):",
    torch.dist(main_diag, main_diag_est).item(),
)
print("Shape of deflation matrix:", tuple(defl_mat.shape))
print("Norm of top-rank component:", top_norm.item())
print("Norm of deflated component:", bottom_norm)
