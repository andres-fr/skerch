# -*- coding: utf-8 -*-

r"""Sketched (Sub-)Diagonals
============================

In this tutorial we create a noisy, numerically low-rank matrix,
and perform sketched estimations of its main diagonal and subdiagonals.
"""

import matplotlib.pyplot as plt
import torch

from skerch.subdiagonals import diagpp
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
# ---------------
#
# For testing purposes, we extract the exact main diagonal, as well as the
# sub-diagonals right above and below

main_diag = torch.diag(linop, diagonal=0)
sub_diag = torch.diag(linop, diagonal=-1)
super_diag = torch.diag(linop, diagonal=1)

# %%
#
# ##############################################################################
#
# Computation of sketched approximations
# --------------------------------------
#
# We can now JUST HUTCHINSON

main_diag_est, _, (top_norm, bottom_norm) = subdiagpp(
    linop,
    int(ORDER * 0.5),
    DTYPE,
    DEVICE,
    SEED + 1,
    0,
    0,
)


fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(main_diag.cpu()[:100], color="black")
ax2.imshow(main_diag_est.cpu()[:100], color="pink")
fig.tight_layout()

# %%
#
# Since we are not deflating, we confirm that all of the obtained diagonal
# comes from the deflated component

print("Norm of top-rank component:", top_norm)
print("Norm of deflated component:", bottom_norm)
