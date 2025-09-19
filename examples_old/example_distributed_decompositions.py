# -*- coding: utf-8 -*-

r"""Distributed Sketched Decompositions
=======================================

In some cases, the main bottleneck to work with a linear operator :math:`A`
is performing linear measurements in the form :math:`w := A v`.

Luckily for us, sketched methods rely on multiple independent vectors
:math:`v_i`, which means that we can parallelize the measurements.

Another related bottleneck is that multiple processes may not be able to write
their  :math:`w_i` measurements to the same file concurrently, and if they
write to separate files, subsequent processing may require to merge them.

This tutorial showcases ``skerch`` functionality to overcome both these
problems and compute a distributed sketched SVD (DSSVD).

.. note::

  Readers are encouraged to read the :ref:`Sketched Decompositions` tutorial
  first and grasp the fundamental quantities and steps involved.
  This tutorial is a follow-up since it builds on top and naturally involves
  more implementation and algorithmic details, due to the distributed and
  system-agnostic nature of the interfaces.


Of course, all of the ``skerch`` functionality introduced in the other tutorials
can be applied to the distributed scenario, including *a priori* and
*a posteriori* methods. But to keep things simple, here we will:

* Compute the exact and sketched SVD of an asymmetric matrix
* Check that Frobenius error between original and recovery is low
* Check that :math:`\ell_2` error between original and recovered singular values
  is low

Let's start with slightly different imports: instead of directly importing
:func:`skerch.decompositions.ssvd`, we import its components:
"""


import tempfile

import matplotlib.pyplot as plt
import torch

from skerch.decompositions import truncate_core
from skerch.distributed_decompositions import (
    DistributedHDF5,
    create_hdf5_layout,
    orthogonalize,
    solve_core_ssvd,
)
from skerch.distributed_measurements import innermeas_idx_torch, ssrft_idx_torch
from skerch.ssrft import SSRFT
from skerch.synthmat import SynthMat

# %%
#
# ##############################################################################
#
# As before, we create and decompose a random test matrix.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
SHAPE, RANK = (200, 222), 10
OUTER, INNER, TRUNCATE = 30, 60, 20
NUM_A_POSTERIORI = 50
EXP_DECAY = 0.2
PROCESSING_FLAG, SUCCESS_FLAG = "0", "1"
SEED = 12345

lop = SynthMat.exp_decay(
    SHAPE,
    RANK,
    EXP_DECAY,
    symmetric=False,
    device=DEVICE,
    dtype=DTYPE,
    seed=SEED,
)
U_lop, S_lop, Vt_lop = torch.linalg.svd(lop)


# %%
#
# ##############################################################################
#
# In a distributed system, we assume that each measurement is performed and
# stored on a separate unit. This is handled by creating a virtual HDF5
# dataset (see :class:`skerch.distributed_measurements.DistributedHDF5` for
# more details).

tmpdir = tempfile.TemporaryDirectory()
(
    (lo_path, lo_subpaths),
    (ro_path, ro_subpaths),
    (inn_path, inn_subpaths),
) = create_hdf5_layout(
    tmpdir.name,
    SHAPE,
    lop.dtype,
    OUTER,
    INNER,
    lo_fmt="leftouter_{}.h5",
    ro_fmt="rightouter_{}.h5",
    inner_fmt="core_{}.h5",
    with_ro=True,
)


# %%
#
# ##############################################################################
#
# Now that we have our HDF5 layout in place, we can perform the distributed
# measurements. Although we have ``for`` loops here, you can imagine that
# each of the iterations is running in a separate process or device.
# They will be able to write concurrently to the distributed HDF5 layout.
#
# The loop over :func:`skerch.distributed_measurements.ssrft_idx_torch` performs
# a matrix-free random SSRFT projection :math:`A \Phi`, where :math:`A` is our
# original linear operator and each iteration is a column of the random
# projection :math:`\Phi` (see :mod:`skerch.ssrft` for more details).
#
# The :func:`skerch.distributed_measurements.innermeas_idx_torch` function is a
# slight variation in which each :math:`i^{th}` loop iteration computes
# :math:`\Psi^* A \Phi_i`, where :math:`\Psi` and :math:`\Phi` are SSRFT
# projections with different seeds (i.e. uncorrelated to each other).

# SSRFT random seeds
lo_seed, ro_seed, li_seed, ri_seed = SEED + 1, SEED + 2, SEED + 3, SEED + 4

# left outer measurements
for i in range(OUTER):
    vals, flag, h5 = DistributedHDF5.load(lo_subpaths[i])
    ssrft_idx_torch(
        i,
        OUTER,
        lop,
        DEVICE,
        DTYPE,
        vals,
        lo_seed,
        flag=flag,
        processing_flag=PROCESSING_FLAG,
        success_flag=SUCCESS_FLAG,
        adjoint=True,
    )
    # check that file has been successfully written to
    assert flag[0].decode() == SUCCESS_FLAG, f"Unsuccessful measurement? {i}"
    h5.close()

# right outer measurements
for i in range(OUTER):
    vals, flag, h5 = DistributedHDF5.load(ro_subpaths[i])
    ssrft_idx_torch(
        i,
        OUTER,
        lop,
        DEVICE,
        DTYPE,
        vals,
        ro_seed,
        flag=flag,
        processing_flag=PROCESSING_FLAG,
        success_flag=SUCCESS_FLAG,
        adjoint=False,
    )
    # check that file has been successfully written to
    assert flag[0].decode() == SUCCESS_FLAG, f"Unsuccessful measurement? {i}"
    h5.close()

# inner measurements
for i in range(INNER):
    vals, flag, h5 = DistributedHDF5.load(inn_subpaths[i])
    innermeas_idx_torch(
        i,
        INNER,
        lop,
        DEVICE,
        DTYPE,
        vals,
        li_seed,
        ri_seed,
        flag=flag,
        processing_flag=PROCESSING_FLAG,
        success_flag=SUCCESS_FLAG,
    )
    # check that file has been successfully written to
    assert flag[0].decode() == SUCCESS_FLAG, f"Unsuccessful measurement? {i}"
    h5.close()


# %%
#
# Once we are done with the measurements, merge the respective HDF5 files into
# a centralized one. This is to avoid OS errors (see
# :class:`skerch.distributed_measurements.DistributedHDF5` for an
# explanation).
#
# Since we are deleting each measurement after merging it, there is no
# substantial memory overhead and we simply "repack" existing data.

_ = DistributedHDF5.merge_all(
    lo_path,
    out_path=None,
    delete_subfiles_while_merging=True,
    check_success_flag=SUCCESS_FLAG,
)
_ = DistributedHDF5.merge_all(
    ro_path,
    out_path=None,
    delete_subfiles_while_merging=True,
    check_success_flag=SUCCESS_FLAG,
)
_ = DistributedHDF5.merge_all(
    inn_path,
    out_path=None,
    delete_subfiles_while_merging=True,
    check_success_flag=SUCCESS_FLAG,
)


# %%
#
# ##############################################################################
#
# The most conceptually involved step in the sketched SVD is to "solve the core
# matrix". This is done in a centralized manner by taking the QR decomposition
# of the outer measurements and solving two least squares problems against
# the inner measurements (see `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_).
#
# To speed things up, and in the lack of an out-of-core QR routine, this needs
# to be done on a single computer with sufficient RAM to perform the QR step.

# load all measurements and check that all flags are OK
lo, lo_flags, lo_h5 = DistributedHDF5.load(lo_path, filemode="r+")
ro, ro_flags, ro_h5 = DistributedHDF5.load(ro_path, filemode="r+")
inn, inn_flags, inn_h5 = DistributedHDF5.load(inn_path, filemode="r+")
assert all(
    f.decode() == SUCCESS_FLAG for f in lo_flags
), "Bad left outer flags!"
assert all(
    f.decode() == SUCCESS_FLAG for f in ro_flags
), "Bad right outer flags!"
assert all(f.decode() == SUCCESS_FLAG for f in inn_flags), "Bad inner flags!"

# orthogonalize outer measurements (in-place)
orthogonalize(lo, overwrite=True)
orthogonalize(ro, overwrite=True)

# Solve core op and decompose
left_inner_ssrft = SSRFT((INNER, SHAPE[0]), seed=li_seed)
right_inner_ssrft = SSRFT((INNER, SHAPE[1]), seed=ri_seed)
core_U, core_S, core_Vt = solve_core_ssvd(
    ro, lo, inn, left_inner_ssrft, right_inner_ssrft
)
# for this utest, convert everything to explicit torch tensors
ro = torch.from_numpy(ro[:]).to(DEVICE)
lo = torch.from_numpy(lo[:]).to(DEVICE)
core_U = torch.from_numpy(core_U).to(DEVICE)
core_S = torch.from_numpy(core_S).to(DEVICE)
core_Vt = torch.from_numpy(core_Vt).to(DEVICE)
core_U, core_S, core_Vt = truncate_core(TRUNCATE, core_U, core_S, core_Vt)
# close HDF5 files and we are done
lo_h5.close()
ro_h5.close()
inn_h5.close()


# %%
#
# ##############################################################################
#
# At this point we are done with the sketched SVD! To naively test the quality
# of the recovered decomposition, we measure the error overall and for the
# singular values, verifying that they are low:

appr_lop = ro @ (core_U @ torch.diag(core_S) @ core_Vt) @ lo.T

# plot part of original and recovery
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(lop.cpu()[:50, :50], cmap="bwr")
ax2.imshow(appr_lop.cpu()[:50, :50], cmap="bwr")
fig.tight_layout()

print("Frobenius error:", torch.dist(lop, appr_lop).item())
print(
    "Singular value error:",
    torch.dist(S_lop[:TRUNCATE], core_S[:TRUNCATE]).item(),
)

# Done!
tmpdir.cleanup()
