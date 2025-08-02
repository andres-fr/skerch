#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

TODO:
* Implement 3 recovery methods
  - test correctness and formal
* Implement all sketched algorithms as meas-recovery
* HDF5?


* check that all 3 are close enough?? how?
* check that they work on general matrices



CHANGELOG:
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByVector
  - New measurement linops: Rademacher, Gaussian, Phase, SSRFT
* Sketching API:
  - Modular measurement API supporting MP parallelization
  - Modular recovery methods (singlepass, oversampled, Nystrom)
  - Algorithms: XDiag/DiagPP, SSVD, Sketchlord, Triangular
"""

import torch

from .linops import TransposedLinOp
from .utils import qr, svd

# ##############################################################################
# # HELPERS
# ##############################################################################


# ##############################################################################
# # SINGLE-PASS
# ##############################################################################


def singlepass(
    sketch_right,
    sketch_left,
    mop_right,
):
    """Recovering the SVD of a matrix ``A`` from left and right sketches.

    :param sketch_right: Sketches ``A @ measmat_right``.
    :param sketch_left: Sketches ``measmat_left @ A``.
    :param mop_right: Right measurement linop.
    :returns: The triple ``U, S, V`` with ``U @ diag(S) @ V.T`` approximating
      ``A``, and ``U, V`` having orthonormal columns.

    Assuming ``A \approx A @ Q @ Q.T``, and given ``Y, W.T, Omega``, where
    ``Y = A @ Omega`` and ``W.T = Psi.T @ A`` are our random measurements,
    we can recover ``A`` without any further measurements.

    First, obtain ``Q`` via QR decomposition of ``Y``. Then, it holds

    .. math::

       \begin{align}
       Y @ (Q^T \Omega)^{-1} Q^T = (A \Omega) (Q^T \Omega)^{-1} Q^T\\
                        &\approx (A Q Q^T \Omega) (Q^T \Omega)^{-1} Q^T \\
                              &= (A Q)  Q^T \approx A \\
       \end{align}

    Thus, we just need to solve a well-conditioned least-squares problem to
    approximate ``A``. To obtain the full SVD, we further need to compute the
    SVD of ``Y @ pinv(Q.T @ Omega)`` and recombine.

    Reference: `[TYUC2018, 4.1] <https://arxiv.org/abs/1609.00048>`_)
    """
    Q = qr(sketch_left)
    B = Q @ mop_right
    YBinv = lstsq(B.H, sketch_right.H).H
    U, S, Vh = svd(YBinv)
    return U, S, (Q @ Vh.H)

    # breakpoint()
    # Q, R = torch.linalg.qr(sketch_left)  # Q spans V
    # B = Q.T @ measmat_right
    # YBinv = torch.linalg.lstsq(B.T, sketch_right.T).solution.T  # S @ inv(B)
    # U, Sigma, Zt = torch.linalg.svd(YBinv, full_matrices=False)
    # return U, Sigma, (Q @ Zt.T)


# ##############################################################################
# # OVERSAMPLED
# ##############################################################################
def ___oversampled_recovery(
    lop,
    num_meas,
    sketch1,
    sketch2,
    trunc_ratio=0.99,
    trunc_overhead=5,
    li_seed=123456,
    ri_seed=123457,
):
    """
    lop, diag, inner_dim, sketch_right, sketch_left, seed
    """
    h, w = lop.shape
    device, dtype = sketch1.device, sketch1.dtype
    # perform core measurements
    li_lop = RademacherIidLinOp((num_meas, h), li_seed, partition="row")
    ri_lop = RademacherIidLinOp((num_meas, w), ri_seed, partition="row")
    meas = torch.empty((num_meas, num_meas), dtype=dtype, device=device)
    for i in range(num_meas):
        meas[i, :] = ri_lop @ (li_lop.sample(h, i, device).to(dtype) @ lop)
    # Solve core op to yield initial approximation
    sketch1 = torch.linalg.qr(sketch1)[0]
    sketch2 = torch.linalg.qr(sketch2)[0]
    u, Sigma, vt = solve_core_ssvd(sketch1, sketch2, meas, li_lop, ri_lop)
    idx = torch.searchsorted(
        pca_scree(Sigma, normalized=True), trunc_ratio
    ).item()
    idx += trunc_overhead
    u, Sigma, vt = u[:, :idx], Sigma[:idx], vt[:idx]
    # collapse result and return
    U = sketch1 @ u
    V = sketch2 @ vt.T
    return U, Sigma, V


# ##############################################################################
# # GENERALIZED NYSTROM
# ##############################################################################
