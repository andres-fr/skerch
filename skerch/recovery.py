#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

TODO:
* Implement recoveries for symmetric matrices
  - how to handle truncation?
* extend measurement API to HDF5
  - add HDF5 bells and whistles
  - gonna need to test recovery when mop are linops (we are doing .H and stuff)
at this point, we will have:
- nice lord synthmats
- all linops and linop support we need
- nice measurement API working on parallel single core or HDF5 distributed
- all recovery methods we could care about, for general and symmetric

Remaining:
* add all in-core algorithms
* a-priori/posteriori stuff
* out-of-core wrappers for QR, SVD, LSTSQ

The idea is to
1. Write all in-core algorithms, modularly. code should look very compact
  - Ensure building blocks allow for flawless adaption to HDF5. minimal code
2. Incorporate the priori/posteriori stuff
3. Write integration tests:
  a) Compare all




CHANGELOG:
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByVector
  - New measurement linops: Rademacher, Gaussian, Phase, SSRFT
* Sketching API:
  - Modular measurement API supporting multiprocessing and HDF5
  - Modular recovery methods (singlepass, compact oversampled, Nystrom) for
    general and symmetric cases
  - Algorithms: XDiag/DiagPP, SSVD, Sketchlord, Triangular
* A-posteriori error verification
* A-priori hyperparameter selection
"""


import torch
from .utils import qr, svd, lstsq, eigh, htr


# ##############################################################################
# # RECOVERY FOR GENERAL MATRICES (UV/SVD)
# ##############################################################################
def singlepass(
    sketch_right,
    sketch_left,
    mop_right,
    rcond=1e-6,
    as_svd=True,
):
    r"""Recovering the SVD of a matrix ``A`` from left and right sketches.

    :param sketch_right: Sketches ``A @ measmat_right`` (typically a tall
      matrix).
    :param sketch_left: Sketches ``(measmat_left @ A)`` (typically a fat
      matrix).
    :param mop_right: Right measurement linop.
    :returns: The triple ``U, S, V`` with ``U @ diag(S) @ V.T`` approximating
      ``A``, and ``U, V`` having orthonormal columns.

    Assuming ``A \approx A @ Q @ Q.T``, and given ``Y, W.T, Omega``, where
    ``Y = A @ Omega`` and ``W.T = Psi.T @ A`` are our random measurements,
    we can recover ``A`` without any further measurements.

    First, obtain ``Q`` via QR decomposition of ``W``. Then, it holds

    .. math::

       \begin{align}
       Y  (Q^T \Omega)^{-1} Q^T = (A \Omega) (Q^T \Omega)^{-1} Q^T\\
                        &\approx (A Q Q^T \Omega) (Q^T \Omega)^{-1} Q^T \\
                              &= (A Q)  Q^T \approx A \\
       \end{align}

    Thus, we just need to solve a well-conditioned least-squares problem to
    approximate ``A``. To obtain the full SVD, we further need to compute the
    SVD of ``Y @ pinv(Q.T @ Omega)`` and recombine.

    Reference: `[TYUC2018, 4.1] <https://arxiv.org/abs/1609.00048>`_)
    """
    Qh = qr(sketch_left.conj().T, in_place_q=False, return_R=False).conj().T
    B = Qh @ mop_right
    if as_svd:
        P, S = qr(sketch_right, in_place_q=False, return_R=True)
        core = lstsq(B.conj().T, S.conj().T).conj().T
        U, S, Vh = svd(core)
        result = (P @ U), S, (Vh @ Qh)

    else:
        YBinv = lstsq(B.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        result = YBinv, Qh
    return result


def nystrom(sketch_right, sketch_left, mop_right, rcond=1e-6, as_svd=True):
    """ """
    if not as_svd:
        # the original nystrom recovery, cheaper
        Q, R = qr(sketch_left @ mop_right, in_place_q=False, return_R=True)
        rightRinv = (
            lstsq(R.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        )
        result = rightRinv, (Q.conj().T @ sketch_left)  # U, Vh
    else:
        # return in SVD form, more expensive
        # orthogonalization of sketches is needed
        P1, S1 = qr(sketch_right, in_place_q=False, return_R=True)
        P2, S2 = qr(sketch_left.conj().T, in_place_q=False, return_R=True)
        # now invert the Nystrom core upon S2 and compute a small SVD with S1
        coreInvS2t = lstsq(sketch_left @ mop_right, S2.conj().T, rcond=rcond)
        U, S, Vh = svd(S1 @ coreInvS2t)
        result = (P1 @ U), S, (Vh @ P2.conj().T)
    return result


def oversampled(
    sketch_right,
    sketch_left,
    sketch_inner,
    lilop,
    rilop,
    as_svd=True,
):
    """ """
    P = qr(sketch_right, in_place_q=False, return_R=False)
    Qh = qr(sketch_left.conj().T, in_place_q=False, return_R=False).conj().T
    core = lstsq(lilop @ P, sketch_inner)
    core = lstsq((rilop.T @ Qh.T), core.T).T
    if as_svd:
        U, S, Vh = svd(core)
        result = (P @ U), S, (Vh @ Qh)
    else:
        result = P, core @ Qh
    return result


# ##############################################################################
# # RECOVERY FOR SYMMETRIC MATRICES (EIGH)
# ##############################################################################
def singlepass_h(
    sketch_right,
    mop_right,
    rcond=1e-6,
    as_eigh=True,
):
    r""" """
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    B = Q.conj().T @ mop_right
    core = lstsq(B.conj().T, sketch_right.conj().T @ Q).conj().T
    if not as_eigh:
        result = core, Q
    else:
        ews, Z = eigh(core)
        result = ews, Q @ Z
    return result


def nystrom_h(
    sketch_right,
    mop_right,
    rcond=1e-6,
    as_eigh=True,
):
    """ """
    Q, R = qr(sketch_right @ mop_right, in_place_q=False, return_R=True)

    B = Q.conj().T @ mop_right
    core = htr(lstsq(B.conj().T, sketch_right.conj().T @ Q), inplace=True)

    if not as_eigh:
        result = NotImplemented
    else:
        result = NotImplemented
    return result
    breakpoint()


# def nystrom(sketch_right, sketch_left, mop_right, rcond=1e-6, as_svd=True):
#     """ """
#     if not as_svd:
#         # the original nystrom recovery, cheaper
#         Q, R = qr(sketch_left @ mop_right, in_place_q=False, return_R=True)
#         rightRinv = (
#             lstsq(R.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
#         )
#         result = rightRinv, (Q.conj().T @ sketch_left)  # U, Vh
#     else:
#         # return in SVD form, more expensive
#         # orthogonalization of sketches is needed
#         P1, S1 = qr(sketch_right, in_place_q=False, return_R=True)
#         P2, S2 = qr(sketch_left.conj().T, in_place_q=False, return_R=True)
#         # now invert the Nystrom core upon S2 and compute a small SVD with S1
#         coreInvS2t = lstsq(sketch_left @ mop_right, S2.conj().T, rcond=rcond)
#         U, S, Vh = svd(S1 @ coreInvS2t)
#         result = (P1 @ U), S, (Vh @ P2.conj().T)
#     return result


def oversampled_h():
    pass
