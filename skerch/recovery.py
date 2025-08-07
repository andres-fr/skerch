#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """


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
    # note we use .T instead of .conj().T: using conj+T several times is
    # equivalent to just using T, but more expensive (conj may return copy)
    Qh = qr(sketch_left.T, in_place_q=False, return_R=False).T
    B = Qh @ mop_right
    if as_svd:
        P, S = qr(sketch_right, in_place_q=False, return_R=True)
        core = lstsq(B.T, S.T).T
        U, S, Vh = svd(core)
        result = (P @ U), S, (Vh @ Qh)

    else:
        YBinv = lstsq(B.T, sketch_right.T, rcond=rcond).T
        result = YBinv, Qh
    return result


def nystrom(sketch_right, sketch_left, mop_right, rcond=1e-6, as_svd=True):
    """ """
    if not as_svd:
        # the original nystrom recovery, cheaper
        Q, R = qr(sketch_left @ mop_right, in_place_q=False, return_R=True)
        rightRinv = lstsq(R.T, sketch_right.T, rcond=rcond).T
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
    # note we use .T instead of .conj().T: using conj+T several times is
    # equivalent to just using T, but more expensive (conj may return copy)
    P = qr(sketch_right, in_place_q=False, return_R=False)
    Qh = qr(sketch_left.T, in_place_q=False, return_R=False).T
    core = lstsq(lilop @ P, sketch_inner)
    core = lstsq((Qh @ rilop).T, core.T).T
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
    # If the placements of the .conj() don't make sense, note that we
    # leverage symmetries, adding conj in some places and removed from others,
    # to an equivalent result, but less overal scalar conjugations.
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    B = Q.T @ mop_right.conj()
    core = lstsq(B.T, sketch_right.conj().T @ Q).conj().T
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
    if not as_eigh:
        # same as lstsq(sr.conj().T @ mop, sr.conj().T) but cheaper because
        # of less total scalar conjugations
        coreInvSt = lstsq(
            sketch_right.T @ mop_right.conj(), sketch_right.T, rcond=rcond
        ).conj()
        result = sketch_right, coreInvSt
    else:
        P, S = qr(sketch_right, in_place_q=False, return_R=True)
        coreInvSt = lstsq(
            sketch_right.conj().T @ mop_right, S.conj().T, rcond=rcond
        )
        ews, Z = eigh(S @ coreInvSt)
        result = ews, P @ Z
    return result


def oversampled_h(
    sketch_right,
    sketch_inner,
    lilop,
    rilop,
    as_eigh=True,
):
    """ """
    P = qr(sketch_right, in_place_q=False, return_R=False)
    core = lstsq(lilop @ P, sketch_inner)
    # equivalent to lstsq((rilop.conj().T @ P), core.conj().T).conj().T
    # but less total scalar conjugations
    core = lstsq((rilop.T @ P.conj()), core.T).T
    if not as_eigh:
        result = core, P
    else:
        ews, Z = eigh(core)
        result = ews, P @ Z
    return result
