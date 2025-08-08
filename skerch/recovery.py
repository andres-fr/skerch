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
    P, S = qr(sketch_right, in_place_q=False, return_R=True)
    if as_svd:
        core = lstsq(B.T, S.T, rcond=rcond).T
        U, S, Vh = svd(core)
        result = (P @ U), S, (Vh @ Qh)

    else:
        # equivalent to sketch_right @ inv(B)
        YBinv = P @ lstsq(B.T, S.T, rcond=rcond).T
        result = YBinv, Qh
    return result


def nystrom(sketch_right, sketch_left, mop_right, rcond=1e-6, as_svd=True):
    """ """
    P1, S1 = qr(sketch_right, in_place_q=False, return_R=True)
    if not as_svd:
        Q, R = qr(sketch_left @ mop_right, in_place_q=False, return_R=True)
        # P1 @ SRinv equals sketch_right @ inv(R)
        SRinv = lstsq(R.T, S1.T, rcond=rcond).T
        result = P1, (SRinv @ Q.conj().T @ sketch_left)
    else:
        # return in SVD form, more expensive
        # orthogonalization of sketches is needed
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
    rcond=1e-6,
    as_svd=True,
):
    """ """
    # note we use .T instead of .conj().T: using conj+T several times is
    # equivalent to just using T, but more expensive (conj may return copy)
    P = qr(sketch_right, in_place_q=False, return_R=False)
    Qh = qr(sketch_left.T, in_place_q=False, return_R=False).T
    core = lstsq(lilop @ P, sketch_inner, rcond=rcond)
    core = lstsq((Qh @ rilop).T, core.T, rcond=rcond).T
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
    by_mag=True,
):
    r""" """
    # If the placements of the .conj() don't make sense, note that we
    # leverage symmetries, adding conj in some places and removed from others,
    # to an equivalent result, but less overal scalar conjugations.
    # But also, we can't directly conj() mop_right, so we also avoid that
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    B = (Q.conj().T @ mop_right).conj().T
    core = lstsq(B, sketch_right.conj().T @ Q, rcond=rcond).conj().T
    if not as_eigh:
        result = core, Q
    else:
        ews, Z = eigh(core, by_descending_magnitude=by_mag)
        result = ews, Q @ Z
    return result


def nystrom_h(
    sketch_right,
    mop_right,
    rcond=1e-6,
    as_eigh=True,
    by_mag=True,
):
    """ """
    P, S = qr(sketch_right, in_place_q=False, return_R=True)
    if not as_eigh:
        # (coreInvSt @ P.H) equals inv(rsketch.H @ rmop) @ rsketch.H
        coreInvSt = lstsq(
            sketch_right.conj().T @ mop_right, S.conj().T, rcond=rcond
        )
        # result = P, S @ coreInvSt @ P.conj().T
        result = (S @ coreInvSt), P
    else:
        coreInvSt = lstsq(
            sketch_right.conj().T @ mop_right, S.conj().T, rcond=rcond
        )
        ews, Z = eigh(S @ coreInvSt, by_descending_magnitude=by_mag)
        result = ews, P @ Z
    return result


def oversampled_h(
    sketch_right,
    sketch_inner,
    lilop,
    rilop,
    rcond=1e-6,
    as_eigh=True,
    by_mag=True,
):
    """ """
    P = qr(sketch_right, in_place_q=False, return_R=False)
    core = lstsq(lilop @ P, sketch_inner, rcond=rcond)
    # equivalent to lstsq((rilop.conj().T @ P), core.conj().T).conj().T
    # but we avoid direct congugation of rilop, which may not be supported
    core = lstsq((P.conj().T @ rilop).T, core.T, rcond=rcond).T
    if not as_eigh:
        result = core, P
    else:
        ews, Z = eigh(core, by_descending_magnitude=by_mag)
        result = ews, P @ Z
    return result
