#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Functionality to recover linop approximations from their sketches."""


from .utils import eigh, lstsq, qr, svd


# ##############################################################################
# # RECOVERY FOR GENERAL MATRICES (UV/SVD)
# ##############################################################################
def hmt(sketch_right, lop, as_svd=True):
    r"""HMT sketched low-rank recovery.

    The core idea behind this method is outlined in
    `[HMT2009] <https://arxiv.org/abs/0909.4061>`_:
    Given a linear operator :math:`A`, we assume that there exists a *thin*,
    orthogonal matrix :math:`Q` such that  :math:`A \approx Q Q^H A`.
    Crucially, it is possible to
    efficiently obtain :math:`Q` from random measurements, or sketches,
    followed by QR orthogonalization.

    Then, to produce an SVD, it remains to decompose the "thin"
    matrix :math:`B^H = P^H A = U \Sigma V^H`, yielding the final
    SVD: :math:`A \approx (P U) \Sigma V^H`.

    :param sketch_right: Sketches ``A @ mop_right`` (typically a tall
      matrix).
    :param lop: Our target linear operator ``A``.
    :returns: If ``as_svd``, a triple ``U, S, Vh`` with
      :math:`A \approx U diag(S) V^H` being a *thin SVD*. Otherwise,
      :math:`(Q, B^H)`.
    """
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    Bh = Q.conj().T @ lop  # second pass over lop!
    if as_svd:
        U, S, Vh = svd(Bh)
        result = (Q @ U), S, Vh
    else:
        result = Q, Bh
    return result


def singlepass(sketch_right, sketch_left, mop_right, rcond=1e-6, as_svd=True):
    r"""Single-pass recovery of linop from left and right sketches.

    Single-pass recovery from
    `[TYUC2018, 4.1] <https://arxiv.org/abs/1609.00048>`_:
    Assuming :math:`A \approx A  Q  Q^H`, and given ``Y, W.H, Omega``, where
    :math:`Y = A \Omega` and :math:`W^H = \Psi^H A` are our random sketches,
    we can recover ``A`` without any further measurements, thus being a
    single-pass method.

    First, we obtain ``Q`` via QR decomposition of ``W``. Then, it holds

    .. math::

       \begin{align}
       Y  (Q^H \Omega)^{-1} Q^H &= (A \Omega) (Q^H \Omega)^{-1} Q^H\\
                   &\approx (A Q Q^H \Omega) (Q^H \Omega)^{-1} Q^H \\
                   &= (A Q)  Q^H \approx A \\
       \end{align}

    Thus, we just need to solve a well-conditioned least-squares problem to
    approximate ``A``. To obtain the full SVD, we further need to compute the
    SVD of ``Y @ pinv(Q.H @ Omega)`` and recombine.

    :param sketch_right: Sketches ``A @ mop_right`` (typically a tall
      matrix).
    :param sketch_left: Sketches ``(mop_left @ A)`` (typically a fat
      matrix).
    :param mop_right: Linop used to obtain ``sketch_right``.
    :param rcond: Singular value threshold using during the least square
      solving process. See :func:`skerch.utils.lstsq`.
    :returns: If ``as_svd``, the triple ``U, S, Vh`` with
      :math:`A \approx U diag(S) V^H` being a *thin SVD*. Otherwise,
      the pair ``Y, Bh`` with :math:`A \approx Y B^H`.
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
    r"""Generalized Nyström recovery of linop from left and right sketches.

    Single-pass recovery from
    `[Naka2020] <https://arxiv.org/abs/2009.11392>`_.
    Here, we assume the low-rank approximation
    :math:`A \approx A \Omega C \Psi^H A`, with *core matrix* in the form
    :math:`C = (\Psi^H A \Omega)^\dagger`.

    Given sketches :math:`Y = A \Omega`, :math:`W^H = \Psi^H A`, this method
    obtains the approximation without requiring any further measurements and
    by solving a single, compact least-squares system in the form
    :math:`A \approx Y (W^H \Omega)^\dagger W^H`.

    To obtain the SVD approximation, ``Y, W`` are further orthogonalized.

    :param sketch_right: Sketches ``A @ mop_right`` (typically a tall
      matrix).
    :param sketch_left: Sketches ``(mop_left @ A)`` (typically a fat
      matrix).
    :param mop_right: Linop used to obtain ``sketch_right``.
    :param rcond: Singular value threshold using during the least square
      solving process. See :func:`skerch.utils.lstsq`.
    :returns: If ``as_svd``, the triple ``U, S, Vh`` with
      :math:`A \approx U diag(S) V^H` being a *thin SVD*. Otherwise,
      the pair ``Y, Bh`` with :math:`A \approx Y B^H`.
    """
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
    r"""Oversampled recovery of linop from left, right and inner sketches.

    Single-pass, oversampled recovery from
    `[BWZ2016] <https://arxiv.org/abs/1504.06729>`_:

    Assuming :math:`A \approx P (P^H A @ Q) @ Q^H = P C Q^H`, this method
    aims to obtain a *core matrix* :math:`C` via an independent, oversampled
    sketch. The above approximation is satisfied by the following core matrix:
    :math:`C = (\Psi^H P)^\dagger \Psi^H A \Omega (Q^H \Omega)^\dagger`,
    and :math:`P, Q` can be been obtained via left and right outer
    sketches and subsequent orthogonalization.

    The key observation here is that :math:`C` can be *oversampled*, i.e.
    the number of measurements taken by :math:`\Psi, \Omega` can be larger
    than the number of columns in :math:`P, Q`. This helps conditioning
    the pseudoinverses and has been shown in
    `[TYUC2018, 4.1] <https://arxiv.org/abs/1609.00048>`_ to yield better
    recoveries when the singular values of :math:`A` decay slowly and the
    number of outer measurements doesn't sufficiently cover for that.

    The trade-off here is that this method requires the extra (independent)
    inner measurements, plus the extra least-squares steps involved
    in solving the two pseudoinverses.
    To return output as SVD, the core matrix is further diagonalized.


    :param sketch_right: Sketches ``A @ mop_right`` (typically a tall
      matrix).
    :param sketch_left: Sketches ``(mop_left @ A)`` (typically a fat
      matrix).
    :param sketch_inner: Sketches ``(lilop^H @ A @ rilop)`` (typically a
      small, square matrix).
    :param lilop: Left inner linop used to obtain ``sketch_inner``.
    :param rilop: Right inner linop used to obtain ``sketch_inner``.
    :param rcond: Singular value threshold using during the least square
      solving process. See :func:`skerch.utils.lstsq`.
    :returns: If ``as_svd``, the triple ``U, S, Vh`` with
      :math:`A \approx U diag(S) V^H` being a *thin SVD*. Otherwise,
      the pair ``Y, Bh`` with :math:`A \approx Y B^H`.
    """
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
def hmt_h(sketch_right, lop, as_eigh=True, by_mag=True):
    r"""HMT sketched low-rank recovery (Hermitian).

    Hermitian version of :func:`hmt`. Since :math:`A = A^H`, we
    can further compact the core matrix into a Hermitian, "small" matrix:
    :math:`A \approx Q Q^H A Q Q^H \Rightarrow C = Q^H A Q`.
    Then we can obtain the eigendecomposition of :math:`C`.

    :param by_mag: see :func:`skerch.utils.eigh`.
    :returns: If ``as_eigh``, the pair :math:`\Lambda, X` corresponding to the
      eigendecomposition :math:`A \approx X diag(\Lambda) X^H`. Otherwise,
      the pair :math:`C, Q` (see derivation above).
    """
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    core = (Q.conj().T @ lop) @ Q  # second pass over lop!
    if not as_eigh:
        result = core, Q
    else:
        ews, Z = eigh(core, by_descending_magnitude=by_mag)
        result = ews, Q @ Z
    return result


def singlepass_h(
    sketch_right, mop_right, rcond=1e-6, as_eigh=True, by_mag=True
):
    r"""Single-pass recovery of Hermitian linop from right sketch.

    Hermitian version of :func:`singlepass`. Since :math:`A = A^H`, we
    only need sketches from one side, because in the Hermitian case ``Q``
    can also be obtained by orthogonalizing ``sketch_right``, and the
    method remains unchanged.

    :param by_mag: see :func:`skerch.utils.eigh`.
    :returns: If ``as_eigh``, the pair :math:`\Lambda, X` corresponding to the
      eigendecomposition :math:`A \approx X diag(\Lambda) X^H`. Otherwise,
      the pair :math:`C, Q` (see derivation above).
    """
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


def nystrom_h(sketch_right, mop_right, rcond=1e-6, as_eigh=True, by_mag=True):
    r"""Generalized Nyström recovery of Hermitian linop from right sketch.

    Hermitian version of :func:`nystrom`. Since :math:`A = A^H`, we
    only need sketches from one side:
    :math:`A \approx A \Omega C \Omega^H A`, with Hermitian *core matrix*
    in the form :math:`C = (\Omega^H A \Omega)^\dagger`.

    :param by_mag: see :func:`skerch.utils.eigh`.
    :returns: If ``as_eigh``, the pair :math:`\Lambda, X` corresponding to the
      eigendecomposition :math:`A \approx X diag(\Lambda) X^H`. Otherwise,
      the pair :math:`C, Q` (see derivation above).
    """
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
    r"""Oversampled recovery of linop from right and inner sketches.

    Hermitian version of :func:`oversampled`. Since :math:`A = A^H`, we
    only need outer sketches from one side. But part of the strength
    in the oversampled method is that the inner measurement matrices
    are uncorrelated.
    To allow for this possibility, this function still admits separate
    parameters for ``lilop`` and ``rilop``.

    (see `[FSMH2025] <https://openreview.net/forum?id=yGGoOVpBVP>`_
    for more discussion).

    :param by_mag: see :func:`skerch.utils.eigh`.
    :returns: If ``as_eigh``, the pair :math:`\Lambda, X` corresponding to the
      eigendecomposition :math:`A \approx X diag(\Lambda) X^H`. Otherwise,
      the pair :math:`C, Q` (see derivation above).
    """
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
