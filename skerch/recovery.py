#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

TODO:
* Implement recoveries for symmetric matrices
  - how to handle truncation?
* extend measurement API to HDF5
  - add HDF5 bells and whistles

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
from .utils import qr, svd, lstsq, eigh


# ##############################################################################
# # RECOVERY FOR GENERAL MATRICES (UV/SVD)
# ##############################################################################
def singlepass(
    sketch_right,
    sketch_left,
    mop_right,
    rcond=1e-8,
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
        YBinv = lstsq(B.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        U, S, Vh = svd(YBinv)
        result = U, S, (Vh @ Qh)
    else:
        YBinv = lstsq(B.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        result = YBinv, Qh
    return result


def compact(sketch_right, sketch_left, mop_right, rcond=1e-8, as_svd=True):
    """Recovering the SVD of a matrix ``L`` from left and right sketches.

    This function is an extension of :func:`singlepass`, using a more compact
    procedure when ``as_svd`` is true. In that case, this method has one more
    *thin* QR decomposition than single-pass, but all other numerical
    linear algebra routines are compact, which may result in substantial
    speedup depending on problem dimensionality.
    """
    Qh = qr(sketch_left.conj().T, in_place_q=False, return_R=False).conj().T
    B = Qh @ mop_right
    if not as_svd:
        YBinv = lstsq(B.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        result = YBinv, Qh
    else:
        P, R = qr(sketch_right, in_place_q=False, return_R=True)
        RBinv = lstsq(B.conj().T, R.conj().T, rcond=rcond).conj().T

        BRinv = lstsq(R.conj().T, B.conj().T, rcond=rcond).conj().T
        Z, S2, _ = svd(RBinv.conj().T @ RBinv)
        S = S2**0.5
        Vh = Z.conj().T @ Qh
        #
        Z, D = qr(RBinv @ Z, return_R=True)
        try:
            D[0] = 2 * (D[range(len(D)), range(len(D))].real > 0) - 1
        except:
            breakpoint()
        U = (P @ Z) * D[0]
        result = U, S, Vh
    return result


def nystrom(sketch_right, sketch_left, mop_right, rcond=1e-8, as_svd=True):
    """ """
    Q, R = qr(sketch_left @ mop_right, in_place_q=False, return_R=True)
    if not as_svd:
        # the original nystrom recovery, cheaper
        rightRinv = (
            lstsq(R.conj().T, sketch_right.conj().T, rcond=rcond).conj().T
        )
        result = rightRinv, (Q.conj().T @ sketch_left)  # U, Vh
    else:
        # return in SVD form, more expensive
        P, S = qr(sketch_right, in_place_q=False, return_R=True)
        rightRinv = lstsq(R.conj().T, S.conj().T, rcond=rcond).conj().T
        U, S, Vh = svd(rightRinv @ (Q.conj().T @ sketch_left))
        result = (P @ U), S, Vh
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
    rcond=1e-8,
    as_eigh=True,
):
    r""" """
    Q = qr(sketch_right, in_place_q=False, return_R=False)
    C = lstsq((Q.conj().T @ mop_right).conj().T, sketch_right.conj().T @ Q)
    if as_eigh:
        ews, Z = eigh(C)
        result = ews, Q @ Z
    else:
        result = Q, C @ Q.conj().T
    return result


def compact_h():
    pass


def nystrom_h():
    pass


def oversampled_h():
    pass
