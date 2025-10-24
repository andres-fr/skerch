#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""In-core sketched algorithms."""


import warnings

import torch

from .linops import (
    BaseLinOp,
    CompositeLinOp,
    SumLinOp,
    TransposedLinOp,
    check_linop_input,
)
from .measurements import (
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    RademacherNoiseLinOp,
    SsrftNoiseLinOp,
)
from .recovery import (
    hmt,
    hmt_h,
    nystrom,
    nystrom_h,
    oversampled,
    oversampled_h,
    singlepass,
    singlepass_h,
)
from .utils import (
    COMPLEX_DTYPES,
    BadShapeError,
    qr,
    serrated_hadamard_pattern,
)


# ##############################################################################
# # DISPATCHER
# ##############################################################################
class SketchedAlgorithmDispatcher:
    """Provides sketched algos with measurement linops and recovery algorithms.

    Sketched methods are generally flexible in the kind of noise or recovery
    method they use. In order to accommodate for this flexibility,
    ``skerch`` algorithms call this dispatcher with their
    requested configuration (e.g. ``"gaussian"`` for noise and ``"nystrom"``
    for recovery).

    This modular and extendible framework makes it easy to change between
    existing options, and also to add new options without having to modify
    the sketched algorithms. For example, if users want to run :func:`ssvd`
    with a new type of noise linop called ``MyNoise``, they just need to
    extend :meth:`mop` into a new dispatcher class, and provide the
    dispatcher to SSVD.

    Detailed examples on how to do this can be found in
    the documentation.
    """

    @staticmethod
    def recovery(recovery_type, hermitian=False):
        """Returns recovery funtion with given specs.

        :param recovery_type: String specifying which recovery type to be used
          (e.g. ``"singlepass"``, ``"nystrom"``, ``"oversampled_500"``).
        :returns: A tuple ``(fn, inner_dims)``, where ``fn`` is a recovery
          function. If ``inner_dims`` is None, ``fn`` should
          have an interface like :func:`skerch.recovery.singlepass`. If
          ``inner_dims`` is a positive integer, ``fn`` should behave like
          :func:`skerch.recovery.oversampled`.
        """
        supported = "hmt, singlepass, nystrom, oversampled_123"
        inner_dims = None
        if recovery_type == "hmt":
            recovery_fn = hmt_h if hermitian else hmt
        elif recovery_type == "singlepass":
            recovery_fn = singlepass_h if hermitian else singlepass
        elif recovery_type == "nystrom":
            recovery_fn = nystrom_h if hermitian else nystrom
        elif "oversampled" in recovery_type:
            recovery_fn = oversampled_h if hermitian else oversampled
            inner_dims = int(recovery_type.split("_")[-1])
        else:
            raise ValueError(
                f"Unknown recovery {recovery_type}! Supported: {supported}"
            )
        #
        return recovery_fn, inner_dims

    @staticmethod
    def mop(mop_type, hw, seed, dtype, blocksize=1, register=False):
        """Returns measurement linop with given specs.

        The returned linop must follow the same interface as
        :class:`skerch.linops.ByBlockLinOp`, i.e. must support:

        * Left and right matmul to matrices via ``@``
        * A ``get_blocks(dtype, device)`` method that returns pairs in the form
          ``(block, idxs)``, where ``block`` is a subset of vectors, assumed to
          be columns of the returned linop, and corresponding to the given
          ``idxs``.

        :param mop_type: A string defining the measurement linop type (e.g.
          ``"rademacher"``, ``"gaussian"``, ``"ssrft"``, ``"phase"``).
        :param hw: Tuple ``(height, width)`` with the desired linop shape.
        :param seed: Random seed for the measurement linop.
        :param blocksize: How many measurements should be done at once.
          Ideally, as many as it fits in memory.
          See :class:`skerch.linops.ByBlockLinOp` for more details.
        :param register: Whether to register this linop in a global tracker
          that will raise an exception if other linops with overlapping
          seeds have been already created. This is useful to debug new methods
          if they have e.g. issues due to the use of correlated noise. See
          :class:`skerch.measurements.RademacherNoiseLinOp` for an example.
        """
        supported = "rademacher, gaussian, ssrft, phase"
        if mop_type == "rademacher":
            mop = RademacherNoiseLinOp(
                hw, seed, by_row=False, blocksize=blocksize, register=register
            )
        elif mop_type == "gaussian":
            mop = GaussianNoiseLinOp(
                hw, seed, by_row=False, blocksize=blocksize, register=register
            )
        elif mop_type == "ssrft":
            mop = SsrftNoiseLinOp(
                hw, seed, blocksize=blocksize, register=register
            )
        elif mop_type == "phase":
            if dtype not in COMPLEX_DTYPES:
                raise ValueError(
                    "Phase noise expects complex dtype! Use Rademacher instead"
                )
            mop = PhaseNoiseLinOp(
                hw,
                seed,
                by_row=False,
                blocksize=blocksize,
                register=register,
                conj=False,
            )
        else:
            raise ValueError(
                f"Unknown type! {mop_type} Supported: {supported}"
            )
        #
        return mop

    @staticmethod
    def unitnorm_lop_entries(lop_type):
        """True if all ``lop_type`` entries are supposed to have unit norm.

        E.g. Rademacher or Phase noise entries return ``True``, and Gaussian
        returns ``False``.
        """
        if lop_type in {"rademacher", "phase"}:
            result = True
        elif lop_type in {"gaussian", "ssrft"}:
            result = False
        else:
            warnings.warn(
                f"Unknown linop type {lop_type}. Assumed to be non-unitnorm. ",
                RuntimeWarning,
                stacklevel=2,
            )
            result = False
        #
        return result


# ##############################################################################
# # SSVD/SEIGH
# ##############################################################################
def ssvd(  # noqa:C901
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    recovery_type="hmt",
    lstsq_rcond=1e-6,
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
):
    r"""In-core Sketched SVD.

    The core idea behind sketched SVDs, introduced in
    `[HMT2009] <https://arxiv.org/abs/0909.4061>`_ and outlined in
    :func:`skerch.recovery.hmt`, relies on the existence of orthogonal
    thin matrices :math:`P, Q`, such that the following approximation
    holds:

    .. math::

      A \approx P P^H A Q Q^H = P C Q^H

    The crucial point here is that it is possible to efficiently obtain
    :math:`P, Q` from random measurements, or sketches. Then, it only remains
    to decompose the *small* "core" matrix :math:`C`, which can be done with
    traditional methods.

    Depending on which ``recovery_type`` we use, there are several strategies
    to solve this sketch, each with their nuances and tradeoffs (see
    e.g. note below). Available
    recovery methods are implemented and documented in :mod:`skerch.recovery`.

    .. note::

      The straightforward recovery method from
      `[HMT2009] <https://arxiv.org/abs/0909.4061>`_ (implemented in
      :func:`skerch.recovery.hmt`) requires us to first
      obtain :math:`Q`, and then do a *second round of measurements*
      :math:`A Q` to obtain the core matrix.
      Further developments have shown that it is possible to estimate
      :math:`A Q` without having to run a second pass of measurements by
      solving a least-squares problem,
      which can lead to significant speedup if we leverage parallelization
      (see e.g. `[TYUC2018] <https://arxiv.org/abs/1609.00048>`_).
      These *single-pass* recovery methods have also
      been implemented in :mod:`skerch.recovery`. Check the corresponding
      docs and the ``recovery_type`` parameter for more info.

    :param lop: The linear operator :math:`A` to be decomposed
    :param lop_device: The device where :math:`A` runs
    :param lop_dtype: The datatype :math:`A` interacts with
    :param outer_dims: How many measurements will be used to obtain
      :math:`P` and :math:`Q` (respectively)
    :param seed: Overall random seed for the algorithm
    :param noise_type: Which noise to use. Must be supported by the given
      ``dispatcher`` (see :meth:`SketchedAlgorithmDispatcher.mop`)
    :param recovery_type: Which recovery method to use. Must be supported
      by the given ``dispatcher`` (see
      :meth:`SketchedAlgorithmDispatcher.recovery` and
      :mod:`skerch.recovery`).
    :param lstsq_rcond: Least-squares condition threshold used in the
      recovery, see :meth:`SketchedAlgorithmDispatcher.recovery`
      and :mod:`skerch.recovery`.

    :param meas_blocksize: How many sketched measurements should be done
      at once. Ideally, as many as it fits in memory.
      See :class:`skerch.linops.ByBlockLinOp` for more details.
    :returns: The singular value decomposition ``U, S, Vh`` where
      :math:`A \approx U diag(S) V^H`.
    """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    # figure out recovery settings
    recovery_fn, inner_dims = dispatcher.recovery(recovery_type, False)
    if (outer_dims > max(h, w)) or (
        inner_dims is not None and (inner_dims > max(h, w))
    ):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims <= outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    if meas_blocksize is None:
        idims = 0 if inner_dims is None else inner_dims
        meas_blocksize = max(lop.shape) + outer_dims + idims
    # perform right outer measurements
    ro_seed = seed
    ro_mop = dispatcher.mop(
        noise_type,
        (w, outer_dims),
        ro_seed,
        lop_dtype,
        meas_blocksize,
        register,
    )
    ro_sketch = torch.empty(
        (lop.shape[0], outer_dims), dtype=lop_dtype, device=lop_device
    )
    for block, idxs in ro_mop.get_blocks(lop_dtype, lop_device):
        ro_sketch[:, idxs] = lop @ block  # assuming block is by_col!

    # optionally perform left outer measurements
    if recovery_type != "hmt":
        lo_seed = ro_seed + outer_dims + 1
        lo_mop = dispatcher.mop(
            noise_type,
            (h, outer_dims),
            lo_seed,
            lop_dtype,
            meas_blocksize,
            register,
        )
        lo_sketch = torch.empty(
            (outer_dims, lop.shape[1]), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in lo_mop.get_blocks(lop_dtype, lop_device):
            lo_sketch[idxs, :] = block.conj().T @ lop  # assuming block by_col

    # optionally perform inner measurements
    if inner_dims is not None:
        ri_seed = lo_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = dispatcher.mop(
            noise_type,
            (w, inner_dims),
            ri_seed,
            lop_dtype,
            meas_blocksize,
            register,
        )
        li_mop = TransposedLinOp(
            dispatcher.mop(
                noise_type,
                (h, inner_dims),
                li_seed,
                lop_dtype,
                meas_blocksize,
                register,
            )
        )
        inner_sketch = torch.empty(
            (inner_dims, inner_dims), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in ri_mop.get_blocks(lop_dtype, lop_device):
            inner_sketch[:, idxs] = li_mop @ (lop @ block)  # assuming by_col!

    # solve sketches:
    if recovery_type == "hmt":
        U, S, Vh = recovery_fn(ro_sketch, lop, as_svd=True)
    elif inner_dims is None:
        U, S, Vh = recovery_fn(
            ro_sketch, lo_sketch, ro_mop, rcond=lstsq_rcond, as_svd=True
        )
    elif inner_dims is not None:  # oversampled:
        U, S, Vh = recovery_fn(
            ro_sketch,
            lo_sketch,
            inner_sketch,
            li_mop,
            ri_mop,
            rcond=lstsq_rcond,
            as_svd=True,
        )
    else:
        raise RuntimeError("Should never happen!")
    #
    return U, S, Vh


def seigh(  # noqa:C901
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    recovery_type="hmt",
    lstsq_rcond=1e-6,
    meas_blocksize=None,
    by_mag=True,
    dispatcher=SketchedAlgorithmDispatcher,
):
    r"""In-core Sketched Hermitian eigendecomposition (EIGH).

    This function is the Hermitian version of
    :func:`ssvd`. It behaves largely the same, with the following differences:

    * It assumes that the provided ``lop`` is Hermitian
    * Less measurements are performed due to Hermitian symmetry
    * Unlike the SVD, the returned value is a pair :math:`(\Lambda, Q)`, where
      :math:`A \approx Q diag(\Lambda) Q^H`. Since :math:`A` is Hermitian,
      this is an eigendecomposition: :math:`\Lambda` may contain negative
      values, and both left and right matrices are identical.

    Refer to the :func:`ssvd` docs for more details.
    """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise ValueError("SEIGH expects square operators!")
    dims = h
    # figure out recovery settings
    recovery_fn, inner_dims = dispatcher.recovery(recovery_type, True)
    if (outer_dims > dims) or (inner_dims is not None and (inner_dims > dims)):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims <= outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    if meas_blocksize is None:
        idims = 0 if inner_dims is None else inner_dims
        meas_blocksize = max(lop.shape) + outer_dims + idims
    # perform outer measurements
    ro_seed = seed
    ro_mop = dispatcher.mop(
        noise_type,
        (dims, outer_dims),
        ro_seed,
        lop_dtype,
        meas_blocksize,
        register,
    )
    ro_sketch = torch.empty(
        (lop.shape[0], outer_dims), dtype=lop_dtype, device=lop_device
    )
    for block, idxs in ro_mop.get_blocks(lop_dtype, lop_device):
        ro_sketch[:, idxs] = lop @ block  # assuming block is by_col!
    # optionally perform inner measurements
    if inner_dims is not None:
        ri_seed = ro_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = dispatcher.mop(
            noise_type,
            (dims, inner_dims),
            ri_seed,
            lop_dtype,
            meas_blocksize,
            register,
        )
        li_mop = TransposedLinOp(
            dispatcher.mop(
                noise_type,
                (dims, inner_dims),
                li_seed,
                lop_dtype,
                meas_blocksize,
                register,
            )
        )
        inner_sketch = torch.empty(
            (inner_dims, inner_dims), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in ri_mop.get_blocks(lop_dtype, lop_device):
            inner_sketch[:, idxs] = li_mop @ (lop @ block)  # assuming by_col!

    # solve sketch
    if recovery_type == "hmt":
        ews, evs = recovery_fn(ro_sketch, lop, as_eigh=True, by_mag=by_mag)
    elif inner_dims is None:
        ews, evs = recovery_fn(
            ro_sketch,
            ro_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    elif inner_dims is not None:  # oversampled:
        ews, evs = recovery_fn(
            ro_sketch,
            inner_sketch,
            li_mop,
            ri_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    #
    return ews, evs


# ##############################################################################
# # TRACE AND DIAGONAL
# ##############################################################################
def hutch(
    lop,
    lop_device,
    lop_dtype,
    num_meas,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    return_diag=True,
    defl_Q=None,
):
    r"""Girard-Hutchinson trace and diagonal estimator.

    Given a square linear operator :math:`A`, and random vectors
    :math:`v \sim \mathcal{R}` with :math:`\mathbb{E}[v v^H] = I`, consider
    the Girard-Hutchinson **diagonal** estimator:

    .. math::

      \mathbb{E}_{v \sim \mathcal{R}} \big[ \bar{v} \odot Av \big]_i
             = \sum_j A_{ij} \mathbb{E}\big[ \bar{v}_i v_j \big] = A_{ii}

    The convergence of this estimator can be accelerated greatly if we
    also incorporate rank deflation :math:`A = (Q Q^H) A + (I - (Q Q^H)) A`,
    where :math:`Q` is an orthogonal matrix spanning the top space of
    :math:`A`. Similarly to :func:`ssvd`, it turns out that we can obtain
    :math:`Q` efficiently from ``defl_dims`` random measurements.
    Then, the diagonal of :math:`(Q Q^H) A` can be obtained exactly, and
    :math:`diag((I - (Q Q^H)) A)` is then estimated via Girard-Hutchinson
    using ``extra_gh_meas`` (uncorrelated) measurements. This is the
    ``Hutch++`` algorithm (see `[BN2022] <https://arxiv.org/abs/2201.10684>`_).

    A very similar logic applies for the estimation of the **trace**, which
    is the sum of the diagonal entries. In fact, most computations can be
    recycled, and this algorithm computes and returns both quantities jointly
    with minimal overhead. The Hutchinson estimator for the trace is:

    .. math::

      tr(A) = \langle A, I \rangle
      = \langle A, \mathbb{E}_{v \sim \mathcal{R}}[v v^H] \rangle
      = \mathbb{E}_{v \sim \mathcal{R}}[v^H A v]

    .. note::

      As it can be seen in Table 1 from
      `[BN2022] <https://arxiv.org/abs/2201.10684>`_, Gaussian noise is
      generally less efficient than Rademacher for the Girard-Hutchinson
      step. In general, we observe that noise entries that are more or less
      of the same magnitud help with stabiltiy. We also see that the number
      of required G-H samples for good recovery is in the millions,
      i.e. this is not efficient for smaller matrices.

    :param lop: The :math:`A` operator whose diagonal we wish to estimate.
    :param lop_device: The device where :math:`A` runs.
    :param lop_dtype: The datatype :math:`A` interacts with.
    :param num_meas: How many measurements will be used.
    :param seed: Overall random seed for the algorithm.
    :param noise_type: Which noise to use. Must be supported by the given
      ``dispatcher`` (see :meth:`SketchedAlgorithmDispatcher.mop`).
    :param meas_blocksize: How many sketched measurements should be done
      at once. Ideally, as many as it fits in memory.
      See :class:`skerch.linops.ByBlockLinOp` for more details.
    :param return_diag: If true, diagonal estimation is also returned.
      Otherwise just the trace.
    :param defl_Q: The tall :math:`Q` matrix used for optional deflation (see
      explanation). It should be composed of orthonormal columns and have
      shape ``(dims, defl_dims)``. Ideally it is aligned with the top-space
      of ``lop``. Also, it should be obtained from measurements that are
      uncorrelated to this estimator, so make sure to use a
      different source of random noise (e.g. a different seed) here.
    :returns: A dictionary in the form ``{"tr": tr, "diag": diag}`` with
      the trace and diagonal (if ``return_diag`` is true) estimations.
    """
    # housekeeping
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w or h <= 0:
        raise BadShapeError("hutch expects nonempty square operators!")
    dims = h
    # figure out recovery settings
    if num_meas <= 0:
        raise ValueError("num_meas must be positive!")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    if not is_noise_unitnorm:
        warnings.warn(
            "Non-unitnorm noise can be unstable for trace/diag estimation! "
            + "Check output quality and consider using Rademacher/PhaseNoise.",
            RuntimeWarning,
            stacklevel=2,
        )
    #
    mop = dispatcher.mop(
        noise_type,
        (dims, num_meas),
        seed,
        lop_dtype,
        num_meas if meas_blocksize is None else meas_blocksize,
        register,
    )
    trace = 0
    if return_diag:
        diag = torch.zeros(dims, dtype=lop_dtype, device=lop_device)
    for block, idxs in mop.get_blocks(lop_dtype, lop_device):
        # transpose block: all measurements adjoint since Q deflates lop on
        # the left space
        block = block.T  # after transp: (idxs, dims)
        # nonscalar normalization before deflation
        if not is_noise_unitnorm:
            # so gram matrix of (dims, dims) has diag=len(idxs)
            # and adding every subtrace/gh_meas yields unit diagonal.
            block *= (len(idxs) ** 0.5) / block.norm(dim=0)
        # deflate block and perform adj meas
        if defl_Q is None:
            block_defl = block
        else:
            block_defl = block - (block @ defl_Q.conj()) @ defl_Q.T
        b_lop = block_defl.conj() @ lop
        if return_diag:
            # just accumulate diagonal, then sum at end to get trace
            diag += (block_defl * b_lop).sum(0)
        else:
            # just accumulate the trace, no diag needed
            trace += (block_defl * b_lop).sum() / num_meas
    #
    if return_diag:
        diag /= num_meas
        result = {"diag": diag, "tr": diag.sum()}
    else:
        result = {"tr": trace}
    return result


def xhutchpp(  # noqa:C901
    lop,
    lop_device,
    lop_dtype,
    x_dims=0,
    gh_meas=0,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    return_diag=True,
    cache_xmop=True,
):
    r"""Diagonal and trace sketched approximation via Hutch/XDiag.

    In :func:`hutch` we see how to estimate the trace and diagonal via
    Girard-Hutchinson and ``Hutch++``. This function extends this
    functionality with ``XTrace/XDiag``
    `[ETW2024] <https://arxiv.org/pdf/2301.07825>`_, which allow
    us to perform ``x_dims`` dimensional deflation, and then recycle the
    ``x_dims`` measurements for the subsequent deflated Girard-Hutchinson
    estimator, which we  can also further enrich with ``gh_meas``
    extra measurements. The memory cost of this function is dominated
    by the deflation: we need to store a matrix of ``(dims, x_dims)``
    size. The runtime cost is typically dominated by the total number of
    measurements: ``2 * x_dims + gh_meas``.

    .. note::

      This function is equivalent to plain Girard-Hutchinson for
      ``x_dims=0``, and equivalent to ``XDiag`` if ``gh_meas=0``.
      For the ``Hutch++`` estimator, run :func:`hutch` providing the
      desired ``Q_defl`` deflation matrix.

    .. seealso::

      `This blogpost <https://aferro.dynu.net/math/xdiagpp/>`_ provides
      derivations and elaborates on the relationship between Hutch++
      and XDiag, and how this can lead to a common implementation such as
      this one.

    :param defl_dims: How many measurements will be used to obtain
      the :math:`Q` deflation matrix
    :param lop: The :math:`A` operator whose diagonal we wish to estimate.
    :param lop_device: The device where :math:`A` runs
    :param lop_dtype: The datatype :math:`A` interacts with
    :param x_dims: How many measurements will be used to obtain
      the :math:`Q` deflation matrix, and the subsequent deflated
      estimation (the ``k`` above).
    :param seed: Overall random seed for the algorithm
    :param noise_type: Which noise to use. Must be supported by the given
      ``dispatcher`` (see :meth:`SketchedAlgorithmDispatcher.mop`)
    :param meas_blocksize: How many sketched measurements should be done
      at once. Ideally, as many as it fits in memory.
      See :class:`skerch.linops.ByBlockLinOp` for more details.
    :param cache_mop: If true, the measurement linear operator (which is
      used twice) is converted to an explicit matrix and kept around.
      This saves computation, since it does not need to be sampled twice,
      at the expense of the memory required to store its entries.
    :returns: A dictionary in the form
      ``{"tr": t, "diag": d, "Q": Q, "R": R, "Sh_k": S, "Psi": P}`` containing
      trace and diagonal (if ``return_diag`` is true) estimations,
      as well as the :math:`Q, R` matrices corresponding to the QR
      decomposition of the deflation measurements, and the
      :math:`S_k^H, \Psi` objects needed to compute the exchangeable estimator.
    """
    # housekeeping
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w or h <= 0:
        raise BadShapeError("xhutchpp expects nonempty square operators!")
    dims = h
    # figure out recovery settings
    if x_dims > dims:
        raise ValueError("More x_dims than diag entries!")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    if not is_noise_unitnorm:
        warnings.warn(
            "Non-unitnorm noise can be unstable for trace/diag estimation! "
            + "Check output quality and consider using Rademacher/PhaseNoise.",
            RuntimeWarning,
            stacklevel=2,
        )
    if (x_dims < 0) or (gh_meas < 0):
        raise ValueError("Negative number of measurements?")
    if x_dims + gh_meas <= 0:
        raise ValueError("Deflation dims and/or GH measurements needed!")
    A_defl = lop
    # if we have x_dims, actually deflate A and obtain deflation estimates
    if x_dims >= 1:
        xmop = dispatcher.mop(
            noise_type,
            (dims, x_dims),
            seed,
            lop_dtype,
            x_dims if meas_blocksize is None else meas_blocksize,
            register,
        )
        if cache_xmop:
            xmop = xmop.to_matrix(lop_dtype, lop_device)
        # compute exchangeability objects
        Q, R = torch.linalg.qr(lop @ xmop)
        Rinv = torch.linalg.inv(R)
        Sh_k = Rinv / (Rinv.norm(dim=1, keepdim=True) * x_dims**0.5)
        Psi = -Sh_k.H @ Sh_k
        Psi[range(x_dims), range(x_dims)] += 1
        Psi_Qh_A = Psi @ Q.H @ lop
        #
        Q_Psi_Qh_A = CompositeLinOp([("Q", Q), ("Psi Qh A", Psi_Qh_A)])
        A_defl = SumLinOp([("A", True, lop), ("Q Psi QhA", False, Q_Psi_Qh_A)])
        # obtain xdiag/xtrace deflation estimates
        if cache_xmop:
            iterator = [(xmop, range(0, x_dims))]
        else:
            iterator = xmop.get_blocks(lop_dtype, lop_device)
        if return_diag:
            xdiag = (Q.T * Psi_Qh_A).sum(0)
            xtrace = xdiag.sum()
            #
            ydiag = torch.zeros_like(xdiag)
            for block, idxs in iterator:
                if not is_noise_unitnorm:
                    # so gram matrix of (dims, dims) has diag=len(idxs)
                    # and adding every subtrace/gh_meas yields unit diagonal.
                    block *= (len(idxs) ** 0.5) / block.norm(dim=0)
                ydiag += (
                    (Q @ Sh_k[idxs].H)
                    * (block.conj() * (Sh_k[idxs] @ R[:, idxs]).diag())
                ).sum(1)
            ytrace = ydiag.sum()
        else:
            xtrace = (Q.T * Psi_Qh_A).sum()
            ytrace = 0
            for block, idxs in iterator:
                if not is_noise_unitnorm:
                    # so gram matrix of (dims, dims) has diag=len(idxs)
                    # and adding every subtrace/gh_meas yields unit diagonal.
                    block *= (len(idxs) ** 0.5) / block.norm(dim=0)
                ytrace += (
                    (Q @ Sh_k[idxs].H)
                    * (block.conj() * (Sh_k[idxs] @ R[:, idxs]).diag())
                ).sum()
    #
    if gh_meas >= 1:
        # Girard-Hutchinson on (optionally deflated) A
        defl = hutch(
            A_defl,
            lop_device,
            lop_dtype,
            gh_meas,
            seed + x_dims,
            noise_type,
            meas_blocksize,
            dispatcher,
            return_diag,
        )
    # merge logic:
    if x_dims == 0:
        result = defl
    elif gh_meas == 0:
        result = {"tr": xtrace + ytrace}
        if return_diag:
            result["diag"] = xdiag + ydiag
    else:
        result = defl
        k, q = x_dims, gh_meas  # both nonzero
        result["tr"] = xtrace + (k * ytrace + q * defl["tr"]) / (k + q)
        if return_diag:
            result["diag"][:] = xdiag + (k * ydiag + q * defl["diag"]) / (
                k + q
            )
    #
    if x_dims >= 1:
        result["Q"] = Q
        result["R"] = R
        result["Sh_k"] = Sh_k
        result["Psi"] = Psi
    #
    return result


# ##############################################################################
# # NORMS
# ##############################################################################
def snorm(
    lop,
    lop_device,
    lop_dtype,
    num_meas=5,
    seed=0b1110101001010101011,
    noise_type="gaussian",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    adj_meas=None,
    norm_types=("fro", "op"),
):
    r"""Sketched norm estimation.

    It is possible to estimate the norm of a linear operator from a few
    random measurements. In
    `[TYUC2019, 6.2] <https://arxiv.org/abs/1902.08651>`_, it is shown
    that the Frobenius norm is proportional to the norm of the measurements
    themselves (akin to a Gram trace estimation). For :math:`q` random
    measurements :math:`\Omega` and ``beta = 2 if complex_dtype else 1``
    we have:

    .. math::

      \lVert A \rVert_F^2 = \frac{1}{\beta q} \mathbb{E} \big[ A \Omega \big]

    From the same measurements, we can also estimate the operator norm
    (largest singular value) as follows. Consider the linear operator
    :math:`A`, and its top-``k`` approximation :math:`\hat{A} = Q Q^H A`
    for orthogonal :math:`Q`. Then:

    .. math::

      \lVert A \rVert_2 = \lVert \hat{A} \rVert_2
      \approx \lVert Q Q^H A \rVert_2 = \lVert Q^H A \rVert_2

    And, as discussed in :func:`ssvd`, :math:`Q` can be efficiently estimated
    from a few random measurements :math:`A \Omega`.

    :param lop: The :math:`A` operator whose norms we wish to estimate.
    :param lop_device: The device where :math:`A` runs
    :param lop_dtype: The datatype :math:`A` interacts with
    :param num_meas: How many measurements will be used to estimate
      the norms (:math:`q` in the above description)
    :param seed: Overall random seed for the algorithm
    :param noise_type: Which noise to use for :math:`\Omega`. Must be
      supported by the given ``dispatcher``
      (see :meth:`SketchedAlgorithmDispatcher.mop`)
    :param meas_blocksize: How many sketched measurements should be done
      at once. Ideally, as many as it fits in memory.
      See :class:`skerch.linops.ByBlockLinOp` for more details.
    :param adj_meas: If true, measurements to obtain :math:`Q` are adjoint.
      This can affect accuracy somewhat.
    :param norm_types: Collection with norm types to be returned. Currently
      supported: ``"fro"`` (Frobenius), ``"op"`` (operator norm)
    :returns: The tuple ``(result, (Q, R))``, where ``(Q, R)`` is the QR
      decomposition of the sketched measurements, and ``result`` is a
      dictionary in the form ``{norm_type: value}``.
    """
    h, w = lop.shape
    if num_meas <= 0 or num_meas > min(h, w):
        raise ValueError(
            "Measurements must be between 1 and number of rows/columns!"
        )
    if meas_blocksize is None:
        meas_blocksize = max(lop.shape) + num_meas
    if adj_meas is None:
        adj_meas = h > w  # this seems to be more accurate
    mop = dispatcher.mop(
        noise_type,
        (h if adj_meas else w, num_meas),
        seed,
        lop_dtype,
        meas_blocksize,
        register=False,
    )
    # perform measurements and obtain top-space Q
    sketch = torch.empty(
        (w if adj_meas else h, num_meas), dtype=lop_dtype, device=lop_device
    )
    for block, idxs in mop.get_blocks(lop_dtype, lop_device):
        # assuming block is by_col!
        sketch[:, idxs] = (block.T @ lop).conj().T if adj_meas else lop @ block
    Q, R = qr(sketch, in_place_q=True, return_R=True)
    # project lop onto Q and obtain largest singular value
    sketch2 = lop @ Q if adj_meas else Q.conj().T @ lop
    h2, w2 = sketch2.shape
    gram = (
        (sketch2.conj().T @ sketch2)
        if h2 > w2
        else (sketch2 @ sketch2.conj().T)
    )
    result = {}
    for norm_type in norm_types:
        if norm_type == "op":
            result[norm_type] = torch.linalg.norm(gram, ord=2) ** 0.5
        elif norm_type == "fro":
            result[norm_type] = gram.diag().real.sum() ** 0.5
        else:
            raise ValueError(f"Unsupported norm type! {norm_type}.")
    #
    return result, (Q, R, sketch2)


# ##############################################################################
# # TRIANGULAR LINEAR OPERATOR
# ##############################################################################
class TriangularLinOp(BaseLinOp):
    r"""Given a square linop, compute products with its lower/upper triangle.

    The triangle of a linear operator can be approximated from the full
    operator via a "staircase pattern" of exact measurements. For example,
    given an operator of shape ``(1000, 1000)``, and
    stairs of size 100, we could obtain 9 exact measurements strictly under
    the diagonal, the first one covering ``lop[100:, :100]``, the next one
    ``lop[200:, 100:200]``, and so on. And we would leave out only 9 small
    triangles near the diagonal in a sort of "serrated" pattern.
    The smaller the stair size, the more measurements we do and the more
    closely the full triangle is approximated (a stair size of 1 results
    in an exact method, doing full measurements).

    The interesting thing is that the "serrated" part can be estimated
    with a variation of the Girard-Hutchinson estimator:

    .. math::

      f(A) =
      \mathbb{E}_{v \sim \mathcal{R}} \big[ \varphi(\bar{v}) \odot Av \big]

    Where :math:`\varphi` follows a
    :func:`skerch.utils.serrated_hadamard_pattern`.
    By linearity, it can be shown that this expectation samples entries of
    :math:`A` from the block-triangles near the diagonal.

    Finally, we add the exact step-wise measurements and the
    serrated Girard-Hutchinson estimation (see e.g. :func:`hutchpp`),
    resulting in a triangular estimation.

    Usage example::

      lop = MyLinOp("...")
      tril = TriangularLinOp(lop, lower=True, stair_width=10, num_gh_meas=1000)
      w = tril @ v

    :param lop: A square linear operator. It must implement a
      ``lop.shape = (dims, dims)`` attribute as well as the left- and right-
      matmul operator ``@``, interacting with torch tensors.
    :param stair_width: Width of each step in the staircase pattern. If
      it is 1, a total of ``dims`` exact measurements will be performed and
      the triangular products will be exact.
      If it equals ``dims``, no exact measurements will be performed (since
      the staircase pattern would cover the full triangle). The step size
      regulates this trade-off: Ideally, we want as many exact measurements
      as possible, but not too many. If no value is provided, ``dims // 2``
      is chosen by default, such that only 1 exact measurement is performed.
    :param num_hutch_measurements: Number of measurements for the
      serrated estimation that complements the staircase estimation.
      This estimator generally requires
      many measurements to be informative, and it can even be counter-
      productive if not enough measurements are given. If ``lop`` is not
      diagonally dominant, consider setting this to 0 for a sufficiently
      good approximation via deterministic ``staircase_measurements``.
      Otherwise, make sure to provide a sufficiently high number of
      measurements.
    :param lower: If true, lower triangular matmuls will be computed.
      Otherwise, upper triangular.
    :param with_main_diagonal: If true, the main diagonal will be included
      in the triangle, otherwise excluded. If you already have precomuted
      the diagonal elsewhere, consider excluding it from this approximation,
      and adding it separately.
    :param use_fft: Whether to use FFT for the Hutchinson estimation. See
      :func:`skerch.utils.subdiag_hadamard_pattern` for more details.
    :param seed: Seed for the random measurements used in the
      serrated estimator.
    :param noise_type: String indicating noise type to be used in the
      serrated estimator. Must be supported by the given ``dispatcher``.
    :param meas_blocksize: How many serrated measurements should be done
      at once. Ideally, as many as it fits in memory.
      See :class:`skerch.linops.ByBlockLinOp` for more details.
    """

    LOP_REPR_CHARS = 30

    def __init__(
        self,
        lop,
        stair_width=None,
        num_gh_meas=0,
        lower=True,
        with_main_diagonal=False,
        use_fft=False,
        #
        seed=0b1110101001010101011,
        noise_type="rademacher",
        max_mp_workers=None,
        dispatcher=SketchedAlgorithmDispatcher,
        meas_blocksize=None,
    ):
        """Initializer. See class docstring."""
        h, w = lop.shape
        if h != w:
            raise BadShapeError("Only square linear operators supported!")
        self.dims = h
        if self.dims < 1:
            raise BadShapeError("Empty linear operators not supported!")
        if stair_width is None:
            stair_width = max(1, self.dims // 2)
        if stair_width < 1 or stair_width > self.dims:
            raise ValueError("Stair width must be >=1 and <= dims!")
        #
        if with_main_diagonal:
            # TODO: current stair measurements do not include diagonal,
            # and when e.g. stair_width=1 and gh=0, the diagonal is completely
            # left out. The desirable behaviour would be that the stair
            # measurements cut into the diagonal whenever required, and that
            # the serrated pattern adjusts accordingly.
            raise NotImplementedError("Triang+diag currently not supported!")
        #
        self.stair_width = stair_width
        self.lop = lop
        self.tlop = TransposedLinOp(lop)
        self.n_gh = num_gh_meas
        self.lower = lower
        self.with_main_diag = with_main_diagonal
        self.use_fft = use_fft
        #
        self.seed = seed
        self.noise_type = noise_type
        self.is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
        self.max_mp_workers = max_mp_workers
        self.dispatcher = dispatcher
        if meas_blocksize is None:
            meas_blocksize = max(lop.shape) + num_gh_meas
        self.meas_blocksize = meas_blocksize
        #
        if num_gh_meas <= 0:
            warnings.warn(
                "num_gh_meas <=0: only staircase measurements will be done! "
                + "Set this to a large integer for more accurate estimation",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if not self.is_noise_unitnorm:
                warnings.warn(
                    "Non-unitnorm noise can be unstable for triangular "
                    "matvecs! Check output and consider using Rademacher "
                    "or PhaseNoise.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        #
        super().__init__(lop.shape)  # this sets self.shape also

    @staticmethod
    def _iter_stairs(dims, stair_width, reverse=False):
        """Helper method to iterate over staircase indices.

        This method implements an iterator that yields ``(begin, end)`` index
        pairs for each staircase-pattern step. It terminates before ``end``
        is equal or greater than ``self.dims``, since only full steps are
        considered.
        """
        beg, end = 0, stair_width
        while end < dims:
            result = (dims - end, dims - beg) if reverse else (beg, end)
            yield result
            #
            beg = end
            end = beg + stair_width

    @staticmethod
    def _gh_meas(
        x,
        lop,
        mop,
        is_noise_unitnorm,
        adjoint,
        stair_width,
        with_main_diag,
        lower=True,
        use_fft=False,
    ):
        """Helper method to perform serrated measurements.

        It is a variation of the Girard-Hutchinson diagonal estimator, see
        class docstring for more details.
        """
        dtype, device = x.dtype, x.device
        result = torch.zeros_like(x)
        for block, _ in mop.get_blocks(dtype, device):
            pattern = serrated_hadamard_pattern(
                block.T,
                stair_width,
                with_main_diag,
                lower,
                use_fft,
            ).T
            if adjoint:
                # equivalent to x @ [lop * (pattern @ block.H)]
                # where (pattern @ block.H) is the staircase pattern
                meas = (pattern.T * x) @ lop
                result += (block.H * meas).sum(dim=0)
            else:
                # equivalent to [lop * (pattern @ block.H)] @ x
                # where (pattern @ block.H) is the staircase pattern
                meas = lop @ (block.H * x).T
                result += (pattern * meas).sum(dim=1)
        #
        result /= mop.shape[1]
        return result

    def _matmul_helper(self, x, adjoint=False):
        """Forward and adjoint triangular matrix multiplications.

        Since forward and adjoint matmul share many common computations, this
        method implements both at the same time. The specific mode can be
        dispatched using the ``adjoint`` parameter.
        """
        check_linop_input(x, self.lop.shape, adjoint=adjoint)
        # we don't factorize this method because we want to share buff
        # across both loops to hopefully save memory
        buff = torch.zeros_like(x)
        result = torch.zeros_like(x)
        # add step computations to result
        for beg, end in self._iter_stairs(
            self.dims, self.stair_width, reverse=False
        ):
            if (not adjoint) and self.lower:
                buff[beg:end] = x[beg:end]
                result[end:] += (self.lop @ buff)[end:]
                buff[beg:end] = 0
            elif adjoint and self.lower:
                buff[end:] = x[end:]
                result[beg:end] += (buff @ self.lop)[beg:end]
                buff[end:] = 0
            #
            elif (not adjoint) and (not self.lower):
                buff[end:] = x[end:]
                result[beg:end] += (self.lop @ buff)[beg:end]
                buff[end:] = 0
            elif adjoint and (not self.lower):
                buff[beg:end] = x[beg:end]
                result[end:] += (buff @ self.lop)[end:]
                buff[beg:end] = 0
            else:
                raise RuntimeError("This should never happen")
        #
        if self.n_gh > 0:
            mop = self.dispatcher.mop(
                self.noise_type,
                (self.dims, self.n_gh),
                self.seed,
                x.dtype,
                self.meas_blocksize,
            )
            result += self._gh_meas(
                x,
                self.lop,
                mop,
                self.is_noise_unitnorm,
                adjoint,
                self.stair_width,
                self.with_main_diag,
                self.lower,
                self.use_fft,
            )
        #
        return result

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``.

        See parent class for more details.
        """
        if len(x.shape) > 1:
            raise NotImplementedError("Only vectors supported!")
        return self._matmul_helper(x, adjoint=False)

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
        if len(x.shape) > 1:
            raise NotImplementedError("Only vectors supported!")
        return self._matmul_helper(x, adjoint=True)

    def __repr__(self):
        """Readable string version of this object.

        Returns a string in the form
        <TriangularlLinOp[lop](lower/upper, with/out main diag)>.
        """
        clsname = self.__class__.__name__
        lopstr = str(self.lop)
        if len(lopstr) >= self.LOP_REPR_CHARS:
            lopstr = lopstr[: (self.LOP_REPR_CHARS - 3)] + "..."
        lower_str = "lower" if self.lower else "upper"
        diag_str = "with" if self.with_main_diag else "no"
        result = f"<{clsname}[{lopstr}]({lower_str}, {diag_str} main diag)>"
        return result
