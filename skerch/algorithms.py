#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""In-core sketched algorithms.

This module features the following functionality:

* A modular and extendable dispatcher that provides the algorithms with
  measurement linops and recovery methods.
* Sketched SVD and EIGH
* Sketched trace (Hutch++) and diagonal (Hutch++, XDiag) estimation
* Sketched linop norm estimation (operator norm, Frobenius)
* Sketched triangular matvec estimation
"""


import warnings
from functools import partial
import torch
from .recovery import singlepass, nystrom, oversampled
from .recovery import singlepass_h, nystrom_h, oversampled_h
from .measurements import (
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SsrftNoiseLinOp,
)
from .linops import (
    BaseLinOp,
    CompositeLinOp,
    TransposedLinOp,
    check_linop_input,
)
from .utils import (
    BadShapeError,
    COMPLEX_DTYPES,
    qr,
    lstsq,
    serrated_hadamard_pattern,
)


# ##############################################################################
# # DISPATCHER
# ##############################################################################
class SketchedAlgorithmDispatcher:
    """

    This static class provides functionality to dispatch simple, high-level
    sets of specifications into detailed components of the sketched methods,
    such as noise measurement types and recovery algorithms.

    The goal of such a dispatcher is to help simplifying the interface of the
    sketched algorithms, so users have to provide less details and there are
    less errors due to incorrect/inconsistent inputs.

    The downside is that flexibility is reduced: If users want to test a new
    type of noise or recovery algorithm, this dispatcher will not work.

    To overcome this issue, users can extend this class to incorporate the
    new desired specs, and then provide the extended dispatcher class as
    an argument to the sketched method, which will now recognize the new specs.

    For example, if users want to run :func:`ssvd` with a new type of noise
    linop called ``MyNoise``:
    1. Extend this class as ``MyDispatcher`` and override the ``mop`` method,
      adding an ``if noise_type=='my_noise'`` clause that returns the
      desired instance.
    2. When calling ``ssvd``, provide the ``noise_type='my_noise'`` and
      ``dispatcher=MyDispatcher`` arguments.

    Now ``ssvd`` should perform all measurements using ``MyNoise`` instances.
    """

    @staticmethod
    def recovery(recovery_type, hermitian=False):
        """Returns recovery funtion with given specs."""
        supported = "singlepass, nystrom, oversampled_123"
        inner_dims = None
        if recovery_type == "singlepass":
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

        The returned linop must support the following operations:
        * Left and right matmul to matrices via ``@``
        * A ``get_blocks(dtype, device)`` method that returns pairs in the form
          ``(block, idxs)``, where ``block`` is a subset of vectors, assumed to
          be columns of the returned linop, and corresponding to the given
          ``idxs``.
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
        """True if all ``lop_type`` entries are supposed to have unit norm."""
        if lop_type in {"rademacher", "phase"}:
            result = True
        elif lop_type in {"gaussian", "ssrft"}:
            result = False
        else:
            warnings.warn(
                f"Unknown linop type {lop_type}. Assumed to be non-unitnorm. ",
                RuntimeWarning,
            )
            result = False
        #
        return result


# ##############################################################################
# # SSVD/SEIGH
# ##############################################################################
def ssvd(
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    recovery_type="singlepass",
    lstsq_rcond=1e-6,
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """In-core Sketched SVD."""
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
        meas_blocksize = max(lop.shape)
    # instantiate measurement linops
    ro_seed = seed
    lo_seed = ro_seed + outer_dims + 1
    ro_mop = dispatcher.mop(
        noise_type,
        (w, outer_dims),
        ro_seed,
        lop_dtype,
        meas_blocksize,
        register,
    )
    lo_mop = dispatcher.mop(
        noise_type,
        (h, outer_dims),
        lo_seed,
        lop_dtype,
        meas_blocksize,
        register,
    )
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
    # perform outer measurements
    ro_sketch = torch.empty(
        (lop.shape[0], outer_dims), dtype=lop_dtype, device=lop_device
    )
    lo_sketch = torch.empty(
        (outer_dims, lop.shape[1]), dtype=lop_dtype, device=lop_device
    )
    for block, idxs in ro_mop.get_blocks(lop_dtype, lop_device):
        ro_sketch[:, idxs] = lop @ block  # assuming block is by_col!
    for block, idxs in lo_mop.get_blocks(lop_dtype, lop_device):
        lo_sketch[idxs, :] = block.conj().T @ lop  # assuming block is by_col!
    # if not oversampled, solve sketch
    if inner_dims is None:
        U, S, Vh = recovery_fn(
            ro_sketch, lo_sketch, ro_mop, rcond=lstsq_rcond, as_svd=True
        )
    # if oversampled, perform inner measurements and then solve
    else:
        inner_sketch = torch.empty(
            (inner_dims, inner_dims), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in ri_mop.get_blocks(lop_dtype, lop_device):
            inner_sketch[:, idxs] = li_mop @ (lop @ block)  # assuming by_col!
        U, S, Vh = recovery_fn(
            ro_sketch,
            lo_sketch,
            inner_sketch,
            li_mop,
            ri_mop,
            rcond=lstsq_rcond,
            as_svd=True,
        )
    #
    return U, S, Vh


def seigh(
    lop,
    lop_device,
    lop_dtype,
    outer_dims,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    recovery_type="singlepass",
    lstsq_rcond=1e-6,
    meas_blocksize=None,
    by_mag=True,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """ """
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
        meas_blocksize = max(lop.shape)
    # instantiate outer measurement linop and perform outer measurements
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
    # if not oversampled, solve sketch
    if inner_dims is None:
        ews, evs = recovery_fn(
            ro_sketch,
            ro_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    # if oversampled, perform inner measurements and then solve
    else:
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
# # TRACEPP AND DIAGPP/XDIAG
# ##############################################################################
def hutchpp(
    lop,
    lop_device,
    lop_dtype,
    defl_dims=0,
    extra_gh_meas=0,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    return_diag=True,
):
    """Diagonal and trace sketched approximation via Hutch++.

    As it can be seen in Table 1 from
    `[BN2022] <https://arxiv.org/abs/2201.10684>`_,
    Gaussian noise is less efficient than Rademacher: These functions provide
    the number of measurements needed to ensure an approximation with at
    most ``eps`` error, with a probability of at least ``1 - delta``
    (try e.g. ``eps=0.001, delta=0.001``). We see that the number of samples
    is in the millions, i.e. GH is not sample-efficient for smaller matrices.

    rad = lambda delta, eps: 2 * np.log(2 / delta) / eps**2
    gau = lambda delta, eps: 4 * np.log2(2**0.5 / delta) / eps**2
    """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise BadShapeError("Diag++ expects square operators!")
    dims = h
    if dims == 0:
        raise BadShapeError("Only nonempty operators supported!")
    if meas_blocksize is None:
        meas_blocksize = max(lop.shape)
    # figure out recovery settings
    if defl_dims > dims:
        raise ValueError("More defl rank than max linop rank!")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    if not is_noise_unitnorm:
        warnings.warn(
            "Non-unitnorm noise can be unstable for trace/diag estimation! "
            + "Check output and consider using Rademacher or PhaseNoise.",
            RuntimeWarning,
        )
    if (defl_dims < 0) or (extra_gh_meas < 0):
        raise ValueError("Negative number of measurements?")
    if defl_dims + extra_gh_meas <= 0:
        raise ValueError("Deflation dims and/or GH measurements needed!")
    # without deflation
    if defl_dims <= 0:
        Q, R, Xh = None, None, None
        t_top = 0
        if return_diag:
            d_top = torch.zeros(dims, dtype=lop_dtype, device=lop_device)
    # with deflation
    else:
        # instantiate deflation linop and perform measurements
        mop = dispatcher.mop(
            noise_type,
            (dims, defl_dims),
            seed,
            lop_dtype,
            meas_blocksize,
            register,
        )
        sketch = torch.empty(
            (lop.shape[0], defl_dims), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in mop.get_blocks(lop_dtype, lop_device):
            sketch[:, idxs] = lop @ block  # assuming block is by_col!
        # leveraging A ~= (Q @ Q.H) A + [I - (Q @ Q.H)] A, we hard-compute
        # the first component. Here, X.H = Q.H @ A
        Q, R = qr(sketch, in_place_q=True, return_R=True)
        Xh = Q.conj().T @ lop
        # top diagonal estimate is then diag(Q @ Xh)
        t_top = (Q * Xh.T).sum()  # no conj here!
        if return_diag:
            d_top = (Q * Xh.T).sum(1)  # no conj here!
    # Girard-Hutchinson:
    t_gh = 0
    if return_diag:
        d_gh = torch.zeros_like(d_top)
    # perform any extra Girard-Hutchinson measurements,
    if extra_gh_meas > 0:
        seed_gh = seed + defl_dims + 1
        mop_gh = dispatcher.mop(
            noise_type,
            (dims, extra_gh_meas),
            seed_gh,
            lop_dtype,
            meas_blocksize,
            register,
        )
        #
        for block, idxs in mop_gh.get_blocks(lop_dtype, lop_device):
            # transpose block: all measurements adjoint since Q deflates lop on
            # the left space
            block = block.T  # after transp: (idxs, dims)
            # nonscalar normalization before deflation
            if not is_noise_unitnorm:
                # so gram matrix of (dims, dims) has diag=len(idxs)
                # and adding every subtrace/gh_meas yields unit diagonal.
                block *= (len(idxs) ** 0.5) / block.norm(dim=0)
            # deflate block and perform adj meas
            block_defl = (
                block if Q is None else block - (block @ Q.conj()) @ Q.T
            )
            b_lop = block_defl.conj() @ lop
            t_gh += (block_defl * b_lop).sum() / extra_gh_meas
            if return_diag:
                if is_noise_unitnorm:
                    d_gh += (block * b_lop).sum(0)
                else:
                    d_gh += ((block * b_lop) / (block * block.conj())).sum(0)
        #
        if return_diag:
            d_gh /= extra_gh_meas
    #
    result = {"QR": (Q, R), "tr": (t_top + t_gh, t_top, t_gh)}
    if return_diag:
        result["diag"] = (d_top + d_gh, d_top, d_gh)
    return result


def xdiag(
    lop,
    lop_device,
    lop_dtype,
    x_dims,
    seed=0b1110101001010101011,
    noise_type="rademacher",
    meas_blocksize=None,
    dispatcher=SketchedAlgorithmDispatcher,
    cache_mop=True,
):
    """Diagonal sketched approximation."""
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise BadShapeError("XDiag expects square operators!")
    dims = h
    if dims == 0:
        raise BadShapeError("Only nonempty operators supported!")
    if meas_blocksize is None:
        meas_blocksize = max(lop.shape)
    # figure out recovery settings
    if x_dims > dims:
        raise ValueError("More measurements than max linop rank!")
    if x_dims <= 0:
        raise ValueError("No measurements?")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    if not is_noise_unitnorm:
        warnings.warn(
            "Non-unitnorm noise can be unstable for diagonal estimation! "
            + "Check output and consider using Rademacher or PhaseNoise.",
            RuntimeWarning,
        )
    # instantiate outer measurement linop and perform outer measurements
    mop = dispatcher.mop(
        noise_type,
        (dims, x_dims),
        seed,
        lop_dtype,
        meas_blocksize,
        register,
    )
    # leveraging A ~= (Q @ S @ Q.H) A + [I - (Q @ S @ Q.H)] A, we hard-compute
    # the first component. For that, obtain Q and Xh = S @ Q.H @ A
    if cache_mop:
        mopmat = mop.to_matrix(lop_dtype, lop_device)
        sketch = lop @ mopmat
    else:
        sketch = torch.empty(
            (lop.shape[0], x_dims), dtype=lop_dtype, device=lop_device
        )
        for block, idxs in mop.get_blocks(lop_dtype, lop_device):
            sketch[:, idxs] = lop @ block  # assuming block is by_col!
    Q, R = qr(sketch, in_place_q=False, return_R=True)  # we need sketch
    S = torch.linalg.pinv(R.conj().T)
    S /= torch.linalg.norm(S, dim=0)
    Smat = (S @ S.conj().T) / -x_dims
    Smat[range(x_dims), range(x_dims)] += 1
    Xh = Smat @ (Q.conj().T @ lop)
    # compute top diagonal as preliminary diag (efficient and exact)
    d_top = (Q * Xh.T).sum(1)  # no conj here!
    d_gh = torch.zeros_like(d_top)
    # refine diagonal with deflated (and Xchanged) Girard-Hutchinson
    is_complex = lop_dtype in COMPLEX_DTYPES
    if cache_mop:
        iterator = ((b, range(0, x_dims)) for b in [mopmat])
    else:
        iterator = mop.get_blocks(lop_dtype, lop_device)
    for block, idxs in iterator:
        sketch[:, idxs] -= Q @ (Xh @ block)
        #
        if is_noise_unitnorm and is_complex:
            d_gh += (block.conj() * sketch).sum(1)
        elif is_noise_unitnorm and not is_complex:
            d_gh += (block * sketch).sum(1)
        elif not is_noise_unitnorm and is_complex:
            block_c = block.conj()
            d_gh += ((sketch * block_c) / (block_c * block)).sum(1)
        elif not is_noise_unitnorm and not is_complex:
            d_gh += (sketch / block).sum(1)
        else:
            raise RuntimeError("Should never happen!")
    #
    d_gh /= x_dims
    return (d_top + d_gh), (d_top, d_gh, Q, R)


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
    """Sketched norms.

    For Frobenius, check 6.2 in Tropp 19.
    """
    h, w = lop.shape
    if num_meas <= 0 or num_meas > min(h, w):
        raise ValueError(
            "Measurements must be between 1 and number of rows/columns!"
        )
    if meas_blocksize is None:
        meas_blocksize = max(lop.shape)
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
    supported_norm_types = {"fro", "op"}
    result = {}
    for norm_type in norm_types:
        if norm_type == "op":
            result[norm_type] = torch.linalg.norm(gram, ord=2) ** 0.5
        elif norm_type == "fro":
            result[norm_type] = gram.diag().sum() ** 0.5
        else:
            raise ValueError(
                f"Unsupported norm type! {norm_type}. "
                "Supported: {supported_norm_types}"
            )
    #
    return result, (Q, R, sketch2)


# ##############################################################################
# # TRIANGULAR LINEAR OPERATOR
# ##############################################################################
class TriangularLinOp(BaseLinOp):
    r"""Given a square linop, compute products with one of its triangles.

    The triangle of a linear operator can be approximated from the full operator
    via a "staircase pattern" of exact measurements, whose computation is exact
    and fast. For example, given an operator of shape ``(1000, 1000)``, and
    stairs of size 100, yields 9 exact measurements strictly under the diagonal,
    the first one covering ``lop[100:, :100]``, the next one
    ``lop[200:, 100:200]``, and so on. The more measurements, the more closely
    the full triangle is approximated.

    Note that this staircase pattern leaves a block-triangular section of the
    linop untouched (near the main diagonal). This part can be then estimated
    with the help of  :func:`serrated_hadamard_pattern`, completing the
    triangular approximation, as follows:


    Given a square linear operator :math:`A`, and random vectors
    :math:`v \sim \mathcal{R}` with :math:`\mathbb{E}[v v^T] = I`, consider
    the generalized Hutchinson diagonal estimator:

    .. math::

      f(A) =
      \mathbb{E}_{v \sim \mathcal{R}} \big[ \varphi(v) \odot Av \big]

    In this case, if the :math:`\varphi` function follows a "serrated
    Hadamard pattern", :math:`f(A)` will equal a block-triangular subset of
    :math:`A`.


    :param lop: A square linear operator of order ``dims``, such that
      ``self @ v`` will equal ``triangle(lop) @ v``. It must implement a
      ``lop.shape = (dims, dims)`` attribute as well as the left- and right-
      matmul operator ``@``, interacting with torch tensors.
    :param stair_width: Width of each step in the staircase pattern. If
      it is 1, a total of ``dims`` exact measurements will be performed.
      If it equals ``dims``, no exact measurements will be performed (since
      the staircase pattern would cover the full triangle). The step size
      regulates this trade-off: Ideally, we want as many exact measurements
      as possible, but not too many. If no value is provided, ``dims // 2``
      is chosen by default, such that only 1 exact measurement is performed.
    :param num_hutch_measurements: The leftover entries from the
      staircase measurements are approximated here using an extension of
      the Hutchinson diagonal estimator. This estimator generally requires
      many measurements to be informative, and it can even be counter-
      productive if not enough measurements are given. If ``lop``is not
      diagonally dominant, consider setting this to 0 for a sufficiently
      good approximation via ``staircase_measurements``. Otherwise,
      make sure to provide a sufficiently high number of measurements.
    :param lower: If true, lower triangular matmuls will be computed.
      Otherwise, upper triangular.
    :param with_main_diagonal: If true, the main diagonal will be included
      in the triangle, otherwise excluded. If you already have precomuted
      the diagonal elsewhere, consider excluding it from this approximation,
      and adding it separately.
    :param seed: Seed for the random SSRFT measurements used in the
      Hutchinson estimator.
    :param use_fft: Whether to use FFT for the Hutchinson estimation. See
      :func:`subdiag_hadamard_pattern` for more details.
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
            # meas_blocksize = max(lop.shape)
            # meas_blocksize = stair_width
            meas_blocksize = 3
        self.meas_blocksize = meas_blocksize
        #
        if num_gh_meas <= 0:
            warnings.warn(
                "num_gh_meas <=0: only staircase measurements will be done! "
                + "Set this to a large integer for more accurate estimation",
                RuntimeWarning,
            )
        else:
            if not self.is_noise_unitnorm:
                warnings.warn(
                    "Non-unitnorm noise can be unstable for triangular "
                    "matvecs! Check output and consider using Rademacher "
                    "or PhaseNoise.",
                    RuntimeWarning,
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
        """Helper method to perform serrated Girard-Hutchinson measurements."""
        dtype, device = x.dtype, x.device
        result = torch.zeros_like(x)
        for block, idxs in mop.get_blocks(dtype, device):
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
