#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
* Implementing triang linop:
  - finish init method
  - in test: create fixture and test case
* Add sketchlord(h) facilities
* Integration tests/docs (add utests where needed):
  - comparing all recoveries for general and herm quasi-lowrank on complex128, using all types of noise -> boxplot
  - scale up: good recovery of very large composite linop, quick.
  - priori and posteriori...
* add remaining todos as GH issues and release!





LATER TODO:
* tracepp xtrace and hermitian
* HDF5 measurement/wrapper API
* a-priori/posteriori/truncation stuff
* out-of-core wrappers for QR, SVD, LSTSQ
* sketchlord and sketchlordh.
* what about generalized_nystrom_xdiag?
* sketched permutations


CHANGELOG:
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByVector
  - New measurement linops: Rademacher, Gaussian, Phase, SSRFT
* Sketching API:
  - Modular measurement API supporting multiprocessing and HDF5
  - Modular recovery methods (singlepass, Nystrom, oversampled) for
    general and symmetric low-rank matrices
* Algorithm API:
  - Algorithms: XDiag/DiagPP, XTrace/TracePP, SSVD, Sketchlord, Triangular
  - Efficient support for Hermitian versions
  - Dispatcher for modularized use of noise sources and recovery types
  - Matrix-free a-posteriori error verification

* A-posteriori error verification
* A-priori hyperparameter selection
"""


import warnings
from functools import partial
import torch
from .recovery import singlepass, nystrom, oversampled
from .recovery import singlepass_h, nystrom_h, oversampled_h
from .measurements import get_measvec, lop_measurement, perform_measurements
from .measurements import (
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SsrftNoiseLinOp,
)
from .linops import BaseLinOp, CompositeLinOp, TransposedLinOp
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
        inner_dims = None
        if recovery_type == "singlepass":
            recovery_fn = singlepass_h if hermitian else singlepass
        elif recovery_type == "nystrom":
            recovery_fn = nystrom_h if hermitian else nystrom
        elif "oversampled" in recovery_type:
            recovery_fn = oversampled_h if hermitian else oversampled
            inner_dims = int(recovery_type.split("_")[-1])
        else:
            supported = "singlepass, nystrom, oversampled_12345"
            warnings.warn(
                f"Unknown recovery type! {recovery_type}! "
                "Supported: {supported}"
            )
            recovery_fn = None
        #
        return recovery_fn, inner_dims

    @staticmethod
    def mop(noise_type, hw, seed, dtype, register=False):
        """Returns measurement linop with given specs."""
        if noise_type == "rademacher":
            mop = RademacherNoiseLinOp(
                hw, seed, dtype, by_row=False, register=register
            )
        elif noise_type == "gaussian":
            mop = GaussianNoiseLinOp(
                hw, seed, dtype, by_row=False, register=register
            )
        elif noise_type == "ssrft":
            mop = SsrftNoiseLinOp(hw, seed, norm="ortho")
        elif noise_type == "phase":
            if dtype not in COMPLEX_DTYPES:
                raise ValueError(
                    "Phase noise expects complex dtype! Use Rademacher instead"
                )
            mop = PhaseNoiseLinOp(
                hw, seed, dtype, by_row=False, register=register, conj=False
            )
        else:
            # unknown recovery type
            supported = "rademacher, gaussian, ssrft, phase"
            warnings.warn(
                f"Unknown recovery type! {recovery_type} "
                "Supported: {supported}"
            )
            mop = None
        #
        return mop

    @staticmethod
    def unitnorm_lop_entries(lop_type):
        """Returns True if all ``lop`` entries must have unit norm."""
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
        if not result:
            warnings.warn(
                "Non-unitnorm noise can be unstable for diagonal estimation! "
                + "Check output and consider using Rademacher or PhaseNoise.",
                RuntimeWarning,
            )
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
    max_mp_workers=None,
    lstsq_rcond=1e-6,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """ """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    recovery_fn, inner_dims = dispatcher.recovery(recovery_type, False)
    if (outer_dims > max(h, w)) or (
        inner_dims is not None and (inner_dims > max(h, w))
    ):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims < outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    # instantiate outer measurement linops
    ro_seed = seed
    lo_seed = ro_seed + outer_dims + 1
    ro_mop = dispatcher.mop(
        noise_type, (w, outer_dims), ro_seed, lop_dtype, register
    )
    lo_mop = dispatcher.mop(
        noise_type, (h, outer_dims), lo_seed, lop_dtype, register
    )
    # instantiate inner measurement linops
    if inner_dims is not None:
        ri_seed = lo_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = dispatcher.mop(
            noise_type, (w, inner_dims), ri_seed, lop_dtype, register
        )
        li_mop = dispatcher.mop(
            noise_type, (h, inner_dims), li_seed, lop_dtype, register
        )
    # perform outer measurements
    _, ro_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=ro_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=False,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    _, lo_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=lo_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=True,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    # solve sketches
    if inner_dims is None:
        U, S, Vh = recovery_fn(
            ro_sketch, lo_sketch, ro_mop, rcond=lstsq_rcond, as_svd=True
        )
    # if oversampled, perform inner measurements before solving
    else:
        lop_mop = CompositeLinOp([("lop", lop), ("ri", ri_mop)])
        _, inner_sketch = perform_measurements(
            partial(
                lop_measurement,
                lop=lop_mop,
                meas_lop=li_mop,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(inner_dims),
            adjoint=True,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        U, S, Vh = recovery_fn(
            ro_sketch,
            lo_sketch,
            inner_sketch,
            TransposedLinOp(li_mop),
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
    max_mp_workers=None,
    lstsq_rcond=1e-6,
    by_mag=True,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """ """
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise ValueError("SEIGH expects square operators!")
    dims = h
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    recovery_fn, inner_dims = dispatcher.recovery(recovery_type, True)
    if (outer_dims > dims) or (inner_dims is not None and (inner_dims > dims)):
        raise ValueError("More measurements than rows/columns not supported!")
    if (inner_dims is not None) and inner_dims < outer_dims:
        raise ValueError(
            "Inner dims must be larger than outer for oversampled!"
        )
    # instantiate outer measurement linop and perform outer measurements
    ro_seed = seed
    ro_mop = dispatcher.mop(
        noise_type, (dims, outer_dims), ro_seed, lop_dtype, register
    )
    _, ro_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=ro_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(outer_dims),
        adjoint=False,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    if inner_dims is None:
        # if no oversampling, solve sketch and return
        ews, evs = recovery_fn(
            ro_sketch,
            ro_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    else:
        # if oversampled, perform inner measurements before solving
        ri_seed = ro_seed + outer_dims + 1
        li_seed = ri_seed + inner_dims + 1
        ri_mop = dispatcher.mop(
            noise_type, (dims, inner_dims), ri_seed, lop_dtype, register
        )
        li_mop = dispatcher.mop(
            noise_type, (dims, inner_dims), li_seed, lop_dtype, register
        )
        #
        lop_mop = CompositeLinOp([("lop", lop), ("ri", ri_mop)])
        _, inner_sketch = perform_measurements(
            partial(
                lop_measurement,
                lop=lop_mop,
                meas_lop=li_mop,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(inner_dims),
            adjoint=True,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        ews, evs = recovery_fn(
            ro_sketch,
            inner_sketch,
            TransposedLinOp(li_mop),
            ri_mop,
            rcond=lstsq_rcond,
            as_eigh=True,
            by_mag=by_mag,
        )
    #
    return ews, evs


# ##############################################################################
# # DIAGPP/XDIAG
# ##############################################################################
def diagpp(
    lop,
    lop_device,
    lop_dtype,
    defl_dims=0,
    extra_gh_meas=0,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    max_mp_workers=None,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """Diagonal sketched approximation via Hutch++."""
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise ValueError("XDiag expects square operators!")
    dims = h
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    if defl_dims > dims:
        raise ValueError("defl_dims larger than operator rank!")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    if not is_noise_unitnorm:
        warnings.warn(
            "Non-unitnorm noise can be unstable for diagonal estimation! "
            + "Check output and consider using Rademacher or PhaseNoise.",
            RuntimeWarning,
        )
    if (defl_dims < 0) or (extra_gh_meas < 0):
        raise ValueError("Negative number of measurements?")
    if defl_dims + extra_gh_meas <= 0:
        raise ValueError("Deflation dims and/or GH measurements needed!")
    # deflation:
    if defl_dims <= 0:
        Q, R, Xh = None, None, None
        d_top = torch.zeros(dims, dtype=lop_dtype, device=lop_device)
    else:
        # instantiate deflation linop and perform measurements
        mop = dispatcher.mop(
            noise_type, (dims, defl_dims), seed, lop_dtype, register
        )
        _, sketch = perform_measurements(
            partial(
                lop_measurement,
                lop=lop,
                meas_lop=mop,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(defl_dims),
            adjoint=False,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        # leveraging A ~= (Q @ Q.H) A + [I - (Q @ Q.H)] A, we hard-compute
        # the first component. Here, X.H = Q.H @ A
        Q, R = qr(sketch, in_place_q=False, return_R=True)
        _, Xh = perform_measurements(
            partial(
                lop_measurement,
                lop=lop,
                meas_lop=Q,
                device=lop_device,
                dtype=lop_dtype,
            ),
            range(defl_dims),
            adjoint=True,
            parallel_mode=parallel_mode,
            compact=True,
            max_mp_workers=max_mp_workers,
        )
        # top diagonal estimate is then Q @ X.H
        d_top = (Q * Xh.T).sum(1)  # here only Xh.T!
    # Girard-Hutchinson:
    # if we deflated, also recycle measurements for estimator
    d_defl = torch.zeros_like(d_top)
    for i in range(defl_dims):
        v_i = get_measvec(i, mop, lop_device, lop_dtype)
        w_i = sketch[:, i] - (Q @ (Xh @ v_i))
        if is_noise_unitnorm:
            d_defl += v_i * w_i
        else:
            d_defl += (v_i * w_i) / (v_i * v_i)
    # perform any extra Girard-Hutchinson measurements,
    # assumed to not fit in memory so done sequentially
    if extra_gh_meas > 0:
        seed_gh = seed + defl_dims + 1
        mop_gh = dispatcher.mop(
            noise_type, (dims, extra_gh_meas), seed_gh, lop_dtype, register
        )
        for i in range(extra_gh_meas):
            v_i = get_measvec(i, mop_gh, lop_device, lop_dtype)
            meas_i = lop @ v_i
            if Q is None:
                w_i = lop @ v_i
            else:
                meas_i = lop @ v_i
                w_i = meas_i - Q @ (meas_i.conj() @ Q).conj()
            if is_noise_unitnorm:
                d_defl += v_i.conj() * w_i
            else:
                v_i_c = v_i.conj()
                d_defl += (v_i_c * w_i) / (v_i_c * v_i)
    d_defl /= defl_dims + extra_gh_meas
    #
    return (d_top + d_defl), (d_top, d_defl, Q, R)


def xdiag(
    lop,
    lop_device,
    lop_dtype,
    defl_dims,
    seed=0b1110101001010101011,
    noise_type="ssrft",
    max_mp_workers=None,
    dispatcher=SketchedAlgorithmDispatcher,
):
    """Diagonal sketched approximation."""
    register = False  # set to True for seed debugging
    h, w = lop.shape
    if h != w:
        raise ValueError("XDiag expects square operators!")
    dims = h
    # figure out parallel mode and recovery settings
    parallel_mode = None if max_mp_workers is None else "mp"
    if defl_dims > dims:
        raise ValueError("defl_dims larger than operator rank!")
    if defl_dims <= 0:
        raise ValueError("No measurements?")
    #
    is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
    # instantiate outer measurement linop and perform outer measurements
    ro_seed = seed
    ro_mop = dispatcher.mop(
        noise_type, (dims, defl_dims), ro_seed, lop_dtype, register
    )
    _, ro_sketch = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=ro_mop,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(defl_dims),
        adjoint=False,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    # leveraging A ~= (Q @ Q.H) A + [I - (Q @ Q.H)] A, we hard-compute
    # the first component. For DiagPP, X.H = Q.H @ A
    Q, R = qr(ro_sketch, in_place_q=False, return_R=True)
    _, Xh = perform_measurements(
        partial(
            lop_measurement,
            lop=lop,
            meas_lop=Q,
            device=lop_device,
            dtype=lop_dtype,
        ),
        range(defl_dims),
        adjoint=True,
        parallel_mode=parallel_mode,
        compact=True,
        max_mp_workers=max_mp_workers,
    )
    # For XDiag, X = I - (1/defl_dims)*(S @ S.H), where S is explained in
    # the Xtrace paper, section 2.1. Then, Xh becomes Z @ Q.H @ A
    S = torch.linalg.pinv(R.conj().T)
    S /= torch.linalg.norm(S, dim=0)
    Z = (S @ S.conj().T) / -defl_dims
    Z[range(defl_dims), range(defl_dims)] += 1
    Xh = Z @ Xh
    # top diagonal estimate is then Q @ X.H
    d_top = (Q * Xh.T).sum(1)
    d_defl = torch.zeros_like(d_top)
    # it remains to estimate the deflated (and optionally Xchanged) part via
    # Girard-Hutchinson estimator.
    for i in range(defl_dims):
        v_i = get_measvec(i, ro_mop, lop_device, lop_dtype)
        w_i = ro_sketch[:, i] - (Q @ (Xh @ v_i))
        if is_noise_unitnorm:
            d_defl += v_i.conj() * w_i
        else:
            v_i_c = v_i.conj()
            d_defl += (v_i_c * w_i) / (v_i_c * v_i)
    d_defl /= defl_dims
    #
    return (d_top + d_defl), (d_top, d_defl, Q, R)


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
        with_main_diagonal=True,
        use_fft=True,
        #
        seed=0b1110101001010101011,
        noise_type="rademacher",
        max_mp_workers=None,
        dispatcher=SketchedAlgorithmDispatcher,
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
        self.max_mp_workers = max_mp_workers
        self.dispatcher = dispatcher
        #
        if num_gh_meas <= 0:
            warnings.warn(
                "num_gh_meas <=0: only staircase measurements will be done! "
                + "Increase this parameter for more accurate estimation",
                RuntimeWarning,
            )
        else:
            is_noise_unitnorm = dispatcher.unitnorm_lop_entries(noise_type)
            if not is_noise_unitnorm:
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
        adjoint,
        stair_width,
        with_main_diag,
        lower=True,
        use_fft=True,
    ):
        """Helper method to perform serrated Girard-Hutchinson measurements."""
        device, dtype = x.device, x.dtype
        result = torch.zeros_like(x)
        num_meas = mop.shape[1]
        normalizer = torch.zeros_like(x)
        for i in range(num_meas):
            m = get_measvec(i, mop, device, dtype)
            pattern = serrated_hadamard_pattern(
                m, stair_width, with_main_diag, lower, use_fft
            )
            if adjoint:
                result += pattern * ((m * x) @ lop)
            else:
                result += pattern * (lop @ (m * x))
            normalizer += m * m
        #
        result = result / normalizer
        return result

    def _matmul_helper(self, x, adjoint=False):
        """Forward and adjoint triangular matrix multiplications.

        Since forward and adjoint matmul share many common computations, this
        method implements both at the same time. The specific mode can be
        dispatched using the ``adjoint`` parameter.
        """
        self.check_input(x, self.lop.shape, adjoint=adjoint)
        # we don't factorize this method because we want to share buff
        # across both loops to hopefully save memory
        buff = torch.zeros_like(x)
        result = torch.zeros_like(x)
        # add step computations to result
        for beg, end in self._iter_stairs(self.dims, self.stair_width):
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
                breakpoint()
                # buff[end:] = x[end:]
                # result[beg:end] += (self.lop @ buff)[beg:end]
                # buff[end:] = 0
            elif adjoint and (not self.lower):
                breakpoint()
                # buff[beg:end] = x[beg:end]
                # result[end:] += (buff @ self.lop)[end:]
                # buff[beg:end] = 0
            else:
                raise RuntimeError("This should never happen")
        #
        if self.n_gh > 0:
            mop = self.dispatcher.mop(
                self.noise_type, (self.dims, self.n_gh), self.seed, x.dtype
            )
            result += self._gh_meas(
                x,
                self.lop,
                mop,
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
        return self._matmul_helper(x, adjoint=False)

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``.

        See parent class for more details.
        """
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


# ##############################################################################
# # TRACEHPP/XTRACEH
# ##############################################################################
