#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""General utilities for the ``skerch`` library."""


import torch
import numpy as np
import scipy

# ##############################################################################
# # DTYPES
# ##############################################################################
REAL_DTYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
}

COMPLEX_DTYPES = {
    torch.complex32,
    torch.complex64,
    torch.complex128,
}


def torch_dtype_as_str(dtype):
    """Torch dtype to string.

    Given a PyTorch datatype object, like ``torch.float32``, returns the
    corresponding string, in this case 'float32'.
    """
    full_str = str(dtype)
    dot_idx = full_str.index(".")
    result = full_str[dot_idx + 1 :]
    return result


def complex_dtype_to_real(dtype):
    """"""
    out_dtype = None
    if dtype in REAL_DTYPES:
        out_dtype = dtype
    elif dtype == torch.complex128:
        out_dtype = torch.float64
    elif dtype == torch.complex64:
        out_dtype = torch.float32
    elif dtype == torch.complex32:
        out_dtype = torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    return out_dtype


# ##############################################################################
# # ERRORS
# ##############################################################################
class BadShapeError(Exception):
    """Error to be thrown when a shape is not as it should."""

    pass


class BadSeedError(Exception):
    """Error to be thrown when a random seed is not as it should."""

    pass


# ##############################################################################
# # REPRODUCIBLE NOISE
# ##############################################################################
def uniform_noise(shape, seed=None, dtype=torch.float64, device="cpu"):
    """Reproducible ``torch.rand`` uniform noise.

    :returns: A tensor of given shape, dtype and device, containing uniform
      random noise between 0 and 1 (analogous to ``torch.rand``), but with
      reproducible behaviour fixed to given random seed.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    noise = torch.rand(shape, generator=rng, dtype=dtype, device=device)
    return noise


def gaussian_noise(
    shape, mean=0.0, std=1.0, seed=None, dtype=torch.float64, device="cpu"
):
    """Reproducible ``torch.normal`` Gaussian noise.

    :returns: A tensor of given shape, dtype and device, containing gaussian
      noise with given mean and std (analogous to ``torch.normal``), but with
      reproducible behaviour fixed to given random seed.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    #
    noise = torch.zeros(shape, dtype=dtype, device=device)
    noise.normal_(mean=mean, std=std, generator=rng)
    return noise


def rademacher_noise(shape, seed=None, device="cpu"):
    """Reproducible Rademacher noise.

    .. note::
      This function makes use of :func:`uniform_noise` to sample the Rademacher
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.

    .. warning::
      PyTorch does not ensure RNG reproducibility across
      devices. This parameter determines the device to generate the noise from.
      If you want cross-device reproducibility, make sure that the noise is
      always generated from the same device.

    :param shape: Shape of the output tensor with Rademacher noise.
    :param seed: Seed for the randomness.
    :param device: Device of the output tensor and also source for the noise.
      See warning.
    """
    noise = (
        uniform_noise(shape, seed=seed, dtype=torch.float32, device=device)
        > 0.5
    ) * 2 - 1
    return noise


def randperm(n, seed=None, device="cpu", inverse=False):
    """Reproducible randperm of ``n`` integers from  0 to (n-1) (both included).

    :param bool inverse: If False, a random permutation ``P`` is provided. If
      true, an inverse permutation ``Q`` is provided, such that both
      permutations are inverse to each other, i.e. ``v == v[P][Q] == v[Q][P]``.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    #
    perm = torch.randperm(n, generator=rng, device=device)
    if inverse:
        # we take the O(N) approach since we anticipate large N
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        perm = inv
    return perm


def phase_noise(
    shape, seed=None, dtype=torch.complex128, device="cpu", conj=False
):
    """Reproducible noise uniformly distributed on the complex unit circle.

    .. note::
      This function makes use of :func:`uniform_noise` to sample the phase
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.

    :param conj: If true, the generated noise is the complex conjugate of the
      noise if all other parameters were equal.
      i.e. the complex noise is complex conjugated.
    :returns: A tensor of given shape, dtype and device, containing complex
      i.i.d. noisy entries uniformly distributed around the unit circle.
      Behaviour is reproducible for a given random seed.
    """
    if dtype not in {torch.complex32, torch.complex64, torch.complex128}:
        raise ValueError(f"Dtype must be complex! was {dtype}")
    real_dtype = complex_dtype_to_real(dtype)
    #
    noise = uniform_noise(shape, seed, real_dtype, device)
    if conj:
        noise = 1 - noise
    noise = noise.mul(2 * torch.pi * 1j).exp()
    return noise


def rademacher_flip(x, seed=None, inplace=True, rng_device="cpu"):
    """Reproducible random sign flip using Rademacher noise.

    .. note::
      This function makes use of :func:`uniform_noise` to sample the Rademacher
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.

    .. warning::
      See :func:`rademacher_noise` for notes on reproducibility and more info.
    """
    mask = rademacher_noise(x.shape, seed, device=rng_device).to(x.device)
    if inplace:
        x *= mask
        return x, mask
    else:
        return x * mask, mask


def phase_shift(x, seed=None, inplace=True, rng_device="cpu", conj=False):
    """Reproducible phase shift using phase noise.

    .. note::
      This function makes use of :func:`uniform_noise` to sample the Rademacher
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.

    .. warning::
      See :func:`rademacher_noise` for notes on reproducibility and more info.
    """
    shift = phase_noise(
        x.shape, seed, x.dtype, device=rng_device, conj=conj
    ).to(x.device)
    if inplace:
        x *= shift
        return x, shift
    else:
        return x * shift, shift


# ##############################################################################
# # MATRIX ROUTINE WRAPPERS
# ##############################################################################
def qr(A, in_place_q=False, return_R=False):
    """Thin QR-decomposition of given matrix.

    :param A: Matrix to orthogonalize, needs to be compatible with either
      ``scipy.linalg.qr`` or ``torch.linalg.qr``. It must be square or tall.
    :param in_place_q: If true, ``A[:] = Q`` will be performed.
    :returns: If ``return_R`` is true, returns ``(Q, R)`` such that ``Q``
      has orthonormal columns, ``R`` is upper triangular and ``A = Q @ R``
      as per the QR decomposition. Otherwise, returns just ``Q``.
    """
    h, w = A.shape
    if h < w:
        raise ValueError("Only non-fat matrices supported!")
    #
    if isinstance(A, torch.Tensor):
        Q, R = torch.linalg.qr(A, mode="reduced")
    else:
        # TODO: support pivoting in all modalities
        Q, R = scipy.linalg.qr(A, mode="economic", pivoting=False)
    #
    if in_place_q:
        A[:] = Q
        Q = A
    if return_R:
        return Q, R
    else:
        return Q


def pinv(A):
    """Pseudo-inversion of a given matrix.

    :param A: matrix to pseudo-invert, of shape ``(h, w)``. It needs to be
      compatible with either ``scipy.linalg.pinv`` or ``torch.linalg.qr``.
    :returns: Pseudoinverse of ``A`` with shape ``(w, h)``.
    """
    if isinstance(A, torch.Tensor):
        result = torch.linalg.pinv(A)
    else:
        result = scipy.linalg.pinv(A)
    return result


def lstsq(A, b, rcond=1e-6):
    """Least-squares solver.

    :returns: ``x`` such that ``frob(Ax - b)`` is minimized.
    """
    if isinstance(A, torch.Tensor):
        # do not use default gelsy driver: nondeterm results yielding errors
        driver = "gels" if b.device.type == "cuda" else "gelsd"
        result = torch.linalg.lstsq(A, b, rcond=rcond, driver=driver).solution
    else:
        result = scipy.linalg.lstsq(A, b, cond=rcond, lapack_driver="gelsd")[0]
    return result


def svd(A):
    """Singular Value Decomposition.

    :returns: The SVD ``(U, S, Vh)`` such that ``A = U @ diag(S) @ Vh``.
    """
    if isinstance(A, torch.Tensor):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    else:
        U, S, Vh = scipy.linalg.svd(A, full_matrices=False)
    return U, S, Vh


def eigh(A, by_descending_magnitude=True):
    """Hermitian Eigendecomposition.

    :param by_descending_magnitude: If true, eigenpairs are given by descending
      magnitude of eigenvalues (e.g. -4, 3, 0.1, -0.001, 0). If false,
      eigenpairs are given by descending value (e.g. 3, 0.1, 0, -0.001, -4).
    :returns: The eigendecomposition ``(Lambda, Q)`` such that
      ``A = Q @ diag(Lambda) @ Q.H``.
    """
    # compute EIGH
    if isinstance(A, torch.Tensor):
        ews, evs = torch.linalg.eigh(A)
        idxs = (abs(ews) if by_descending_magnitude else ews).argsort().flip(0)
    else:
        ews, evs = scipy.linalg.eigh(A)
        idxs = (abs(ews) if by_descending_magnitude else ews).argsort()[::-1]
    # sort eigenpairs and return
    ews, evs = ews[idxs], evs[:, idxs]
    return ews, evs


def htr(x, in_place=False):
    """Hermitian transposition wrapper.

    This convenience wrapper exists for several reasons:
    * While torch supports `.H`, numpy does not.
    * In multiprocessing settings, `.conj()` seems to sometimes not work,
      which is likely related to in_place/view/copy behaviour.
    * Transposition of vectors via `.T` throws a warning since it is a no-op.

    This function avoids all three issues, by returning the input as a
    conjugate, and also transposed if it is a matrix.

    :param x: Numpy or Torch object, expected to be a vector or matrix
      (undefined behaviour otherwise).
    :param in_place: If True, the imaginary entries are flipped in-place.
      Otherwise, a new copy of the input is always returned. No in-between
      View behaviour is possible (thus this function may be suboptimal
      in some circumstances, but avoids multiprocessing issues).
    :returns: The Hermitian transpose of ``x`` (if matrix), or the compex
      conjugate if vector. Undefined otherwise.
    """
    x = x.transpose(0, -1) if isinstance(x, torch.Tensor) else x.T
    if isinstance(x, torch.Tensor):
        if not in_place:
            x = x.clone()
        #
        try:  # conj() seems buggy in multiprocessing contexts
            x.imag *= -1
        except RuntimeError:
            pass  # x is not complex, ignore
    # numpy array-like
    else:
        # conj() here works, but often returns a copy
        if not in_place:
            xconj = x.conj()
            x = xconj.copy() if np.shares_memory(x, xconj) else xconj
        else:
            try:
                x.imag *= -1
            except ValueError:
                pass  # x is not complex, ignore
    #
    return x


# ##############################################################################
# # MEASUREMENT HADAMARD PATTERNS
# ##############################################################################
def subdiag_hadamard_pattern(v, diag_idxs, use_fft=False):
    r"""Shifted copies of vectors for subdiagonal Hutchinson estimation.

    Given a square linear operator :math:`A`, and random vectors
    :math:`v \sim \mathcal{R}` with :math:`\mathbb{E}[v v^H] = I`, consider
    this generalized formulation of the Hutchinson diagonal estimator:

    .. math::

      f(A) =
      \mathbb{E}_{v \sim \mathcal{R}} \big[ \varphi(\bar{v}) \odot Av \big]

    If the :math:`\varphi` function is the identity, then :math:`f(A)` equals
    the main diagonal of :math:`A`. If e.g. :math:`\varphi` shifts the entries
    in :math:`v` downwards by ``k`` positions, we get the ``k``-th subdiagonal.

    .. seealso::

      `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston and Yuji
      Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds
      and an improved algorithm”*.  CoRR abs/2201.10684.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param diag_idxs: Iterator with integers corresponding to the subdiagonal
      indices to include, e.g. 0 corresponds to the main diagonal, 1 to the
      diagonal above, -1 to the diagonal below, and so on.
    :param use_fft: If false, shifted copies of ``v`` are pasted on the result.
      This requires only ``len(v)``  memory, but has ``len(v) * len(diag_idxs)``
      time complexity. If this argument is true, an FFT convolution is used
      instead. This requires at least ``4 * len(v)`` memory, but the arithmetic
      has a complexity of ``len(v) * log(len(v))``, which can be advantageous
      whenever ``len(diag_idxs)`` becomes very large.
    :returns: A vector of same shape, type and device as ``v``, composed of
      shifted copies of ``v`` as given by ``diag_idxs``.
    """
    if len(diag_idxs) <= 0:
        raise ValueError("Empty diag_idxs?")
    len_v = len(v)
    if use_fft:
        fft = torch.fft.fft if v.dtype in COMPLEX_DTYPES else torch.fft.rfft
        ifft = torch.fft.ifft if v.dtype in COMPLEX_DTYPES else torch.fft.irfft
        # create a buffer of zeros to avoid circular conv and store the
        # convolutional impulse response
        buff = torch.zeros(2 * len_v, dtype=v.dtype, device=v.device)
        # padded FFT to avoid circular convolution
        buff[:len_v] = v
        V = fft(buff)
        # now we can write the impulse response on buff
        buff[:len_v] = 0
        for idx in diag_idxs:
            buff[idx] = 1
        # non-circular FFT convolution:
        V *= fft(buff)
        V = ifft(V)[:len_v]
        return V
    else:
        result = torch.zeros_like(v)
        for idx in diag_idxs:
            if idx == 0:
                result += v
            elif idx > 0:
                result[idx:] += v[:-idx]
            else:
                result[:idx] += v[-idx:]
        return result


def serrated_hadamard_pattern(
    v, blocksize, with_main_diagonal=True, lower=True, use_fft=False
):
    """Shifted copies of vectors for block-triangular Hutchinson estimation.

    :param v: Torch vector expected to contain zero-mean, uncorrelated entries.
    :param with_main_diagonal: If true, the main diagonal will be included
          in the patterns, otherwise excluded.
    :param lower: If true, the block-triangles will be below the diagonal,
      otherwise above.
    :param use_fft: See :func:`subdiag_hadamard_pattern`.
    :returns: A vector of same shape, type and device as ``v``, composed of
      shifted copies of ``v`` following a block-triangular (serrated) pattern.

    For example, given a 10-dimensional vector, the corresponding serrated
    pattern with ``blocksize=3, with_main_diagonal=True, lower=True`` yields
    the following entries:

    * ``v1``
    * ``v1 + v2``
    * ``v1 + v2 + v3``
    * ``v4``
    * ``v4 + v5``
    * ``v4 + v5 + v6``
    * ``v7``
    * ``v7 + v8``
    * ``v7 + v8 + v9``
    * ``v10``

    If main diagonal is excluded, it will look like this instead:

    * ``0``
    * ``v1``
    * ``v1 + v2``
    * ``0``
    * ``v4``
    * ``v4 + v5``
    * ``0``
    * ``v7``
    * ``v7 + v8``
    * ``0``

    And if ``lower=False``, it will look like this instead:
    * ``v3 + v2 + v1``
    * ``v3 + v2``
    * ``v3``
    * ``v6 + v5 + v4``
    * ``v6 + v5``
    * ``v6``
    * ``v9 + v8 + v7``
    * ``v9 + v8``
    * ``v9``
    * ``v10``

    * ``v1``
    * ``v4 + v3 + v2``
    * ``v4 + v3``
    * ``v4``
    * ``v7 + v6 + v5``
    * ``v7 + v6``
    * ``v7``
    * ``v10 + v9 + v8``
    * ``v10 + v9``
    * ``v10``
    """
    len_v = len(v)
    if blocksize < 1 or blocksize > len_v:
        raise ValueError("Block size must be an integer from 1 to len(v)!")
    #

    if use_fft:
        if lower:
            idxs = range(len_v) if with_main_diagonal else range(1, len_v)
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, blocksize):
                mark = i + blocksize
                offset = sum(v[i:mark])
                result[mark:] -= offset
        else:
            idxs = (
                range(0, -len_v, -1)
                if with_main_diagonal
                else range(-1, -len_v, -1)
            )
            result = subdiag_hadamard_pattern(v, idxs, use_fft=True)
            for i in range(0, len_v, blocksize):
                mark = len_v - (i + blocksize)
                offset = sum(v[mark : (mark + blocksize)])
                result[:mark] -= offset
    else:
        if with_main_diagonal:
            result = v.clone()
        else:
            result = torch.zeros_like(v)
        #
        for i in range(len_v - 1):
            block_n, block_i = divmod(i + 1, blocksize)
            if block_i == 0:
                continue
            # get indices for result[out_beg:out_end] = v[beg:end]
            if lower:
                beg = block_n * blocksize
                end = beg + block_i
                out_end = min(beg + blocksize, len_v)
                out_beg = out_end - block_i
            else:
                end = len_v - (block_n * blocksize)
                beg = end - block_i
                out_beg = max(0, end - blocksize)
                out_end = out_beg + block_i
            # add to serrated pattern
            result[out_beg:out_end] += v[beg:end]
    #
    return result
