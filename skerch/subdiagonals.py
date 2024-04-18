#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Sketched estimation of any (sub-)diagonals.

TEST: setup a linop

Implement the subdiag at all.

1. Measurements: either ssrft or rademacher
2. store the measurements somewhere, or not??
3. just do the thingy for main diag and test.
4. extend to any combination of subdiags
"""


import torch
from .utils import rademacher_noise

from .distributed_measurements import ssrft_idx_torch
from .ssrft import SSRFT
from .distributed_decompositions import orthogonalize


import matplotlib.pyplot as plt


# ##############################################################################
# # STR
# ##############################################################################
def do_stuff(
    num_meas,
    lop,
    lop_dtype,
    lop_device,
    seed=0b1110101001010101011,
    deflation_rank=0,
):
    """
    HERE WE SHOULD ALSO HAVE A PARAM FOR DEFLATION
    """
    subroutine(num_meas, lop, lop_dtype, lop_device, seed, deflation_rank)


def subroutine(
    num_meas,
    lop,
    lop_dtype,
    lop_device,
    seed=0b1110101001010101011,
    deflation_rank=0,
):
    """Torch dtype to string.

    Given a PyTorch datatype object, like ``torch.float32``, returns the
    corresponding string, in this case 'float32'.
    """
    h, w = lop.shape
    diag_len = min(h, w)
    result_buff = torch.empty(diag_len, dtype=lop_dtype, device=lop_device)
    ssrft = SSRFT((num_meas, w), seed=seed)
    #
    if deflation_rank > 0:
        ssrft_defl = SSRFT((num_meas, w), seed=seed + 1)
        deflation_matrix = torch.empty(
            (h, deflation_rank), dtype=lop_dtype, device=lop_device
        )
        for i in range(deflation_rank):
            deflation_matrix[:, i] = lop @ ssrft_defl.get_row(
                i, lop_dtype, lop_device
            )
        orthogonalize(deflation_matrix, overwrite=True)
        breakpoint()
        """
        HERE DO THE TOP MEASUREMENTS, QR DECOMP, AND STORE INITIAL
        DIAG GUESS
        """
        breakpoint()

    # in_buff = torch.zeros(
    #     h if adjoint else w, dtype=lop_dtype, device=lop_device
    # )
    # out_buff = torch.empty(
    #     w if adjoint else h, dtype=lop_dtype, device=lop_device
    # )

    for i in range(num_meas):
        v = ssrft.get_row(i, lop_dtype, lop_device)
        v[:]
        # v = (
        #     rademacher_noise(in_dim, seed=seed + i, device="cpu")
        #     .to(lop_dtype)
        #     .to(lop_device)
        # )
        meas = (v @ lop) if adjoint else (lop @ v)
        result_buff[:] += v[:diag_len] * meas
        # in_buff[:] = ssrft.get_row(i, lop_dtype, lop_device)
        # breakpoint()
        # in_buff[:] = (
        #     rademacher_noise(in_dim, seed=seed + i, device="cpu")
        #     .to(lop_dtype)
        #     .to(lop_device)
        # )
        # out_buff += in_buff * ((in_buff @ lop) if adjoint else (lop @ in_buff))

    # result_buff /= num_meas
    breakpoint()

    # torch.diag(lop)
    # out_buff
    # plt.plot(result_buff); plt.plot(torch.diag(lop)); plt.show()

    ssrft = SSRFT((in_dim, num_meas))
    # linmeas_idx_torch(idx, lop, meas_lop, lop_device, lop_dtype, adjoint=False)
    # rademacher(x, seed=None, inplace=True, rng_device="cpu")
