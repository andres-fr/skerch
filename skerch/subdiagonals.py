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
from .linops import CompositeLinOp, OrthProjLinOp, NegOrthProjLinOp


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


def projected_ij(i, j, lop, Q):
    """Retrieves ``(i, j)`` entry from ``(QQ.t) @ lop @ (QQ.t)``.

    :param i: Index between 0 and ``height(proj)``
    :param j: Index between 0 and ``width(proj)``
    :param lop: linear operator
    :param proj: Thin orthonormal matrix
    """
    h, w = lop.shape
    assert h == w, "Only square linear operators supported!"
    vj = Q @ Q[j, :]
    vi = vj if (i == j) else (Q @ Q[i, :])
    result = vi @ (lop @ vj)
    return result


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
    assert h == w, "Only square linear operators supported!"
    dims = h
    result_buff = torch.zeros(dims, dtype=lop_dtype, device=lop_device)
    squares_buff = torch.zeros_like(result_buff)
    top_buff = torch.zeros_like(result_buff)
    ssrft = SSRFT((num_meas, dims), seed=seed)
    #
    if deflation_rank > 0:
        # deflate lop: first compute a few random measurements
        ssrft_defl = SSRFT((num_meas, dims), seed=seed + 12345)
        deflation_matrix = torch.empty(
            (dims, deflation_rank), dtype=lop_dtype, device=lop_device
        )
        for i in range(deflation_rank):
            deflation_matrix[:, i] = lop @ ssrft_defl.get_row(
                i, lop_dtype, lop_device
            )
        # orthogonalize measurements to get deflated lop
        orthogonalize(deflation_matrix, overwrite=True)
        proj = OrthProjLinOp(deflation_matrix)
        negproj = NegOrthProjLinOp(deflation_matrix)
        # deflated_lop = CompositeLinOp(
        #     (("negproj", negproj), ("lop", lop), ("negproj", negproj))
        # )
        deflated_lop = CompositeLinOp((("lop", lop), ("negproj", negproj)))

        for i in range(dims):
            top_buff[i] = (lop @ (deflation_matrix @ deflation_matrix[i]))[i]

    else:
        # no deflation
        deflated_lop = lop

    for i in range(num_meas):
        v = ssrft.get_row(i, lop_dtype, lop_device)

        result_buff += v * (deflated_lop @ v)
        squares_buff += v * v
    result_buff /= squares_buff

    true_diag = torch.diag(lop)

    print("\n TOP>>>>", torch.dist(top_buff, true_diag))
    print("BOTTOM>>>>", torch.dist(result_buff, true_diag))
    print("BOTH>>>>", torch.dist(top_buff + result_buff, true_diag))

    plt.plot(true_diag, color="grey")
    plt.plot(top_buff + result_buff)
    plt.show()
    breakpoint()


# def subroutine(
#     num_meas,
#     lop,
#     lop_dtype,
#     lop_device,
#     seed=0b1110101001010101011,
#     deflation_rank=0,
# ):
#     """Torch dtype to string.

#     Given a PyTorch datatype object, like ``torch.float32``, returns the
#     corresponding string, in this case 'float32'.
#     """
#     h, w = lop.shape
#     assert h == w, "Only square linear operators supported!"
#     dims = h
#     result_buff = torch.zeros(dims, dtype=lop_dtype, device=lop_device)
#     squares_buff = torch.zeros_like(result_buff)
#     ssrft = SSRFT((num_meas, dims), seed=seed)
#     #
#     if deflation_rank > 0:
#         # deflate lop: first compute a few random measurements
#         ssrft_defl = SSRFT((num_meas, dims), seed=seed + 12345)
#         deflation_matrix = torch.empty(
#             (dims, deflation_rank), dtype=lop_dtype, device=lop_device
#         )
#         for i in range(deflation_rank):
#             deflation_matrix[:, i] = lop @ ssrft_defl.get_row(
#                 i, lop_dtype, lop_device
#             )
#         # orthogonalize measurements to get deflated lop
#         orthogonalize(deflation_matrix, overwrite=True)
#         proj = OrthProjLinOp(deflation_matrix)
#         negproj = NegOrthProjLinOp(deflation_matrix)
#         deflated_lop = CompositeLinOp(
#             (("negproj", negproj), ("lop", lop), ("negproj", negproj))
#         )
#     else:
#         # no deflation
#         deflated_lop = lop

#     """
#     OK NOW COMPUTE DIAG ON DEFLATED


#     TODO:

#     DEFLATION FOR NONSYMMETRIC MUST USE DIFFERENT ORTH FRAMES!!

#     CAN WE RECYCLE MEASUREMENTS BETWEEN DEFLATION AND ESTIMATION??

#     SCALE NORMALIZATION: HOW? IS SQUARES_BUFF ENOUGH?

#     DIAGPP IMPLEMENTATION IN JULIA: SOME
#     https://github.com/niclaspopp/RandomizedDiagonalEstimation.jl/blob/master/src/PureDiagonalEstimation.jl
#     """

#     for i in range(num_meas):
#         v = (
#             rademacher_noise(dims, seed=seed + i, device="cpu")
#             .to(lop_dtype)
#             .to(lop_device)
#         )
#         # v = ssrft.get_row(i, lop_dtype, lop_device)
#         result_buff += v * (deflated_lop @ v)  # / (v * v)
#         squares_buff += v * v

#     true_diag = torch.diag(lop)
#     result_buff /= squares_buff

#     top_diag = torch.zeros_like(result_buff)
#     if deflation_rank > 0:
#         for i in range(dims):
#             top_diag[i] = projected_ij(i, i, lop, deflation_matrix)

#     # print("<<<<>>>>", torch.dist(result_buff, torch.diag(lop)))
#     print(
#         "\n<<<< JUST TOP >>>>",
#         torch.dist(top_diag, true_diag),
#     )
#     print(
#         "<<<< JUST BOTTOM >>>>",
#         torch.dist((result_buff), true_diag),
#     )
#     print(
#         "<<<< TOP AND BOTTOM >>>>",
#         torch.dist((top_diag + result_buff), true_diag),
#     )

#     # plt.clf()
#     # plt.plot(cooked - torch.diag(lop))
#     # plt.show()

#     plt.plot(true_diag, color="grey")
#     plt.plot(top_diag + result_buff)
#     plt.show()
#     breakpoint()

#     # torch.diag(lop)
#     # out_buff
#     # plt.plot(result_buff); plt.plot(true_diag); plt.show()
#     # plt.plot(top_diag); plt.plot(true_diag); plt.show()
#     # plt.plot(cooked); plt.plot(true_diag); plt.show()
#     # torch.dist(torch.diag(lop), result_buff)
#     # torch.dist(torch.diag(lop), top_diag)

#     """
#     SANDBOX
#     """
#     true_top_diag = torch.diag(
#         deflation_matrix
#         @ (deflation_matrix.T @ lop @ deflation_matrix)
#         @ deflation_matrix.T
#     )
