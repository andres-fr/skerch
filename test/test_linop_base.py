#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for core linop functionality.

The library implements a ``BaseLinOp`` class, plus core functionality to
transpose linops, convert them to matrices and define matrix-free linops in
terms of vector rows or matrices.

This file tests correctness of the above functionality, plus some corner cases.
"""


import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    BaseLinOp,
    ByVectorLinOp,
    TransposedLinOp,
)
from skerch.utils import BadShapeError, gaussian_noise

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def linop_correctness_shapes(request):
    """Shapes to test linop correctness"""
    result = [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (1, 10),
        (10, 1),
        (10, 10),
        (10, 100),
        (100, 10),
    ]
    if request.config.getoption("--quick"):
        result = result[:7]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
class MatrixAsLinOp(ByVectorLinOp):
    """ """

    def __init__(self, mat, by_row=False):
        """ """
        super().__init__(mat.shape, by_row)
        self.mat = mat

    def get_vector(self, idx, device):
        """ """
        if self.by_row:
            return self.mat[idx]
        else:
            return self.mat[:, idx]


# ##############################################################################
# # TESTS
# ##############################################################################
def test_linop_formal():
    """Test case for input shape consistency and value errors

    For forward and adjoint matmul with ``BaseLinOp``, test:
    * Providing vectors of mismatching shape raises ``BadShapeError``
    * Providing tensors that aren't vectors/matrices raises ``BadShapeError``
    * Trying to transpose a transposedlinop raises ``valueError``

    """
    lop = BaseLinOp((10, 20))
    #
    v1, v2 = torch.empty(lop.shape[0] + 1), torch.empty(lop.shape[1] + 1)
    with pytest.raises(BadShapeError):
        _ = v1 @ lop
    with pytest.raises(BadShapeError):
        _ = lop @ v2
    #
    t1, t2 = torch.empty(0), torch.empty(3, 4, 5)
    with pytest.raises(BadShapeError):
        _ = lop @ t1
    with pytest.raises(BadShapeError):
        _ = lop @ t2
    with pytest.raises(BadShapeError):
        _ = t1 @ lop
    with pytest.raises(BadShapeError):
        _ = t2 @ lop
    #
    lopT = TransposedLinOp(lop)
    with pytest.raises(ValueError):
        _ = TransposedLinOp(lopT)


def test_linop_correctness(
    rng_seeds, torch_devices, dtypes_tols, linop_correctness_shapes
):
    """Test case for correctness of linop matmuls.

    For all devices and datatypes, samples Gaussian noise and checks that:
    * MatrixAsLinOp yields same results as direct matmul. This tests
      correctness of the baselinop->byvectorlinop pipeline.
    * linop_to_matrix yields the original matrix.
    * Same thing but with (Hermitian) transposed linop
    * Double transposed is same as original lop
    """
    for seed in rng_seeds:
        for dtype, tol in dtypes_tols.items():
            for h, w in linop_correctness_shapes:
                for device in torch_devices:
                    mat = gaussian_noise(
                        (h, w), dtype=dtype, device=device, seed=seed
                    )
                    for adj in (True, False):
                        phi = gaussian_noise(
                            (2, h) if adj else (w, 2),
                            dtype=dtype,
                            device=device,
                            seed=2 * seed,
                        )
                        for by_row in (True, False):
                            lop = MatrixAsLinOp(mat, by_row=by_row)
                            mat2 = linop_to_matrix(
                                lop, dtype=dtype, device=device, adjoint=adj
                            )
                            assert (
                                mat == mat2
                            ).all(), f"Wrong linop_to_matrix! {adj, by_row}"
                            # matmat operations
                            matmeas = phi @ mat if adj else mat @ phi
                            lopmeas = phi @ lop if adj else lop @ phi
                            assert torch.allclose(
                                matmeas, lopmeas, atol=tol
                            ), "lop@v does not equal mat@v in mat-mat!"
                            # matvec operations
                            matmeas = phi[0] @ mat if adj else mat @ phi[:, 0]
                            lopmeas = phi[0] @ lop if adj else lop @ phi[:, 0]
                            assert torch.allclose(
                                matmeas, lopmeas, atol=tol
                            ), "lop@v does not equal mat@v in mat-vec!"
                            # now test transposition
                            lopT = lop.t()
                            lopTT = lopT.t()
                            matT = linop_to_matrix(
                                lopT, dtype=dtype, device=device, adjoint=adj
                            )
                            assert (matT == mat.H).all(), "Wrong transp?"
                            assert lopTT is lop, "Wrong double transp?"
