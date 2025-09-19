#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for core linop functionality.

The library implements a ``BaseLinOp`` class, plus core functionality to
transpose linops, convert them to matrices and define matrix-free linops in
terms of vector rows or matrices.

This file tests correctness of the above functionality, plus some corner cases.
"""


from time import time
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
# class MatrixAsLinOp(ByVectorLinOp):
#     """ """

#     def __init__(self, mat, by_row=False):
#         """ """
#         super().__init__(mat.shape, by_row)
#         self.mat = mat

#     def get_vector(self, idx, device):
#         """ """
#         if self.by_row:
#             return self.mat[idx]
#         else:
#             return self.mat[:, idx]


class MatMatLinOp(BaseLinOp):
    """ """

    def __init__(self, mat, batch=None):
        """ """
        super().__init__(mat.shape, batch)
        self.mat = mat

    def matmul(self, x):
        """ """
        return self.mat @ x

    def rmatmul(self, x):
        """ """
        return x @ self.mat


class MatVecLinOp(BaseLinOp):
    """ """

    def __init__(self, mat, batch=None):
        """ """
        super().__init__(mat.shape, batch)
        self.mat = mat

    def vecmul(self, x):
        """ """
        return self.mat @ x

    def rvecmul(self, x):
        """ """
        return x @ self.mat


# ##############################################################################
# # TESTS
# ##############################################################################
def test_baselinop_formal():
    """Test case for input shape consistency and value errors

    For forward and adjoint matmul with ``BaseLinOp``, test:
    * Providing empty or malformed shapes raises error
    * Providing vectors of mismatching shape raises ``BadShapeError``
    * Providing tensors that aren't vectors/matrices raises ``BadShapeError``
    * Expected repr behaviour
    * Trying to transpose a transposedlinop raises ``ValueError``
    * No matmat implemented raises warning/error when running on matrices
    * Runtime: matvec > batched > matmat by a substantial amount
    """
    with pytest.raises(ValueError):
        _ = BaseLinOp("NOT A SHAPE")
    with pytest.raises(ValueError):
        _ = BaseLinOp(123)
    with pytest.raises(ValueError):
        _ = BaseLinOp([123])
    with pytest.raises(ValueError):
        _ = BaseLinOp(None)
    with pytest.raises(ValueError):
        _ = BaseLinOp((3, 3, 3))
    with pytest.raises(BadShapeError):
        _ = BaseLinOp((0, 0))
    with pytest.raises(BadShapeError):
        _ = BaseLinOp((0, 3))
    with pytest.raises(BadShapeError):
        _ = BaseLinOp((3, 0))
    #
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
    # repr
    lop = BaseLinOp((5, 5), batch=None)
    assert str(lop) == "<BaseLinOp(5x5)>", "Wrong baselinop repr!"
    lop = BaseLinOp((5, 5), batch=3)
    assert (
        str(lop) == "<BaseLinOp(5x5), batch=3>"
    ), "Wrong baselinop repr! (batch)"
    # twice transposed raises error
    lopT = TransposedLinOp(lop)
    with pytest.raises(ValueError):
        _ = TransposedLinOp(lopT)
    # testing (batched) matmul and matvec
    h, w, n, b = 100, 100, 5000, 11
    slow_factor = 5
    atol = 1e-5
    mat = gaussian_noise((h, w), dtype=torch.float32, device="cpu", seed=12345)
    v1, v2 = torch.ones(w), torch.ones(h)
    a1, a2 = torch.ones(w, n), torch.ones(n, h)
    mmlop = MatMatLinOp(mat, batch=None)
    mmlop2 = MatMatLinOp(mat, batch=b)
    mvlop = MatVecLinOp(mat, batch=None)
    mvlop2 = MatVecLinOp(mat, batch=b)
    #  no matmat raises a warning when running against matrices
    with pytest.warns(RuntimeWarning):
        mvlop @ a1
    with pytest.warns(RuntimeWarning):
        a2 @ mvlop
    # no matmat crashes when batch was requested
    with pytest.raises(RuntimeError):
        _ = mvlop2 @ a1
    with pytest.raises(RuntimeError):
        _ = a2 @ mvlop2
    # matmat is fastest, matvec slowest, batched inbetween
    # also, all give same result
    t0 = time()
    mmlop @ a1  # matmat
    t1 = time()
    mmlop2 @ a1  # batched
    t2 = time()
    mvlop @ a1  # matvec
    t3 = time()
    assert (slow_factor * (t1 - t0)) < (t2 - t1), "Matmat slower than batched?"
    assert (slow_factor * (t2 - t1)) < (t3 - t2), "Batched slower than matvec?"
    t0 = time()
    a2 @ mmlop  # matmat
    t1 = time()
    a2 @ mmlop2  # batched
    t2 = time()
    a2 @ mvlop  # matvec
    t3 = time()
    assert (slow_factor * (t1 - t0)) < (
        t2 - t1
    ), "Matmat slower than batched? (adj)"
    assert (slow_factor * (t2 - t1)) < (
        t3 - t2
    ), "Batched slower than matvec? (adj)"


def test_baselinop_correctness(
    rng_seeds, torch_devices, dtypes_tols, linop_correctness_shapes
):
    """Test case for correctness of linop matmuls.

    For all devices and datatypes, samples Gaussian noise and checks that:
    * linop_to_matrix yields the original matrix.
    * Same thing but with (Hermitian) transposed linop
    * Double transposed is same as original lop


    TODO:
    * New correctness test for byvector: sample matrices row by row with a seed
      and then check that it is the same thing but memsize is low (by_row T/F)
    * Then, go over the whole API and ensure matmat compatibility
    """
    for seed in rng_seeds:
        for dtype, tol in dtypes_tols.items():
            for h, w in linop_correctness_shapes:
                for device in torch_devices:
                    mat = gaussian_noise(
                        (h, w), dtype=dtype, device=device, seed=seed
                    )
                    lop1 = MatMatLinOp(mat, batch=None)
                    lop2 = MatMatLinOp(mat, batch=3)
                    lop3 = MatVecLinOp(mat, batch=None)
                    for adj in (True, False):
                        msg = " (adj)" if adj else ""
                        mat1 = linop_to_matrix(
                            lop1, dtype=dtype, device=device, adjoint=adj
                        )
                        mat2 = linop_to_matrix(
                            lop2, dtype=dtype, device=device, adjoint=adj
                        )
                        mat3 = linop_to_matrix(
                            lop3, dtype=dtype, device=device, adjoint=adj
                        )
                        assert torch.allclose(
                            mat, mat1
                        ), f"Inconsistent matmat baselinop!{msg}"
                        assert torch.allclose(
                            mat, mat2
                        ), f"Inconsistent batched baselinop!{msg}"
                        assert torch.allclose(
                            mat, mat3
                        ), f"Inconsistent matvec baselinop!{msg}"
                        #
                        # now test transposition
                        for lop, msg in (
                            (lop1, " matmat"),
                            (lop2, " batched"),
                            (lop3, " matvec"),
                        ):
                            lopT = lop.t()
                            lopTT = lopT.t()
                            matT = linop_to_matrix(
                                lopT, dtype=dtype, device=device, adjoint=adj
                            )
                            assert (matT == mat.H).all(), f"Wrong transp?{msg}"
                            assert lopTT is lop, f"Wrong double transp?{msg}"

                        # breakpoint()

                        # phi = gaussian_noise(
                        #     (2, h) if adj else (w, 2),
                        #     dtype=dtype,
                        #     device=device,
                        #     seed=2 * seed + 1,
                        # )

                        # # THIS IS THE BY ROW STUFF
                        # for by_row in (True, False):
                        #     lop = MatrixAsLinOp(mat, by_row=by_row)
                        #     mat2 = linop_to_matrix(
                        #         lop, dtype=dtype, device=device, adjoint=adj
                        #     )
                        #     assert (
                        #         mat == mat2
                        #     ).all(), f"Wrong linop_to_matrix! {adj, by_row}"
                        #     # matmat operations
                        #     matmeas = phi @ mat if adj else mat @ phi
                        #     lopmeas = phi @ lop if adj else lop @ phi
                        #     breakpoint()
                        #     assert torch.allclose(
                        #         matmeas, lopmeas, atol=tol
                        #     ), "lop@v does not equal mat@v in mat-mat!"
                        #     # matvec operations
                        #     matmeas = phi[0] @ mat if adj else mat @ phi[:, 0]
                        #     lopmeas = phi[0] @ lop if adj else lop @ phi[:, 0]
                        #     assert torch.allclose(
                        #         matmeas, lopmeas, atol=tol
                        #     ), "lop@v does not equal mat@v in mat-vec!"
                        #     # now test transposition
                        #     lopT = lop.t()
                        #     lopTT = lopT.t()
                        #     matT = linop_to_matrix(
                        #         lopT, dtype=dtype, device=device, adjoint=adj
                        #     )
                        #     assert (matT == mat.H).all(), "Wrong transp?"
                        #     assert lopTT is lop, "Wrong double transp?"
