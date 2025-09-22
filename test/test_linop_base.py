#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for core linop functionality."""


from time import time
import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    BaseLinOp,
    ByBlockLinOp,
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
        (10, 20),
        (20, 10),
    ]
    if request.config.getoption("--quick"):
        result = result[:7]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
class MatLinOp(BaseLinOp):
    """Implementing matmul."""

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


class MatBBLinOp(ByBlockLinOp):
    """By-Vector linop."""

    def __init__(self, mat, by_row=False, batch=None, blocksize=1):
        """ """
        super().__init__(mat.shape, by_row, batch, blocksize)
        self.mat = mat

    def get_block(self, block_idx, input_dtype, input_device):
        """ """
        idxs = self.get_vector_idxs(block_idx)
        if self.by_row:
            result = self.mat[idxs, :].to(input_device)
        else:
            result = self.mat[:, idxs].to(input_device)
        if len(result.shape) == 1:
            result.unsqueeze_(0 if self.by_row else 1)
        return result


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
    # testing (batched) matmul
    h, w, n, b = 100, 100, 5000, 11
    slow_factor = 5
    atol = 1e-5
    mat = gaussian_noise((h, w), dtype=torch.float32, device="cpu", seed=12345)
    v1, v2 = torch.ones(w), torch.ones(h)
    a1, a2 = torch.ones(w, n), torch.ones(n, h)
    lop1 = MatLinOp(mat, batch=None)
    lop2 = MatLinOp(mat, batch=b)
    lop3 = MatLinOp(mat, batch=1)
    # #     * Runtime: matvec > batched > matmat by a substantial amount
    # # matmat is fastest, matvec slowest, batched inbetween
    # t0 = time()
    # lop1 @ a1  # matmat
    # t1 = time()
    # lop2 @ a1  # batched
    # t2 = time()
    # lop3 @ a1  # matvec
    # t3 = time()
    # assert (slow_factor * (t1 - t0)) < (t2 - t1), "Matmat slower than batched?"
    # assert (slow_factor * (t2 - t1)) < (t3 - t2), "Batched slower than matvec?"
    # t0 = time()
    # a2 @ lop1  # matmat
    # t1 = time()
    # a2 @ lop2  # batched
    # t2 = time()
    # a2 @ lop3  # matvec
    # t3 = time()
    # assert (slow_factor * (t1 - t0)) < (
    #     t2 - t1
    # ), "Matmat slower than batched? (adj)"
    # assert (slow_factor * (t2 - t1)) < (
    #     t3 - t2
    # ), "Batched slower than matvec? (adj)"


def test_byblock_formal(torch_devices, dtypes_tols):
    """Formal correctness test for by-block linops.

    Creates small-shaped by-block linops and checks that:

    * Repr creates correct strings
    * Negative blocksizes trigger error
    * out-of-bounds block idxs trigger error in get_vector_idxs
    * out-of-bounds vector idxs trigger error in get_idx_coords
    """
    shape = (5, 7)
    # repr
    s = str(ByBlockLinOp(shape, by_row=True, batch=10, blocksize=2))
    assert (
        s == "<ByBlockLinOp(5x7), by row, batch=10, blocksize=2>"
    ), "Unexpected repr!"
    #
    for by_row in (True, False):
        # negative blocksize triggers error
        with pytest.raises(ValueError):
            _ = ByBlockLinOp(shape, by_row=by_row, blocksize=-1)
        #
        for bsize in range(1, max(shape)):
            lop = ByBlockLinOp(shape, by_row=by_row, blocksize=bsize)
            # OOB block_idx triggers error
            assert lop.num_blocks >= 1, "Empty num_blocks?"
            with pytest.raises(ValueError):
                _ = lop.get_vector_idxs(-1)
            with pytest.raises(ValueError):
                _ = lop.get_vector_idxs(lop.num_blocks)
            assert isinstance(
                lop.get_vector_idxs(0), range
            ), "Invalid otuput for block_idx=0"
            assert isinstance(
                lop.get_vector_idxs(lop.num_blocks - 1), range
            ), "Invalid output for block_idx=N-1"
            # OOB vector idx triggers error
            assert lop.num_vecs >= 1, "Empty num_vecs?"
            with pytest.raises(ValueError):
                _ = lop.get_idx_coords(-1)
            with pytest.raises(ValueError):
                _ = lop.get_idx_coords(lop.num_vecs)
            assert isinstance(
                lop.get_idx_coords(0), tuple
            ), "Invalid output for vec_idx=0"
            assert isinstance(
                lop.get_idx_coords(lop.num_vecs - 1), tuple
            ), "Invalid output for vec_idx=N-1"


def test_byblock_correctness_extra(rng_seeds, torch_devices, dtypes_tols):
    """Extra correctness tests for by-block linop:

    Creates various small-shaped random by-block linops and, for all
    block sizes and by row/column, tests that:

    * to_matrix retieves correct matrix
    * get_idx coords and get_vector_idxs are consistent, lead to generation
      of correct blocks and point to correct idxs.
    """
    h, w = 5, 7
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                mat = gaussian_noise(
                    (h, w), dtype=dtype, device=device, seed=seed
                )
                # for all blocksizes and by row/col
                for by_row in (True, False):
                    for bsize in range(1, max(h, w)):
                        lop = MatBBLinOp(
                            mat, by_row=by_row, batch=None, blocksize=bsize
                        )
                        # get_idx_coords and get_vector_idxs retrieve
                        # consistent and right coordinates
                        for idx in range(h if by_row else w):
                            b, v = lop.get_idx_coords(idx)
                            b_idxs = lop.get_vector_idxs(b)
                            block = lop.get_block(b, dtype, device)
                            if by_row:
                                assert (mat[idx] == block[v]).all()
                            else:
                                assert (mat[:, idx] == block[:, v]).all()
                        # to_matrix is correct
                        mat2 = lop.to_matrix(dtype, device)
                        assert (
                            mat == mat2
                        ).all(), "Incorrect BB to_matrix for blocksize {block}"


def test_linop_correctness(
    rng_seeds, torch_devices, dtypes_tols, linop_correctness_shapes
):
    """Test case for correctness of base linop and by-block linop.

    For all devices and datatypes, samples Gaussian noise and checks that:
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
                    lop1 = MatLinOp(mat, batch=None)
                    lop2 = MatLinOp(mat, batch=3)
                    lop3 = MatLinOp(mat, batch=1)
                    Ileft = torch.eye(h, dtype=dtype, device=device)
                    Iright = torch.eye(w, dtype=dtype, device=device)
                    for adj in (True, False):
                        msg = " (adj)" if adj else ""
                        mat1 = Ileft @ lop1 if adj else lop1 @ Iright
                        mat2 = Ileft @ lop2 if adj else lop2 @ Iright
                        mat3 = Ileft @ lop3 if adj else lop3 @ Iright
                        assert torch.allclose(
                            mat, mat1
                        ), f"Inconsistent matmat baselinop!{msg}"
                        assert torch.allclose(
                            mat, mat2
                        ), f"Inconsistent batched baselinop!{msg}"
                        assert torch.allclose(
                            mat, mat3
                        ), f"Inconsistent matvec baselinop!{msg}"
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
                        # by-vector correctness tests:
                        for by_row in (True, False):
                            msg = f"adj={adj}, by_row={by_row}"
                            bb1 = MatBBLinOp(
                                mat, by_row=by_row, batch=None, blocksize=1
                            )
                            bb2 = MatBBLinOp(
                                mat, by_row=by_row, batch=None, blocksize=3
                            )
                            bb3 = MatBBLinOp(
                                mat, by_row=by_row, batch=3, blocksize=1
                            )
                            bb4 = MatBBLinOp(
                                mat, by_row=by_row, batch=3, blocksize=3
                            )
                            bb1mat = linop_to_matrix(
                                bb1, dtype=dtype, device=device, adjoint=adj
                            )
                            bb2mat = linop_to_matrix(
                                bb2, dtype=dtype, device=device, adjoint=adj
                            )
                            bb3mat = linop_to_matrix(
                                bb3, dtype=dtype, device=device, adjoint=adj
                            )
                            bb4mat = linop_to_matrix(
                                bb4, dtype=dtype, device=device, adjoint=adj
                            )
                            # correctness via linop_to_matrix
                            assert torch.allclose(
                                mat, bb1mat
                            ), f"Inconsistent BB1/matrix! {msg}"
                            assert torch.allclose(
                                mat, bb2mat
                            ), f"Inconsistent BB2/matrix! {msg}"
                            assert torch.allclose(
                                mat, bb3mat
                            ), f"Inconsistent BB3/matrix! {msg}"
                            assert torch.allclose(
                                mat, bb4mat
                            ), f"Inconsistent BB4/matrix! {msg}"
                            # transposed
                            bbT = bb1.t()
                            bbTT = bbT.t()
                            bbTmat = linop_to_matrix(
                                bbT, dtype=dtype, device=device, adjoint=adj
                            )
                            assert (
                                bbTmat == mat.H
                            ).all(), f"Wrong by-block transposition?{msg}"
                            assert (
                                bbTT is bb1
                            ), f"Wrong by-block double transp?{msg}"
