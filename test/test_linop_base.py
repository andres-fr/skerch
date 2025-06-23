#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for ``NoiseLinOp``.

TODO:
test all linops, add subtratction func to sumlinop
What about providing a (Hermitian) transpose wrapper?

we don't necessarily need to support matrices, as long as we support
parallelization as per below.
The point is that we don't wanna impose much restriction on lop,
since this burden can be on our noisy measurement linops

So each meas linop that WE implement must support the idea
that we provide it to perform_measurement, and it can be parallelized

Probably by providing a row or column?

also implement the complex version of Rad, fourier measurements?
And the real-valued (nonzero version of lord Omega)



LINOP BASE:
* broadcasts matrices into vectors to feed to matmul
* methods to obtain row and column
* regular and hermitian transpose state: A.H @ x = (x.H @ A).H


THE DREAM: we don't have to tiptoe around linops, and can cheaply manip them

If a user wants to provide a linop for measurements, all they need to do is
implement @ and shape.

The composite linops and the transposition, row/col only depend on that so we can feed it in the pipeline
so linops.py is user-compat

Then, if we want to implement a measurement noise, we lowkey dont need much new:
* SSRFT is implemented in both directions, but we need to do it for complex too.
  - consider getting rid of cosine transform?
  - then, there is nothing preventing the linop from providing rows or columns,
    and from being transposed: IN REALITY, WE WANT TO IMPLEMENT IT LIKE THE
    GAUSSIAN AND RADEMACHER!! IT SHOULD WORK





def perform_measurement(lop, meas_lop, parallel_mode=None):
    """ """
    if parallel_mode is None:
        print("WARNING: speedup can be gained. see docs")
    pass


TTODO:

1. test baselinop functionality and random somehow
2. test that transposed works for matmat and also complex
3. test .t() method

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


# @pytest.fixture
# def shapes(request):
#     """Shapes for correctness of SumLinOp."""
#     result = [
#         (1, 1),
#         (1, 2),
#         (2, 1),
#         (5, 5),
#         (7, 19),
#         (19, 7),
#         (20, 20),
#         (10, 51),
#         (51, 10),
#     ]
#     if request.config.getoption("--quick"):
#         result = result[:6]
#     return result


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
# # BASE LINOP TESTS
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


# def test_noiselinop_reproducibility(rng_seeds, torch_devices, dtypes, shapes):
#     """Test case for random reproducibility of ``NoisyLinOp``.

#     Tests that:
#     * Running the same instance twice yields same results
#     * Running two instances of same seed yields same results
#     * Running two instances of different seed yields same results
#     * Left and right matmul result are consistent (i.e. lead to same matrix)
#     """
#     for seed in rng_seeds:
#         for dtype in dtypes:
#             for shape in shapes:
#                 for device in torch_devices:
#                     for part in ("longer", "shorter", "row", "column"):
#                         # create 3 noisy linear operators
#                         lop = GaussianIidLinOp(shape, seed, partition=part)
#                         lop_same = GaussianIidLinOp(
#                             shape, seed, partition=part
#                         )
#                         lop_diff = GaussianIidLinOp(
#                             shape, seed + 1, partition=part
#                         )
#                         # convert them to matrices, from left and right
#                         A = linop_to_matrix(lop, dtype, device, adjoint=False)
#                         A_twice = linop_to_matrix(
#                             lop, dtype, device, adjoint=False
#                         )
#                         A_same = linop_to_matrix(
#                             lop_same, dtype, device, adjoint=False
#                         )
#                         A_diff = linop_to_matrix(
#                             lop_diff, dtype, device, adjoint=False
#                         )
#                         B = linop_to_matrix(lop, dtype, device, adjoint=True)
#                         B_twice = linop_to_matrix(
#                             lop, dtype, device, adjoint=True
#                         )
#                         B_same = linop_to_matrix(
#                             lop_same, dtype, device, adjoint=True
#                         )
#                         B_diff = linop_to_matrix(
#                             lop_diff, dtype, device, adjoint=True
#                         )
#                         # check that running matmul twice yields same results
#                         assert (
#                             A == A_twice
#                         ).all(), "Inconsistent forward matmul in same instance"
#                         assert (
#                             B == B_twice
#                         ).all(), "Inconsistent adjoint matmul in same instance"
#                         # check that 2 objects of same seed yield same results
#                         assert (
#                             A == A_same
#                         ).all(), "Inconsistent forward matmul for same seed!"
#                         assert (
#                             B == B_same
#                         ).all(), "Inconsistent adjoint matmul for same seed!"
#                         # check that 2 objects of different seed are different
#                         assert not torch.allclose(
#                             A, A_diff
#                         ), "Different seed -> similar noisy linops? (fwd.)"
#                         assert not torch.allclose(
#                             B, B_diff
#                         ), "Different seed -> similar noisy linops? (adj.)"
#                         #
#                         # finally check that forward and adjoint are same
#                         assert (
#                             A == B
#                         ).all(), "Forward and adjoint matmul differ!"


# def test_noiselinop_partition(shapes):
#     """Test case for different partitions in noisy linop.

#     Creates a ramped linop, as well as ramped matrices of same shape, and tests
#     that:

#     * Providing an unknown partition name raises a valueError
#     * Partition by row generates Row-ramped matrices, both in fwd and adjoint
#     * Partition by col generates Col-ramped matrices, both in fwd and adjoint
#     * Partition by longer generates matching matrices, both in fwd and adjoint
#     * Partition by shorter generates matching matrices, both in fwd and adjoint
#     """
#     with pytest.raises(ValueError):
#         lop = RampedIidLinOp((10, 10), partition="made up partition XXX")
#     for shape in shapes:
#         # create test matrices, by row and by column
#         Col = torch.outer(torch.arange(shape[0]) + 1, torch.ones(shape[1]))
#         Row = torch.outer(torch.ones(shape[0]), torch.arange(shape[1]) + 1)
#         # partition by column
#         lop = RampedIidLinOp(shape, partition="column")
#         A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
#         B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
#         assert (A == Col).all(), "Wrong forward matmul by [col]"
#         assert (B == Col).all(), "Wrong adjoint matmul by [col]"
#         # partition by row
#         lop = RampedIidLinOp(shape, partition="row")
#         A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
#         B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
#         assert (A == Row).all(), "Wrong forward matmul by [row]"
#         assert (B == Row).all(), "Wrong adjoint matmul by [row]"
#         # partition by longer
#         lop = RampedIidLinOp(shape, partition="longer")
#         A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
#         B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
#         Test = Col if (shape[0] >= shape[1]) else Row
#         assert (A == Test).all(), "Wrong forward matmul by [longer]"
#         assert (B == Test).all(), "Wrong adjoint matmul by [longer]"
#         # partition by shorter
#         lop = RampedIidLinOp(shape, partition="shorter")
#         A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
#         B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
#         Test = Row if (shape[0] >= shape[1]) else Col
#         assert (A == Test).all(), "Wrong forward matmul by [longer]"
#         assert (B == Test).all(), "Wrong adjoint matmul by [longer]"
