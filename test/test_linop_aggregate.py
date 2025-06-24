#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for aggregate linops (composite, sum,...)


* draw a bunch of random mats and do composites and sums
*
* test-transpose everything, including custom linops

TTODO:
* add sub fn to sumlinop
* test composite and sum correctness
* test that they can be transposed
* test that custom linops can be aggregated (sum and compo, compo of sums...)

TODO:
* Rad, fourier, sketchlord, ssrft measurements (test for seed consistency, complex etc)
  - ssrft should also be complex and work like gaussian and rademacher
* generic lop supported, and our measurement lops supporting parallelism.


def perform_measurement(lop, meas_lop, parallel_mode=None):
    """ """
    if parallel_mode is None:
        print("WARNING: speedup can be gained. see docs")
    pass
"""


import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    ByVectorLinOp,
    TransposedLinOp,
    SumLinOp,
    CompositeLinOp,
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
def composite_sizes(request):
    """Chained shapes for the composite linear operator."""
    result = [
        (1, 1, 1),
        (1, 5, 1),
        (5, 1, 5),
        (10, 10, 10, 10),
        (1, 5, 50, 5, 1),
    ]
    if request.config.getoption("--quick"):
        result = result[:-1]
    return result


@pytest.fixture
def sum_shapes(request):
    """Shapes for correctness of SumLinOp."""
    result = [
        ((1, 1),),
        ((1, 1), (1, 1)),
        ((5, 1), (5, 1), (5, 1)),
        ((1, 5), (1, 5), (1, 5)),
        ((13, 23), (13, 23)),
        ((100, 101), (100, 101)),
    ]
    if request.config.getoption("--quick"):
        result = result[:5]
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


class ScalarLinOp:
    """Basic scalar linop with shape and matmul, but is not a baselinop."""

    def __init__(self, shape, scale=1.0):
        """Initializer. See class docstring."""
        self.shape = shape
        self.scale = scale

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``."""
        return x * self.scale

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``."""
        return x * self.scale

    def __repr__(self):
        """Returns a string in the form <classname(shape)>."""
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})>"
        return s


# ##############################################################################
# # TESTS
# ##############################################################################
def test_aggregate_formal_and_basic_correctness(sum_shapes):
    """Formal/basic correctness test case for aggregate linops.

    When creating aggregate linops, tests that:

    * Providing an empty collection triggers a ValueError
    * Repr creates correct strings
    * Mismatching shapes trigger a BadShapeError
    * (adjoint-) matmul with vector of wrong size triggers BadShapeError

    And also:
    * Linops look as they should when converted to matrix (also when mixing
      matrices and linops)
    """
    # empty input raises error
    with pytest.raises(ValueError):
        _ = SumLinOp([])
    with pytest.raises(ValueError):
        _ = CompositeLinOp([])
    # correct string conversion
    hw = (5, 5)
    m, l1, l2 = (torch.ones(hw), ScalarLinOp(hw, 1.0), ScalarLinOp(hw, 2.0))
    assert (
        str(SumLinOp((("M", 1, m), ("L1", 1, l1), ("L2", 1, l2))))
        == "M + L1 + L2"
    ), "Unexpected repr for sum linop!"
    assert (
        str(SumLinOp((("M", 0, m), ("L1", 0, l1), ("L2", 0, l2))))
        == "-M - L1 - L2"
    ), "Unexpected repr for sum linop!"
    assert (
        str(CompositeLinOp((("M", m), ("L1", l1), ("L2", l2))))
        == "M @ L1 @ L2"
    ), "Unexpected repr for composite linop!"
    # matrices and linops can be mixed and result is correct
    v = m[0]
    slop = SumLinOp((("M", 1, m), ("L1", 1, l1), ("L2", 1, l2)))
    clop = CompositeLinOp((("M", m), ("L1", l1), ("L2", l2)))
    assert (
        linop_to_matrix(slop) == m + torch.eye(len(m)) * 3
    ).all(), "Incorrect sum linop to matrix?"
    assert (
        linop_to_matrix(clop) == m * 2
    ).all(), "Incorrect composite linop to matrix?"
    # shape inconsistencies raise errors
    for shapes in sum_shapes:
        # create list of linops with inconsistent shapes
        linops = [ScalarLinOp(s) for s in shapes]
        linops.append(ScalarLinOp((shapes[0][0] + 1, shapes[0][1])))
        # test that sum linop raises error
        with pytest.raises(BadShapeError):
            _ = SumLinOp((f"L{i}", 1, l) for i, l in enumerate(linops))
        # test that composite linop raises error
        with pytest.raises(BadShapeError):
            _ = CompositeLinOp((f"L{i}", l) for i, l in enumerate(linops))


# def test_aggregate_correctness():
#     """Test

#     Draw a bunch of random linops and also custom AND ALSO REGULAR MATRICES, and test:
#     * the sum is indeed correct
#     * the composition is indeed correct
#     * the transposition is indeed correct
#     * basic
#     * The string of repr matches some expectation

#     """
#     pass


def test_sum_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    sum_shapes,
):
    """Test case for correctness of ``SumLinOp``.

    Creates a set of random matrices and linops. Then sums them explicitly, and
    creates a ``SumLinOp``. Tests that:
    * Matmul and rmatmul with sum linop is same as with matrix
    * Same with (Hermitian) transposed sum linop
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for shapes in sum_shapes:
                    # sample random submatrices
                    submatrices = []
                    for i, shape in enumerate(shapes):
                        submatrices.append(
                            gaussian_noise(
                                shape,
                                seed=seed + i,
                                dtype=dtype,
                                device=device,
                            )
                        )
                    # all-sum
                    mat = sum(submatrices)
                    lop = SumLinOp(
                        (f"M_{i}", 1, m) for i, m in enumerate(submatrices, 1)
                    )
                    v = gaussian_noise(
                        mat.shape[1],
                        dtype=dtype,
                        device=device,
                        seed=seed + 100,
                    )
                    w = gaussian_noise(
                        mat.shape[0],
                        dtype=dtype,
                        device=device,
                        seed=seed + 101,
                    )
                    assert torch.allclose(
                        mat @ v, lop @ v, atol=tol
                    ), "Incorrect forward sum+ linop!"
                    assert torch.allclose(
                        w @ mat, w @ lop, atol=tol
                    ), "Incorrect adjoint sum+ linop!"
                    # alternating sum and diff: M1 - M2 + M3 ...
                    mat = submatrices[0].clone()
                    for i, m in enumerate(submatrices[1:]):
                        # breakpoint()
                        if i % 2 == 0:
                            mat -= m
                        else:
                            mat += m
                    lop = SumLinOp(
                        (f"M_{i}", i % 2, m)
                        for i, m in enumerate(submatrices, 1)
                    )
                    assert torch.allclose(
                        mat @ v, lop @ v, atol=tol
                    ), "Incorrect alternating sum linop!"
                    try:
                        assert torch.allclose(
                            w @ mat, w @ lop, atol=tol
                        ), "Incorrect adjoint alternating sum linop!"
                    except:
                        print(
                            "THIS TEST IS FAILING PROBABLY DUE TO SIGN ORDER"
                        )
                        breakpoint()


def test_composite_correctness(
    rng_seeds,
    torch_devices,
    f64_atol,
    f32_atol,
    composite_sizes,
    num_test_measurements,
):
    """Test case for correctness of ``CompositeLinOp``.

    Creates a chain of random submatrices, and then composes them explicity
    using ``CompositeLinOp``. Tests that:

    * Composed shapes are as expected
    * Explicit and implicit compositions are identical both with left and right
      matrix multiplication
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, atol in {**f32_atol, **f64_atol}.items():
                for sizes in composite_sizes:
                    # sample random submatrices
                    submatrices = []
                    for i, (h, w) in enumerate(zip(sizes[:-1], sizes[1:])):
                        mat = gaussian_noise(
                            (h, w), seed=seed + i, dtype=dtype, device=device
                        )
                        submatrices.append((f"{(h, w)}", mat))
                    # compose submatrices, either explicit or implicit
                    explicit = submatrices[0][1]
                    for _, mat in submatrices[1:]:
                        explicit = explicit @ mat
                    composite = CompositeLinOp(submatrices)
                    # check that shapes are correct
                    expected_shape = (sizes[0], sizes[-1])
                    assert (
                        composite.shape == expected_shape
                    ), "Incorrect composite shape?"
                    assert (
                        explicit.shape == expected_shape
                    ), "Incorrect explicit shape?"
                    # check that explicit and implicit are identical for both
                    # left and right matmul
                    for adjoint in [True, False]:
                        (_, _, err_estimate), _ = a_posteriori_error(
                            explicit,
                            composite,
                            num_test_measurements,
                            seed=seed + len(sizes),
                            dtype=dtype,
                            device=device,
                            adjoint=adjoint,
                        )
                        assert (
                            err_estimate / explicit.numel()
                        ) <= atol, "Incorrect composite operation!"


# ##############################################################################
# ???
# ##############################################################################
def test_linop_correctness(
    rng_seeds, torch_devices, dtypes_tols, linop_correctness_shapes
):
    """Test case for correctness of baselinop-based matmuls.

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


# def test_aggregate_correctness():
#     """Test

#     Draw a bunch of random linops and also custom AND ALSO REGULAR MATRICES, and test:
#     * the sum is indeed correct
#     * the composition is indeed correct
#     * the transposition is indeed correct
#     * basic
#     * The string of repr matches some expectation

#     """
#     pass
