#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for ``NoiseLinOp``."""

import pytest
import torch

from skerch.linops import GaussianIidLinOp, NoiseLinOp
from skerch.utils import BadShapeError, gaussian_noise

from . import linop_to_matrix, rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes():
    """Absolute error tolerance for float64."""
    return [torch.float64, torch.float32]


@pytest.fixture
def shapes(request):
    """Shapes for correctness of SumLinOp."""
    result = [
        (1, 1),
        (1, 2),
        (2, 1),
        (5, 5),
        (7, 19),
        (19, 7),
        (20, 20),
        (10, 51),
        (51, 10),
    ]
    if request.config.getoption("--quick"):
        result = result[:6]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
class RampedIidLinOp(NoiseLinOp):
    """Testing linear operator with ``torch.arange`` entries."""

    def sample(self, dims, idx, device):
        """Produces a vector in the form ``[1, 2, ... dims]``."""
        result = torch.arange(dims) + 1
        return result.to(device)


# ##############################################################################
# # TESTS
# ##############################################################################
def test_noiselinop_reproducibility(rng_seeds, torch_devices, dtypes, shapes):
    """Test case for random reproducibility of ``NoisyLinOp``.

    Tests that:
    * Running the same instance twice yields same results
    * Running two instances of same seed yields same results
    * Running two instances of different seed yields same results
    * Left and right matmul result are consistent (i.e. lead to same matrix)
    """
    for seed in rng_seeds:
        for dtype in dtypes:
            for shape in shapes:
                for device in torch_devices:
                    for part in ("longer", "shorter", "row", "column"):
                        # create 3 noisy linear operators
                        lop = GaussianIidLinOp(shape, seed, partition=part)
                        lop_same = GaussianIidLinOp(shape, seed, partition=part)
                        lop_diff = GaussianIidLinOp(
                            shape, seed + 1, partition=part
                        )
                        # convert them to matrices, from left and right
                        A = linop_to_matrix(lop, dtype, device, adjoint=False)
                        A_twice = linop_to_matrix(
                            lop, dtype, device, adjoint=False
                        )
                        A_same = linop_to_matrix(
                            lop_same, dtype, device, adjoint=False
                        )
                        A_diff = linop_to_matrix(
                            lop_diff, dtype, device, adjoint=False
                        )
                        B = linop_to_matrix(lop, dtype, device, adjoint=True)
                        B_twice = linop_to_matrix(
                            lop, dtype, device, adjoint=True
                        )
                        B_same = linop_to_matrix(
                            lop_same, dtype, device, adjoint=True
                        )
                        B_diff = linop_to_matrix(
                            lop_diff, dtype, device, adjoint=True
                        )
                        # check that running matmul twice yields same results
                        assert (
                            A == A_twice
                        ).all(), "Inconsistent forward matmul in same instance"
                        assert (
                            B == B_twice
                        ).all(), "Inconsistent adjoint matmul in same instance"
                        # check that 2 objects of same seed yield same results
                        assert (
                            A == A_same
                        ).all(), "Inconsistent forward matmul for same seed!"
                        assert (
                            B == B_same
                        ).all(), "Inconsistent adjoint matmul for same seed!"
                        # check that 2 objects of different seed are different
                        assert not torch.allclose(
                            A, A_diff
                        ), "Different seed -> similar noisy linops? (fwd.)"
                        assert not torch.allclose(
                            B, B_diff
                        ), "Different seed -> similar noisy linops? (adj.)"
                        #
                        # finally check that forward and adjoint are same
                        assert (
                            A == B
                        ).all(), "Forward and adjoint matmul differ!"


def test_input_shape_mismatch():
    """Test case for input shape consistency.

    Tests that providing vectors of mismatching shape raises BadShapeError,
    both in forward and adjoint matmul.
    """
    lop = GaussianIidLinOp((10, 20), seed=12345, partition="longer")
    v1, v2 = torch.empty(lop.shape[0] + 1), torch.empty(lop.shape[1] + 1)
    with pytest.raises(BadShapeError):
        _ = v1 @ lop
    with pytest.raises(BadShapeError):
        _ = lop @ v2


def test_noiselinop_partition(shapes):
    """Test case for different partitions in noisy linop.

    Creates a ramped linop, as well as ramped matrices of same shape, and tests
    that:

    * Providing an unknown partition name raises a valueError
    * Partition by row generates Row-ramped matrices, both in fwd and adjoint
    * Partition by col generates Col-ramped matrices, both in fwd and adjoint
    * Partition by longer generates matching matrices, both in fwd and adjoint
    * Partition by shorter generates matching matrices, both in fwd and adjoint
    """
    with pytest.raises(ValueError):
        lop = RampedIidLinOp((10, 10), partition="made up partition XXX")
    for shape in shapes:
        # create test matrices, by row and by column
        Col = torch.outer(torch.arange(shape[0]) + 1, torch.ones(shape[1]))
        Row = torch.outer(torch.ones(shape[0]), torch.arange(shape[1]) + 1)
        # partition by column
        lop = RampedIidLinOp(shape, partition="column")
        A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
        B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
        assert (A == Col).all(), "Wrong forward matmul by [col]"
        assert (B == Col).all(), "Wrong adjoint matmul by [col]"
        # partition by row
        lop = RampedIidLinOp(shape, partition="row")
        A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
        B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
        assert (A == Row).all(), "Wrong forward matmul by [row]"
        assert (B == Row).all(), "Wrong adjoint matmul by [row]"
        # partition by longer
        lop = RampedIidLinOp(shape, partition="longer")
        A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
        B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
        Test = Col if (shape[0] >= shape[1]) else Row
        assert (A == Test).all(), "Wrong forward matmul by [longer]"
        assert (B == Test).all(), "Wrong adjoint matmul by [longer]"
        # partition by shorter
        lop = RampedIidLinOp(shape, partition="shorter")
        A = linop_to_matrix(lop, torch.float32, "cpu", adjoint=False)
        B = linop_to_matrix(lop, torch.float32, "cpu", adjoint=True)
        Test = Row if (shape[0] >= shape[1]) else Col
        assert (A == Test).all(), "Wrong forward matmul by [longer]"
        assert (B == Test).all(), "Wrong adjoint matmul by [longer]"
