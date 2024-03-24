#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for distributed measurement utils."""


import os
import tempfile

import numpy as np
import pytest
import torch

from skerch.distributed_measurements import DistributedHDF5, ssrft_idx_torch
from skerch.ssrft import SSRFT
from skerch.utils import gaussian_noise, torch_dtype_as_str

from . import rng_seeds


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def f64_atol():
    """Absolute error tolerance for float64."""
    result = {torch.float64: 1e-30}
    return result


@pytest.fixture
def f32_atol():
    """Absolute error tolerance for float32."""
    result = {torch.float32: 1e-5}
    return result


@pytest.fixture
def shapes_numfiles():
    """Shapes of individual HDF5 files and how many individual files."""
    result = [
        ((1,), 1),
        ((1,), 10),
        ((10,), 100),
        ((1000,), 100),
        #
        ((1, 2), 1),
        ((1, 2), 10),
        ((10, 2), 100),
        ((1000, 2), 100),
    ]
    return result


@pytest.fixture
def shapes_too_many_files(request):
    """Provides virtual datasets with too many files to cause errors."""
    result = [((5, 1), 20000)]
    if request.config.getoption("--skip_toomanyfiles"):
        result = []
    return result


@pytest.fixture
def shapes_measurements():
    """Shape of linear operator and SSRFT height.

    Since SSRFT can only be fat, SSRFT height must always be <= than
    the smallest dimension in shape.
    """
    result = [
        ((1, 1), 1),
        ((1, 10), 1),
        ((10, 1), 1),
        ((10, 10), 1),
        ((10, 10), 5),
        ((10, 10), 10),
        ((10, 100), 10),
        ((100, 100), 50),
        ((100, 1000), 50),
        ((1000, 1000), 500),
    ]
    return result


# ##############################################################################
# # DISTRIBUTED MEASUREMENTS
# ##############################################################################
def test_ssrft_idx_torch(rng_seeds, f64_atol, f32_atol, shapes_measurements):
    """Test case for applying a SSRFT indexed measurement.

    Creates a random matrix, and a SSRFT linop with given shapes. Tests that:

    * Applying SSRFT in indexed fashion is same as using @, both for left- and
      right-matmul
    """
    for seed in rng_seeds:
        ssrft_seed = seed + 1
        for dtype, atol in {**f32_atol, **f64_atol}.items():
            for lop_shape, num_measurements in shapes_measurements:
                lop_h, lop_w = lop_shape
                lop = gaussian_noise(
                    (lop_h, lop_w), seed=seed, dtype=dtype, device="cpu"
                )
                #
                for adjoint in (True, False):
                    ssrft_w = lop_h if adjoint else lop_w
                    ssrft = SSRFT((num_measurements, ssrft_w), seed=ssrft_seed)
                    #
                    target = (ssrft @ lop) if adjoint else (ssrft @ lop.T).T
                    # check that target shape is as expected
                    if adjoint:
                        expected_shape = (num_measurements, lop_w)
                    else:
                        expected_shape = (lop_h, num_measurements)
                    assert target.shape == expected_shape, "Wrong result shape!"
                    #
                    measurements = np.zeros_like(target)
                    flags = ["" for _ in range(num_measurements)]
                    flags = np.empty(num_measurements, dtype="str")
                    for i in range(num_measurements):
                        ssrft_idx_torch(
                            i,
                            num_measurements,
                            lop,
                            lop.device,
                            dtype,
                            measurements[i] if adjoint else measurements[:, i],
                            seed=ssrft_seed,
                            flag=flags[i : i + 1],
                            processing_flag="0",
                            success_flag="1",
                            adjoint=adjoint,
                        )
                    assert np.allclose(
                        measurements, target, atol=atol
                    ), "distributed SSRFT produced wrong result!"
                    assert all(
                        (f == "1" for f in flags)
                    ), "distributed SSRFT produced unsuccessful flags!"


# ##############################################################################
# # DISTRIBUTED HDF5
# ##############################################################################
def test_distributed_files(  # noqa: C901  # ignore "is too complex"
    rng_seeds, f64_atol, f32_atol, shapes_numfiles
):
    """Test case for ``DistributedHDF5`` creation and write/read.

    Creates a temporary DistributedHDF5 with random values, and checks that:
    * Files are actually created
    * Values can be written
    * Shapes of sub-datasets and main dataset are as expected
    * All values between sub-datasets, main dataset and test are consistent
    """
    for seed in rng_seeds:
        for dtype, atol in {**f32_atol, **f64_atol}.items():
            for shape, num_files in shapes_numfiles:
                for filedim_last in (True, False):
                    # target tensor
                    target_vals = torch.empty((num_files, *shape), dtype=dtype)
                    for i in range(num_files):
                        target_vals[i] = gaussian_noise(
                            shape, seed=seed + i, dtype=dtype, device="cpu"
                        )

                    if filedim_last:
                        target_vals = target_vals.moveaxis(0, -1)
                    # check dataset is actually being created on (temp) disk
                    tmpdir = tempfile.TemporaryDirectory()
                    out_path = os.path.join(tmpdir.name, "distdata_{}.h5")
                    h5_path, h5_subpaths = DistributedHDF5.create(
                        out_path,
                        num_files,
                        shape,
                        torch_dtype_as_str(dtype),
                        filedim_last=filedim_last,
                    )
                    assert os.path.isfile(h5_path), "Merged HDF5 not created?"
                    for sp in h5_subpaths:
                        assert os.path.isfile(sp), "Partial HDF5 not created?"
                    # write values to sub-datasets
                    for i in range(num_files):
                        vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
                        vals[:] = gaussian_noise(
                            shape, seed=seed + i, dtype=dtype, device="cpu"
                        )
                        flag[0] = str(i)
                        h5.close()
                    # test that separate files and merger are correct
                    all_vals, all_flags, all_h5 = DistributedHDF5.load_virtual(
                        h5_path
                    )
                    assert (
                        all_vals.shape == target_vals.shape
                    ), "Wrong merged dataset shape!"
                    assert np.allclose(
                        target_vals, all_vals, atol=atol
                    ), "Wrong merged dataset values!"
                    assert all_flags.shape == (
                        num_files,
                    ), "Wrong merged flags shape!"
                    assert all_flags
                    for i in range(num_files):
                        vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
                        assert vals.shape == shape, "Wrong sub-dataset shape!"
                        int_flag = int(flag[0].decode())
                        assert i == int_flag, "Wrong flag value!"
                        #
                        if filedim_last:
                            sub_target = target_vals[..., i]
                            sub_all = all_vals[..., i]
                        else:
                            sub_target = target_vals[i]
                            sub_all = all_vals[i]
                        assert np.allclose(
                            vals, sub_target, atol=atol
                        ), "Sub-dataset inconsistent with target!"
                        assert np.allclose(
                            vals, sub_all, atol=atol
                        ), "Sub-dataset inconsistent with merged dataset!"
                        h5.close()
                    #
                    all_h5.close()
                    tmpdir.cleanup()


def test_too_many_files(shapes_too_many_files):
    """Test case for too large ``DistributedHDF5``: read errors.

    Creates a temporary DistributedHDF5 with random values with a lot of
    files, and checks that, while all other tests still pass, loading the
    full virtual dataset actually contains erroneous (missing) columns.
    """
    seed = 0
    dtype, atol = torch.float32, 1e-5
    filedim_last = False
    for shape, num_files in shapes_too_many_files:
        # target tensor
        target_vals = torch.empty((num_files, *shape), dtype=dtype)
        for i in range(num_files):
            target_vals[i] = gaussian_noise(
                shape, seed=seed + i, dtype=dtype, device="cpu"
            )
        if filedim_last:
            target_vals = target_vals.moveaxis(0, -1)
        # check dataset is actually being created on (temp) disk
        tmpdir = tempfile.TemporaryDirectory()
        out_path = os.path.join(tmpdir.name, "distdata_{}.h5")
        h5_path, h5_subpaths = DistributedHDF5.create(
            out_path,
            num_files,
            shape,
            torch_dtype_as_str(dtype),
            filedim_last=filedim_last,
        )
        assert os.path.isfile(h5_path), "Merged HDF5 not created?"
        for sp in h5_subpaths:
            assert os.path.isfile(sp), "Partial HDF5 not created?"
        # write values to sub-datasets
        for i in range(num_files):
            vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
            vals[:] = gaussian_noise(
                shape, seed=seed + i, dtype=dtype, device="cpu"
            )
            flag[0] = str(i)
            h5.close()
        # test that separate files  are correct
        all_vals, all_flags, all_h5 = DistributedHDF5.load_virtual(h5_path)
        assert (
            all_vals.shape == target_vals.shape
        ), "Wrong merged dataset shape!"
        # but here is the catch, merger is not correct!
        # this is due to file limit being exceeded, overflown values are empty
        with pytest.raises(AssertionError):
            assert np.allclose(
                target_vals, all_vals, atol=atol
            ), "Wrong merged dataset values!"
        # it would follow to test that merged is not consistent with
        # sub-datasets, but this itself causes OSErros, so we skip that.
        all_h5.close()
        tmpdir.cleanup()
