#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`skerch.hdf5`."""


import os
import tempfile
import pytest
import torch
import numpy as np

from skerch.utils import torch_dtype_as_str, gaussian_noise
from skerch.hdf5 import DistributedHDF5Tensor
from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-3,
        torch.complex64: 1e-3,
        torch.float64: 3e-8,
        torch.complex128: 3e-8,
    }
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
def shapes_partsizes():
    """Shapes of individual HDF5 files and how many individual files."""
    result = [
        ((1,), 1),
        ((10,), 10),
        ((10,), 3),
        ((10,), 20),
        ((11, 7, 5), 3),
        ((1000, 100), 100),
    ]
    return result


@pytest.fixture
def shapes_partsizes_toomany(request):
    """Provides virtual datasets with too many files to cause errors."""
    result = [((10_000, 1), 1)]
    if request.config.getoption("--skip_toomanyfiles"):
        result = []
    return result


# ##############################################################################
# # DISTRIBUTED HDF5
# ##############################################################################
def test_hdf5_merge():
    """Test case for ``DistributedHDF5`` merging.

    Creates a DistributedHDF5 layout and fills it with random values. Checks
    that written data is indeed in virtual datasets. Then merges it and checks:
    * Merged flags are correct
    * Merged numerical data is correct
    * Merged datasets are not virtual
    * Virtual files are deleted if in_place, and kept otherwise
    """
    # create test array
    num_files, shape, seed, dtype = 10, (3, 4, 5), 12345, torch.float32
    flag_str = "success"
    arr = gaussian_noise(
        (num_files, *shape), seed=seed, dtype=dtype, device="cpu"
    ).numpy()
    for in_place in (True, False):
        # create HDF5 layout
        tmpdir = tempfile.TemporaryDirectory()
        out_path = os.path.join(tmpdir.name, "distdata_{}.h5")
        h5_path, h5_subpaths = DistributedHDF5.create(
            out_path,
            num_files,
            shape,
            torch_dtype_as_str(dtype),
            filedim_last=False,
        )
        # write array to layout
        for i in range(num_files):
            vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
            vals[:] = arr[i]
            flag[0] = flag_str
            h5.close()
        # load ALL layout, check that it contains virtual data
        all_data, all_flags, all_h5 = DistributedHDF5.load(h5_path)
        for ds in all_h5.values():
            assert ds.is_virtual, f"HDF5 layout not virtual?"
        all_h5.close()
        # merge layout into monolithic file
        DistributedHDF5.merge_all(
            h5_path,
            delete_subfiles_while_merging=in_place,
            check_success_flag=flag_str,
        )
        # load file again. Now it is non-virtual, contents are correct
        all_data, all_flags, all_h5 = DistributedHDF5.load(h5_path)
        assert all(
            [flag_str == s.decode() for s in all_flags[:]]
        ), "Unsuccessful flags in merged array!"
        assert np.array_equal(all_data, arr), "Wrong merged array!"
        for ds in all_h5.values():
            assert not ds.is_virtual, f"HDF5 still virtual? {ds}"
        # check that partial data was/wasn't deleted
        num_tmp = len(os.listdir(tmpdir.name))
        expected = 1 if in_place else num_files + 1
        assert num_tmp == expected, "Wrong number of files after merging?"
        all_h5.close()
        tmpdir.cleanup()


def test_hdf5_io(  # noqa: C901  # ignore "is too complex"
    rng_seeds, dtypes_tols, shapes_partsizes
):
    """Test case for ``DistributedHDF5`` creation and write/read.

    Creates a temporary DistributedHDF5 with random values, and for both
    virtual and individual loading, checks that:
    * Files are actually created
    * Values can be written
    * Shapes of sub-datasets and main dataset are as expected
    * All values between sub-datasets, main dataset and test are consistent
    * Flags are successfully set and retrieved
    """
    for seed in rng_seeds:
        for dtype, tol in dtypes_tols.items():
            for shape, partsize in shapes_partsizes:
                # target tensor
                tnsr = gaussian_noise(
                    shape, seed=seed, dtype=dtype, device="cpu"
                ).numpy()
                # test that virtual+partial HDF5 files are created in disk
                tmpdir = tempfile.TemporaryDirectory()
                path_fmt = os.path.join(tmpdir.name, "h5dataset_{}.h5")
                h5_path, h5_subpaths, begs_ends = DistributedHDF5Tensor.create(
                    path_fmt,
                    shape,
                    partsize,
                    torch_dtype_as_str(dtype),
                )
                assert os.path.isfile(h5_path), "Merged HDF5 not created?"
                for sp in h5_subpaths:
                    assert os.path.isfile(sp), "Partial HDF5 not created?"
                # simulate distributed writing values to partial H5 files
                for i, (beg, end) in enumerate(begs_ends):
                    vals, flgs, h5 = DistributedHDF5Tensor.load(h5_subpaths[i])
                    vals[:] = tnsr[beg:end]
                    flgs[:] = f"good"
                    h5.close()
                # VIRTUAL LOAD TEST
                # test that virtual data and flags are correct in content+shape
                all_vals, all_flags, all_h5 = DistributedHDF5Tensor.load(
                    h5_path
                )
                assert (
                    all_vals.shape == tnsr.shape
                ), "Wrong merged dataset shape!"
                assert np.allclose(
                    all_vals, tnsr, atol=tol
                ), "Wrong merged dataset values!"
                assert all_flags.shape == (
                    shape[0],
                ), "Wrong merged flags shape!"
                for f in all_flags[:]:
                    assert f.decode() == f"good", "Unsuccessful flag?"
                all_h5.close()
                # SUBFILE LOAD TEST
                # now load again (it was closed), but this time one by
                # one via load, and also check content and shape
                for i, (beg, end) in enumerate(begs_ends):
                    vals, flgs, h5 = DistributedHDF5Tensor.load(h5_subpaths[i])
                    sublen = end - beg
                    subshape = (sublen,) + shape[1:]
                    assert vals.shape == subshape, "Wrong partial shape?"
                    for f in flgs[:]:
                        assert f.decode() == "good", "Wrong partial flag?"
                    assert np.allclose(
                        vals[:], tnsr[beg:end], atol=tol
                    ), "Partial data with merged dataset!"
                    h5.close()
                #
                tmpdir.cleanup()


def test_hdf5_too_many_files(shapes_partsizes_toomany):
    """Test case for too large ``DistributedHDF5``: read errors.

    Creates a temporary DistributedHDF5 with random values with a lot of
    files, and checks that, while all other tests still pass, loading the
    full virtual dataset actually contains erroneous (missing) columns.
    """
    seed = 0
    dtype, tol = torch.float32, 1e-5
    for shape, partsize in shapes_partsizes_toomany:
        tnsr = gaussian_noise(shape, seed=seed, dtype=dtype, device="cpu")
        # create layout and check is actually being created on (temp) disk
        tmpdir = tempfile.TemporaryDirectory()
        path_fmt = os.path.join(tmpdir.name, "h5dataset_{}.h5")
        h5_path, h5_subpaths, begs_ends = DistributedHDF5Tensor.create(
            path_fmt,
            shape,
            partsize,
            torch_dtype_as_str(dtype),
        )
        assert os.path.isfile(h5_path), "Merged HDF5 not created?"
        for sp in h5_subpaths:
            assert os.path.isfile(sp), "Partial HDF5 not created?"
        # write values to sub-datasets
        for i, (beg, end) in enumerate(begs_ends):
            vals, flgs, h5 = DistributedHDF5Tensor.load(h5_subpaths[i])
            vals[:] = tnsr[beg:end]
            flgs[:] = f"good"
            h5.close()
        # now, the aggregated dataset is not correct! This is due to OS file
        # limit per process being exceeded, overflown values are empty
        all_vals, all_flags, all_h5 = DistributedHDF5Tensor.load(h5_path)
        assert all_vals.shape == tnsr.shape, "Wrong merged dataset shape!"
        with pytest.raises(AssertionError):
            assert np.allclose(
                all_vals[:], tnsr, atol=tol
            ), "now all_vals is missing entries!"
        # it would follow to test that merged is not consistent with
        # sub-datasets, but this itself causes OSErros, so we skip that.
        all_h5.close()
        tmpdir.cleanup()
