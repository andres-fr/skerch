#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`skerch.hdf5`."""


import os
import tempfile
import pytest
import torch
import numpy as np

from skerch.utils import torch_dtype_as_str, gaussian_noise
from skerch.hdf5 import DistributedHDF5
from . import rng_seeds, torch_devices, max_mp_workers


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
def shapes_too_many_files(request):
    """Provides virtual datasets with too many files to cause errors."""
    result = [((5, 1), 20000)]
    if request.config.getoption("--skip_toomanyfiles"):
        result = []
    return result


# ##############################################################################
# # DISTRIBUTED HDF5
# ##############################################################################
def test_hdf5_merge():
    """
    Test here:

    create the layout, populate it and merge while checking flag

    test behaviour with and without deleting subfiles
    """
    # create test array
    num_files, shape, seed, dtype = 10, (3, 4, 5), 12345, torch.float32
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
            flag[0] = f"good_{i}"
            h5.close()
        # merge layout into monolithic file
        breakpoint()
        DistributedHDF5.merge_all(
            h5_path,
            delete_subfiles_while_merging=in_place,
        )

        all_data, all_flags, all_h5 = DistributedHDF5.load_virtual(h5_path)
        for i in range(num_files):
            vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
            breakpoint()


def test_hdf5_io(  # noqa: C901  # ignore "is too complex"
    rng_seeds, dtypes_tols, shapes_numfiles
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
            for shape, num_files in shapes_numfiles:
                # target tensor
                arr_base = gaussian_noise(
                    (num_files, *shape), seed=seed, dtype=dtype, device="cpu"
                ).numpy()
                for filedim_last in (True, False):
                    arr = (
                        np.moveaxis(arr_base, 0, -1)
                        if filedim_last
                        else arr_base
                    )
                    h5shape = tuple(
                        np.roll(arr.shape, -1) if filedim_last else arr.shape
                    )
                    # test that virtual+partial HDF5 files are created in disk
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
                    # simulate distributed writing values to partial H5 files
                    # then test that load_virtual holds correct values
                    for i in range(num_files):
                        vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
                        vals[:] = arr_base[i]
                        flag[0] = f"good_{i}"
                        h5.close()
                    # LOAD_VIRTUAL TEST
                    # test that separate files and merger are correct in
                    # content+shape, and that flags were all set to correct
                    all_vals, all_flags, all_h5 = DistributedHDF5.load_virtual(
                        h5_path
                    )
                    assert (
                        all_vals.shape == arr.shape
                    ), "Wrong merged dataset shape!"
                    assert np.allclose(
                        arr, all_vals, atol=tol
                    ), "Wrong merged dataset values!"
                    assert all_flags.shape == (
                        num_files,
                    ), "Wrong merged flags shape!"
                    for i, f in enumerate(all_flags):
                        assert f.decode() == f"good_{i}", "Unsuccessful flag?"
                    # LOAD TEST
                    # now load again (it was closed), but this time one by
                    # one via load, and also check content and shape
                    for i in range(num_files):
                        vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
                        assert vals.shape == shape, "Wrong partial shape!"
                        assert flag[0].decode() == f"good_{i}", "Wrong flag?"
                        #
                        if filedim_last:
                            sub_target, sub_all = arr[..., i], all_vals[..., i]
                        else:
                            sub_target, sub_all = arr[i], all_vals[i]
                        # breakpoint()
                        assert np.allclose(
                            vals, sub_target, atol=tol
                        ), "Partial inconsistent with target!"
                        assert np.allclose(
                            vals, sub_all, atol=tol
                        ), "Partial inconsistent with merged dataset!"
                        h5.close()
                    #
                    all_h5.close()
                    tmpdir.cleanup()


def test_hdf5_too_many_files(shapes_too_many_files):
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
