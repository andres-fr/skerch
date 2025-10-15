#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`skerch.hdf5`."""


import itertools
import os
import tempfile

import numpy as np
import pytest
import torch

from skerch.hdf5 import DistributedHDF5Tensor, create_hdf5_layout_lop
from skerch.utils import gaussian_noise, torch_dtype_as_str

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
                    flgs[:] = "good"
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
                    assert f.decode() == "good", "Unsuccessful flag?"
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
            flgs[:] = "good"
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
    shape, partsize = (11, 7, 5), 3
    seed, dtype = 12345, torch.float32
    flag_str = "success"
    tnsr = gaussian_noise(shape, seed=seed, dtype=dtype, device="cpu").numpy()
    for in_place in (True, False):
        # create HDF5 layout
        tmpdir = tempfile.TemporaryDirectory()
        path_fmt = os.path.join(tmpdir.name, "h5dataset_{}.h5")
        h5_path, h5_subpaths, begs_ends = DistributedHDF5Tensor.create(
            path_fmt,
            shape,
            partsize,
            torch_dtype_as_str(dtype),
        )
        # write data to layout
        for i, (beg, end) in enumerate(begs_ends):
            vals, flgs, h5 = DistributedHDF5Tensor.load(h5_subpaths[i])
            vals[:] = tnsr[beg:end]
            flgs[:] = flag_str
            h5.close()
        # load ALL layout, check that it contains virtual data
        all_data, all_flags, all_h5 = DistributedHDF5Tensor.load(h5_path)
        for ds in all_h5.values():
            assert ds.is_virtual, "HDF5 layout not virtual?"
        all_h5.close()
        # merge layout into monolithic file
        DistributedHDF5Tensor.merge(
            h5_path,
            check_success_flag=flag_str,
            delete_subfiles_while_merging=in_place,
        )
        # load file again. Now it is non-virtual, contents are correct
        all_data, all_flags, all_h5 = DistributedHDF5Tensor.load(h5_path)
        assert all(
            flag_str == s.decode() for s in all_flags
        ), "Unsuccessful flags in merged array!"
        assert np.array_equal(all_data, tnsr), "Wrong merged array!"
        for ds in all_h5.values():
            assert not ds.is_virtual, f"HDF5 still virtual? {ds}"
        # check that partial data was/wasn't deleted
        num_tmp = len(os.listdir(tmpdir.name))
        expected = 1 if in_place else len(h5_subpaths) + 1
        assert num_tmp == expected, "Wrong number of files after merging?"
        all_h5.close()
        tmpdir.cleanup()


def test_hdf5_create_layout_lop():
    """Test case for ``create_hdf5_layout_lop`` (formal and correctness).

    Creates a DistributedHDF5 layout and checks that requested files are
    created with the requested specs.
    """
    lo_fmt = "leftouter_{}.h5"
    ro_fmt = "rightouter_{}.h5"
    inner_fmt = "inner_{}.h5"
    shape, partsize = (100, 100), 11
    dtype = torch.float32
    strtype = torch_dtype_as_str(dtype)
    outermeas, innermeas = 30, 60
    mat = torch.ones(shape, dtype=dtype, device="cpu")
    #
    div, mod = divmod(outermeas, partsize)
    num_outer_blocks = div + (mod != 0)
    div, mod = divmod(innermeas, partsize)
    num_inner_blocks = div + (mod != 0)
    #
    for lo, ro, inner in itertools.product((False, True), repeat=3):
        tmpdir = tempfile.TemporaryDirectory()
        (
            (lo_pth, lo_subpaths, lo_begs_ends),
            (ro_pth, ro_subpaths, ro_begs_ends),
            (in_pth, in_subpaths, in_begs_ends),
        ) = create_hdf5_layout_lop(
            tmpdir.name,
            mat.shape,
            torch_dtype_as_str(mat.dtype),
            partsize,
            lo_meas=outermeas if lo else None,
            ro_meas=outermeas if ro else None,
            inner_meas=innermeas if inner else None,
            lo_fmt=lo_fmt,
            ro_fmt=ro_fmt,
            inner_fmt=inner_fmt,
        )
        # left outer
        if not lo:
            assert lo_pth is None, "path created for lo=None?"
            assert lo_subpaths is None, "subpaths created for lo=None?"
            assert lo_begs_ends is None, "begs_ends created for lo=None?"
        else:
            data, flags, h5 = DistributedHDF5Tensor.load(lo_pth)
            assert data.dtype == strtype, "Wrong HDF5 lo dtype?"
            assert data.shape == (outermeas, shape[0]), "Wrong HDF5 lo shape?"
            num = len([x for x in os.listdir(tmpdir.name) if "leftouter" in x])
            assert num == (
                num_outer_blocks + 1
            ), "Wrong number of lo HDF5 subfiles created?"
            h5.close()
        # right outer
        if not ro:
            assert ro_pth is None, "path created for ro=None?"
            assert ro_subpaths is None, "subpaths created for ro=None?"
            assert ro_begs_ends is None, "begs_ends created for ro=None?"
        else:
            data, flags, h5 = DistributedHDF5Tensor.load(ro_pth)
            assert data.dtype == strtype, "Wrong HDF5 ro dtype?"
            assert data.shape == (outermeas, shape[0]), "Wrong HDF5 ro shape?"
            num = len(
                [x for x in os.listdir(tmpdir.name) if "rightouter" in x]
            )
            assert num == (
                num_outer_blocks + 1
            ), "Wrong number of ro HDF5 subfiles created?"
            h5.close()
        # inner
        if not inner:
            assert in_pth is None, "path created for inner=None?"
            assert in_subpaths is None, "subpaths created for inner=None?"
            assert in_begs_ends is None, "begs_ends created for inner=None?"
        else:
            data, flags, h5 = DistributedHDF5Tensor.load(in_pth)
            assert data.dtype == strtype, "Wrong HDF5 inner dtype?"
            assert data.shape == (
                innermeas,
                innermeas,
            ), "Wrong HDF5 inner shape?"
            num = len([x for x in os.listdir(tmpdir.name) if "inner" in x])
            assert num == (
                num_inner_blocks + 1
            ), "Wrong number of inner HDF5 subfiles created?"
            h5.close()
        tmpdir.cleanup()
