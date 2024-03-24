#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytests distributed sketched SVD utils."""


import torch

from skerch.distributed_decompositions import (
    DistributedHDF5,
    create_hdf5_layout,
    orthogonalize,
    solve_core_ssvd,
)
from skerch.distributed_measurements import innermeas_idx_torch, ssrft_idx_torch
from skerch.ssrft import SSRFT


# ##############################################################################
# # MOCK DISTRIBUTED DECOMPOSITIONS
# ##############################################################################
def mock_dssvd(dirname, lop, lop_device, lop_dtype, outer, inner, seed):
    """Mock implementation of a distributed sketched SVD.

    This function emulates, in a single process, the steps that would be done
    in a distributed system to compute a sketched SVD. Usage example::

      tmpdir = tempfile.TemporaryDirectory()
      mock_dssvd(
          tmpdir.name, mat, device, dtype, outer, inner, seed
      )
      tmpdir.cleanup()
    """
    processing, success = "0", "1"  # arbitrary flags
    # convenience params
    h, w = lop.shape
    lo_seed, ro_seed, li_seed, ri_seed = seed, seed + 1, seed + 2, seed + 3
    # create HDF5 layout (centralized)
    (
        (lo_path, lo_subpaths),
        (ro_path, ro_subpaths),
        (inn_path, inn_subpaths),
    ) = create_hdf5_layout(
        dirname,
        (h, w),
        lop_dtype,
        outer,
        inner,
        lo_fmt="leftouter_{}.h5",
        ro_fmt="rightouter_{}.h5",
        inner_fmt="core_{}.h5",
        with_ro=True,
    )
    # perform left outer measurements (distributed)
    for i in range(outer):
        vals, flag, h5 = DistributedHDF5.load(lo_subpaths[i])
        ssrft_idx_torch(
            i,
            outer,
            lop,
            lop_device,
            lop_dtype,
            vals,
            lo_seed,
            flag,
            processing_flag=processing,
            success_flag=success,
            adjoint=True,
        )
        assert flag[0].decode() == success, f"Unsuccessful measurement? {i}"
        h5.close()

    # perform right outer measurements (distributed)
    for i in range(outer):
        vals, flag, h5 = DistributedHDF5.load(ro_subpaths[i])
        ssrft_idx_torch(
            i,
            outer,
            lop,
            lop_device,
            lop_dtype,
            vals,
            ro_seed,
            flag,
            processing_flag=processing,
            success_flag=success,
            adjoint=False,
        )
        assert flag[0].decode() == success, f"Unsuccessful measurement? {i}"
        h5.close()

    # perform inner measurements (distributed)
    for i in range(inner):
        vals, flag, h5 = DistributedHDF5.load(inn_subpaths[i])
        innermeas_idx_torch(
            i,
            inner,
            lop,
            lop_device,
            lop_dtype,
            vals,
            li_seed,
            ri_seed,
            flag=flag,
            processing_flag=processing,
            success_flag=success,
        )
        assert flag[0].decode() == success, f"Unsuccessful measurement? {i}"
        h5.close()

    # merge virtual HDF5 datasets into monolithic without memory overhead
    DistributedHDF5.merge_all(
        lo_path,
        out_path=None,
        delete_subfiles_while_merging=True,
        check_success_flag=success,
    )
    DistributedHDF5.merge_all(
        ro_path,
        out_path=None,
        delete_subfiles_while_merging=True,
        check_success_flag=success,
    )
    DistributedHDF5.merge_all(
        inn_path,
        out_path=None,
        delete_subfiles_while_merging=True,
        check_success_flag=success,
    )

    # load all components and check that all flags are OK (centralized)
    lo, lo_flags, lo_h5 = DistributedHDF5.load(lo_path, filemode="r+")
    ro, ro_flags, ro_h5 = DistributedHDF5.load(ro_path, filemode="r+")
    inn, inn_flags, inn_h5 = DistributedHDF5.load(inn_path, filemode="r+")
    assert all(f.decode() == "1" for f in lo_flags), "Bad left outer flags!"
    assert all(f.decode() == "1" for f in ro_flags), "Bad right outer flags!"
    assert all(f.decode() == "1" for f in inn_flags), "Bad inner flags!"
    # orthogonalize outer measurements (in-place)
    orthogonalize(lo, overwrite=True)
    orthogonalize(ro, overwrite=True)
    # Solve core op and decompose
    left_inner_ssrft = SSRFT((inner, h), seed=li_seed)
    right_inner_ssrft = SSRFT((inner, w), seed=ri_seed)
    core_U, core_S, core_Vt = solve_core_ssvd(
        ro, lo, inn, left_inner_ssrft, right_inner_ssrft
    )
    # for this utest, convert everything to explicit torch tensors
    ro = torch.from_numpy(ro[:]).to(lop_device)
    lo = torch.from_numpy(lo[:]).to(lop_device)
    core_U = torch.from_numpy(core_U).to(lop_device)
    core_S = torch.from_numpy(core_S).to(lop_device)
    core_Vt = torch.from_numpy(core_Vt).to(lop_device)
    # close HDF5 files and return
    lo_h5.close()
    ro_h5.close()
    inn_h5.close()
    return ro, core_U, core_S, core_Vt, lo.T
