#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytests distributed sketched EIGH utils."""


import torch

from skerch.distributed_decompositions import (
    DistributedHDF5,
    create_hdf5_layout,
    orthogonalize,
    solve_core_seigh,
)
from skerch.distributed_measurements import innermeas_idx_torch
from skerch.ssrft import SSRFT


# ##############################################################################
# # MOCK DISTRIBUTED DECOMPOSITIONS
# ##############################################################################
def mock_dseigh(dirname, lop, lop_device, lop_dtype, outer, inner, seed):
    """Mock implementation of a distributed sketched EIGH.

    This function emulates, in a single process, the steps that would be done
    in a distributed system to compute a sketched EIGH.
    """
    assert inner >= outer, "Can't have more outer than inner measurements!"
    processing, success = "0", "1"  # arbitrary flags
    # convenience params
    h, w = lop.shape
    assert h == w, "Square matrix expected!"
    # outer_seed, inner_seed = seed, seed + 1
    li_seed, ri_seed = seed, seed + 1
    # create HDF5 layout (centralized)
    (
        (outer_path, outer_subpaths),
        (_, _),
        (inner_path, inner_subpaths),
    ) = create_hdf5_layout(
        dirname,
        (h, w),
        lop_dtype,
        outer,
        inner,
        lo_fmt="leftouter_{}.h5",
        ro_fmt="rightouter_{}.h5",
        inner_fmt="core_{}.h5",
        with_ro=False,
    )
    # perform inner (and recycled outer) random measurements (distributed)
    for i in range(inner):
        inn_vals, inn_flag, inn_h5 = DistributedHDF5.load(inner_subpaths[i])
        out_buff = innermeas_idx_torch(
            i,
            inner,
            lop,
            lop_device,
            lop_dtype,
            inn_vals,
            li_seed,
            ri_seed,
            flag=inn_flag,
            processing_flag=processing,
            success_flag=success,
        )
        # check successful flags
        assert inn_flag[0].decode() == success, f"Bad inner measurement? {i}"
        inn_h5.close()
        # recycle outer measurements
        if i < outer:
            out_vals, out_flag, out_h5 = DistributedHDF5.load(outer_subpaths[i])
            out_vals[:] = out_buff
            out_flag[0] = success
            assert (
                out_flag[0].decode() == success
            ), f"Bad outer measurement? {i}"
            out_h5.close()
    del out_buff
    # merge virtual HDF5 datasets into monolithic without memory overhead
    DistributedHDF5.merge_all(
        outer_path,
        out_path=None,
        delete_subfiles_while_merging=True,
        check_success_flag=success,
    )
    DistributedHDF5.merge_all(
        inner_path,
        out_path=None,
        delete_subfiles_while_merging=True,
        check_success_flag=success,
    )
    # load all components and check that all flags are OK (centralized)
    outr, outr_flags, o_h5 = DistributedHDF5.load(outer_path, filemode="r+")
    innr, innr_flags, i_h5 = DistributedHDF5.load(inner_path, filemode="r+")
    assert all(f.decode() == "1" for f in outr_flags), "Bad outer flags!"
    assert all(f.decode() == "1" for f in innr_flags), "Bad inner flags!"
    # orthogonalize outer measurements (in-place)
    orthogonalize(outr, overwrite=True)
    # Solve core op and decompose via eigh
    li_ssrft = SSRFT((inner, h), seed=li_seed)
    ri_ssrft = SSRFT((inner, w), seed=ri_seed)
    core_U, core_S = solve_core_seigh(outr, innr, li_ssrft, ri_ssrft)
    # for this utest, convert everything to explicit torch tensors
    outr = torch.from_numpy(outr[:]).to(lop_device)
    core_S = torch.from_numpy(core_S).to(lop_device)
    core_U = torch.from_numpy(core_U).to(lop_device)
    # close HDF5 files and return
    o_h5.close()
    i_h5.close()
    return outr, core_U, core_S
