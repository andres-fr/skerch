#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""CLI plugin to merge distributed HDF5 measurements into a single one.


See :mod:`skerch.hdf5` and :meth:`skerch.hdf5.DistributedHDF5Tensor.merge`.
"""


from ..hdf5 import DistributedHDF5Tensor


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(
    all_path,
    out_path=None,
    ok_flag=None,
    delete_subfiles=True,
):
    """Entry point for this CLI script. See module docstring."""
    # merge virtual HDF5 datasets into monolithic without memory overhead
    merged_path = DistributedHDF5Tensor.merge(
        all_path,
        out_path=out_path,
        check_success_flag=ok_flag,
        delete_subfiles_while_merging=delete_subfiles,
    )
    print("Merged all sub-files of", all_path, "into monolithic", merged_path)
    if delete_subfiles:
        print("Also deleted sub-files.")
