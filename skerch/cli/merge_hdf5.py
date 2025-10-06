#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""CLI plugin to merge distributed HDF5 measurements into a single one."""


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
    """See :meth:`.distributed_measurements.DistributedHDF5.merge_all`."""
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
