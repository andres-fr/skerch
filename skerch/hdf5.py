#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""

import os
import h5py
from .utils import torch_dtype_as_str


# ##############################################################################
# # DISTRIBUTED HDF5 MANAGER CLASS
# ##############################################################################
class DistributedHDF5Tensor:
    """Class to create and manage distributed HDF5 tensors.

    In general, multiple processes are not allowed to concurrently open and
    write to HDF5 files. In order to allow for distributed, but coherent
    writing, this class allows to create many individual HDF5 files, and then
    'virtually' merge them into a single coherent dataset.

    Unfortunately, most OS don't allow a single process to open many files at
    once, and each sub-file here counts as one. As a result, loading such
    distributed HDF5 datasets will reach a breaking point, after which contents
    are not really being opened, and reported as "empty" (e.g. full of zeros).
    For those cases, this class also provides a ``merge_all`` method,
    to efficiently merge the distributed dataset into a monolithic one.

    In the virtual mode, all files are created in the same directory following
    a consistent naming convention, and this is expected to remain unchanged.
    Since this class is naturally intended for multiple concurrent
    processes/devices, it is designed in the form of a static class.

    Usage example (see :mod:`.hdf5` for more examples)::

      # create the empty separate HDF5 files on disk, and the "virtual" merger
      out_path = "/tmp/my_dataset_{}.h5"
      each_shape, num_files = (1000,), 5
      h5_path, h5_subpaths = DistributedHDF5.create(
        out_path, num_files, each_shape, torch_dtype_as_str(op_dtype)
      )

      # in a (possibly distributed & parallelized) loop, load and write parts
      for i in range(num_files):
          vals, flag, h5 = DistributedHDF5.load(h5_subpaths[i])
          vals[:] += 1
          flag[0] = 'done'
          h5.close()  # remember to close file handles when done!

      # merged data can be used as a (1000, 50) matrix by a separate machine
      all_data, all_flags, all_h5 = DistributedHDF5.load_virtual(h5_path)
      print(scipy.diag(all_data))
      print(all_flags[::2])
      all_h5.close()

      # convert virtual database into single monolithic one
      DistributedHDF5.merge_all(
          h5_path,
          delete_subfiles_while_merging=True,
      )
      all_data, all_flags, all_h5 = DistributedHDF5.load_virtual(h5_path)
      ...
    """

    MAIN_PATH = "ALL"
    DATA_NAME = "data"
    FLAG_NAME = "flags"
    FLAG_DTYPE = h5py.string_dtype()
    INITIAL_FLAG = "initialized"

    @staticmethod
    def iter_partition_idxs(max_idx, partition_size):
        """Iterate from 0 to ``max_idx`` by ``partition_size``.

        :returns: Pairs of ``(beg, end)`` where ``beg`` is included and begins
          with 0, and ``end`` is exluded and equals at most ``max_idx``.
        """
        for beg in range(0, max_idx, partition_size):
            end = min(beg + partition_size, max_idx)
            yield (beg, end)

    @staticmethod
    def get_idxs_format(max_idx):
        """ """
        result = (
            "{0:0" + str(len(str(max_idx)) + 1) + "d}-"
            "{1:0" + str(len(str(max_idx)) + 1) + "d}"
        )
        return result

    @classmethod
    def create(
        cls,
        basepath_fmt,
        shape,
        partition_size,
        dtype,
        compression="lzf",
    ):
        """Create a distributed HDF5 measurement dataset.

        :param str base_path: Format string with the path to store created HFG5
          files, in the form ``<DIR>/my_dataset_{}.h5``.
        :param num_files: Number of HDF5 files to be created, each
          corresponding to one measurement.
        :param shape: Shape of the global tensorarray inside each individual
          file. For linear measurements, this is a vector, e.g. ``(1000,)``.
        :param dtype: Datatype of the HDF5 arrays to be created.
        :param filedim_last: If true, the virtual dataset result of merging
          all files will be of shape ``shape + (num_files,)``. Otherwise,
          ``(num_files,) + shape``.
        :returns: The pair ``(all_path, subpaths)``, where ``all_path`` is the
          path of the virtual dataset encompassing all subpaths, which
          correspond to the individual files.
        """
        # extract total idxs and figure how are they partitioned into subfiles
        max_idx = shape[0]
        div, mod = divmod(max_idx, partition_size)
        num_partitions = div + (mod != 0)
        idxs_fmt = cls.get_idxs_format(max_idx)
        all_path = basepath_fmt.format(cls.MAIN_PATH)
        begs_ends = list(cls.iter_partition_idxs(max_idx, partition_size))
        # create virtual dataset to hold everything together via softlinks
        # use relative paths to just assume everything is in the same dir
        data_lyt = h5py.VirtualLayout(shape=shape, dtype=dtype)
        flag_lyt = h5py.VirtualLayout(shape=(max_idx,), dtype=cls.FLAG_DTYPE)
        subpaths = []
        for beg, end in begs_ends:
            subpath = basepath_fmt.format(idxs_fmt.format(beg, end))
            subpaths.append(subpath)
            p = os.path.basename(subpath)
            subshape = (end - beg,) + shape[1:]
            #
            dvs = h5py.VirtualSource(p, cls.DATA_NAME, shape=subshape)
            data_lyt[beg:end] = dvs
            #
            fvs = h5py.VirtualSource(p, cls.FLAG_NAME, shape=(end - beg,))
            flag_lyt[beg:end] = fvs
        #
        all_h5 = h5py.File(all_path, "w")  # , libver="latest")
        all_h5.create_virtual_dataset(cls.DATA_NAME, data_lyt)
        all_h5.create_virtual_dataset(cls.FLAG_NAME, flag_lyt)
        all_h5.close()
        # now create the actual sub-files
        for beg, end in begs_ends:
            subpath = basepath_fmt.format(idxs_fmt.format(beg, end))
            sublen = end - beg
            subshape = (sublen,) + shape[1:]
            h5f = h5py.File(subpath, "w")
            h5f.create_dataset(
                cls.DATA_NAME,
                shape=subshape,
                maxshape=subshape,
                dtype=dtype,
                compression=compression,
                chunks=subshape,
            )
            h5f.create_dataset(
                cls.FLAG_NAME,
                shape=(sublen,),
                maxshape=(sublen,),
                compression=compression,
                dtype=cls.FLAG_DTYPE,
                chunks=(sublen,),
            )
            h5f[cls.FLAG_NAME][:] = cls.INITIAL_FLAG
            h5f.close()
        #
        return all_path, subpaths, begs_ends

    @classmethod
    def load(cls, path, filemode="r+"):
        """Load an individual dataset.

        Load an individual dataset, such as the ones created via
        :meth:`.create` or merged via :meth:`merge_all`.

        :param path: One of the subpaths returned by :meth:`.create`.
        :param filemode: Default is 'r+', read/write, file must preexist. See
          documentation of ``h5py.File`` for more details.
        :returns: ``(data, flag, h5f)``, where ``data`` is the dataset
          for the numerical measurements, ``flag`` is the dataset for state
          tracking, and ``h5f`` is the (open) HDF5 file handle.

        .. note::

          Remember to ``h5f.close()`` once done with this file.
        """
        h5f = h5py.File(path, filemode)
        data = h5f[cls.DATA_NAME]
        flag = h5f[cls.FLAG_NAME]
        return data, flag, h5f

    @classmethod
    def merge(
        cls,
        all_path,
        out_path=None,
        compression="lzf",
        check_success_flag=None,
        delete_subfiles_while_merging=False,
    ):
        """Merges distributed HDF5 dataset into a single, monolithic HDF5 file.

        :param all_path: The ``all_path`` of a virtual HDF5 dataset like the
          ones created via :meth:`.create`.
        :param out_path: If None, merged dataset will be written over the given
          ``all_path``. Otherwise, path of the resulting HDF5 monolithic file.
        :param check_success_flag: If given, this method will check that all
          HDF5 flags equal this value, raise an ``AssertionError`` otherwise.
        :param bool delete_subfiles_while_merging: If true, each distributed
          HDF5 file that is visited will be deleted right after it is merged
          onto the monolithic HDF5 file. Useful to avoid large memory
          overhead.
        :returns: ``out_path``.
        """
        all_data, all_flags, all_h5 = cls.load(all_path)
        shape = all_data.shape
        data_dtype = all_data.dtype
        flags_dtype = all_flags.dtype
        max_idx = shape[0]
        if not all_data.is_virtual:
            raise ValueError(f"{dataset_name} in {all_path} not virtual!")
        # inspect virtual sources to get info about the chunks
        vs_info = {}
        partition_size = float("-inf")
        for vs in all_data.virtual_sources():
            path = os.path.join(os.path.dirname(all_h5.filename), vs.file_name)
            begs, ends = vs.vspace.get_select_bounds()
            beg, end = begs[0], ends[0] + 1
            vs_info[(beg, end)] = path
            partition_size = max(partition_size, end - beg)
        all_h5.close()
        num_partitions = len(vs_info)
        # check that all expected indices exist, and flags are as expected
        for beg, end in cls.iter_partition_idxs(max_idx, partition_size):
            if (not (beg, end) in vs_info) or (
                not os.path.isfile(vs_info[(beg, end)])
            ):
                raise ValueError(f"Can't merge! malformed dataset: {vs_info}")
            if check_success_flag is not None:
                subpath = vs_info[(beg, end)]
                subdata, subflags, h5 = cls.load(subpath, filemode="r")
                for flg in subflags:
                    if flg.decode() != check_success_flag:
                        raise ValueError(f"Can't merge! Bad flag: {flg}")
        # OK to merge: create merged output dataset, initially empty
        if out_path is None:
            out_path = all_path
        h5f = h5py.File(out_path, "w")
        h5f.create_dataset(
            cls.DATA_NAME,
            shape=(0,) + shape[1:],
            maxshape=shape,
            dtype=data_dtype,
            compression=compression,
            chunks=(partition_size,) + shape[1:],
        )
        h5f.create_dataset(
            cls.FLAG_NAME,
            shape=0,
            maxshape=max_idx,
            dtype=flags_dtype,
            compression=compression,
            chunks=partition_size,
        )
        # iterate over contents in sorted order and extend h5f with them
        for beg, end in cls.iter_partition_idxs(max_idx, partition_size):
            subpath = vs_info[(beg, end)]
            subdata, subflags, h5 = cls.load(subpath, filemode="r")
            datashape = h5f[cls.DATA_NAME].shape
            new_datashape = (datashape[0] + end - beg,) + datashape[1:]
            #
            h5f[cls.DATA_NAME].resize(new_datashape)
            h5f[cls.DATA_NAME][beg:end] = subdata
            #
            h5f[cls.FLAG_NAME].resize((new_datashape[0],))
            h5f[cls.FLAG_NAME][beg:end] = subflags
            #
            h5.close()
            # optionally, delete subfile
            if delete_subfiles_while_merging:
                os.remove(subpath)
        #
        h5f.close()
        return out_path


# ##############################################################################
# # CONVENIENCE FUNCTIONS
# ##############################################################################
def create_hdf5_layout_lop(
    root,
    lop_shape,
    lop_dtype,
    partition_size,
    num_outer_measurements=None,
    num_inner_measurements=None,
    lo=True,
    ro=True,
    inner=True,
    lo_fmt="leftouter_{}.h5",
    ro_fmt="rightouter_{}.h5",
    inner_fmt="inner_{}.h5",
):
    """Creation of persistent HDF5 files to store linop sketches.

    :param str dirpath: Where to store the HDF5 files.
    :param lop_shape: Shape of linear operator to sketch from, in the form
      ``(height, width)``.
    :param lop_dtype: Torch dtype of the operator, e.g. ``torch.float32``. The
      HDF5 arrays will be of same type.
    :param int num_outer_measurements: Left outer measurement layout contains
      ``(width, outer)`` entries, and right outer layout ``(height, outer)``.
    :param int num_inner_measurements: Inner measurement layout contains
      ``(inner, inner)`` entries.
    :lo_fmt: Format string for the left-outer HDF5 filenames.
    :ro_fmt: Format string for the right-outer HDF5 filenames.
    :inner_fmt: Format string for the inner HDF5 filenames.
    :param with_ro: If false, no right outer layout will be created (useful
      when working with symmetric matrices where only one side is needed).
    """
    h, w = lop_shape
    strtype = torch_dtype_as_str(lop_dtype)
    #
    if (ro or lo) and (num_outer_measurements is None):
        raise ValueError("lo/ro require to provide num_outer_measurements!")
    if inner and (num_inner_measurements is None):
        raise ValueError("inner requires to provide num_inner_measurements!")
    #
    lo_pth, lo_subpaths, lo_begs_ends = None, None, None
    ro_pth, ro_subpaths, ro_begs_ends = None, None, None
    in_pth, in_subpaths, in_begs_ends = None, None, None
    #
    if lo:
        lo_pth, lo_subpaths, lo_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, lo_fmt),
            (num_outer_measurements, w),
            partition_size,
            strtype,
        )
    #
    if ro:
        ro_pth, ro_subpaths, ro_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, ro_fmt),
            (num_outer_measurements, h),
            partition_size,
            strtype,
        )
    #
    if inner:
        in_pth, n_subpaths, in_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, inner_fmt),
            (num_inner_measurements, num_inner_measurements),
            partition_size,
            strtype,
        )
    #
    return (
        (lo_pth, lo_subpaths, lo_begs_ends),
        (ro_pth, ro_subpaths, ro_begs_ends),
        (in_pth, in_subpaths, in_begs_ends),
    )
