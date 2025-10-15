#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Persistent and out-of-core tensor functionality via HDF5."""

import os

import h5py


# ##############################################################################
# # DISTRIBUTED HDF5 MANAGER CLASS
# ##############################################################################
class DistributedHDF5Tensor:
    """Class to create and manage distributed HDF5 tensors.

    In general, multiple processes are not allowed to concurrently open and
    write to HDF5 files. In order to allow for distributed, but coherent
    writing, this class allows to create many individual HDF5 files, and then
    "virtually" merge them into a single coherent dataset.
    This allows us to distribute a (potentially large) tensor across processes
    and machines, and write to it concurrently.

    Once we are done writing, we may want to access the result.
    Unfortunately, most OS don't allow a single process to open many files at
    once. As a result, any files above the limit would be silently ignored.
    This class also solves this by providing a :meth:`merge` method, to gather
    the distributed chunks back into a single, monolithic file.

    See docs for more examples, also on how to create and merge HDF5 datasets
    directly from command line.

    .. note:

      This class creates all files with a given naming pattern that should
      not be modified. All files are created, and expected to be in the same
      directory, which should be uniquely dedicated to a particular dataset.
      To ensure correct function, avoid manual modification of filepaths
      and use this class to create, modify and merge datasets.
      Since this is a static class, it can also work across multiple
      concurrent processes/devices.
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
        """Helper method, retrieves format strings to index HDF5 chunks."""
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

        :param basepath_fmt: Format string with the directory and dataset
          name to store created HDF5 files, in the form
          ``<DIR>/my_dataset_{}.h5``. Directory must be empty.
        :param shape: Shape of the global tensor corresponding to the whole
          HDF5 dataset.
        :param partition_size: The HDF5 dataset will be partitioned across
          its first axis on sub-files. E.g. a shape of ``(10, 20)`` with a
          partition size of 6 will result in 2 files of shapes ``(6, 20)``
          and ``(4, 20)``.
        :param dtype: Datatype of the HDF5 arrays to be created.
        :returns: The tuple ``(all_path, subpaths, begs_ends)``, where
          ``all_path`` is the  path of the virtual dataset encompassing
          the global tensor, ``subpaths`` are the paths to the respective
          chunk HDF5 files, and ``begs_ends`` are the begining (included)
          and end (excluded) indices that were used to partition the global
          tensor on subfiles, across its first axis.
        """
        # extract total idxs and figure how are they partitioned into subfiles
        max_idx = shape[0]
        div, mod = divmod(max_idx, partition_size)
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
        """Load the given HDF5 dataset.

        Load an individual dataset, such as the virtual ones created via
        :meth:`.create` or the monolithic ones merged via :meth:`merge`.

        :param path: Path to the HDF5 file.
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
    def merge(  # noqa: C901
        cls,
        all_path,
        out_path=None,
        compression="lzf",
        check_success_flag=None,
        delete_subfiles_while_merging=False,
    ):
        """Merges distributed HDF5 dataset into a single, monolithic HDF5 file.

        :param all_path: The ``all_path`` of a virtual HDF5 dataset like the
          ones created via :meth:`.create`. It must be a "virtual" dataset,
          i.e. composed of chunks that are distributed across other files.
        :param out_path: If None, merged dataset will be written over the given
          ``all_path``, i.e. it will be converted from virtual into monolithic
          in-place. Otherwise, path for a new HDF5 monolithic file where
          the contents will be written into.
        :param check_success_flag: If given, this method will check that all
          HDF5 flags equal this value, raise an error otherwise.
        :param bool delete_subfiles_while_merging: If true, each distributed
          HDF5 file that is visited will be deleted right after it is merged
          onto the monolithic HDF5 file. Useful to avoid large memory
          overhead.
        :returns: Path of the merged HDF5 file.
        """
        all_data, all_flags, all_h5 = cls.load(all_path)
        shape = all_data.shape
        data_dtype = all_data.dtype
        flags_dtype = all_flags.dtype
        max_idx = shape[0]
        if not all_data.is_virtual:
            raise ValueError(f"data in {all_path} not virtual!")
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
    dtype,
    partition_size,
    lo_meas=None,
    ro_meas=None,
    inner_meas=None,
    lo_fmt="leftouter_{}.h5",
    ro_fmt="rightouter_{}.h5",
    inner_fmt="inner_{}.h5",
):
    """Creation of persistent HDF5 files to store linop sketches.

    This convenience method prepaers the HDF5 placeholders that can be used
    to store sketches from a linop of shape ``lop_shape``.
    It supports independent creation of left-, right- and inner measurements,
    thus supporting most use cases involving linear sketches.

    :param root: Where to store the created HDF5 files. Must be an empty
      directory.
    :param lop_shape: Shape of linear operator to sketch from, in the form
      ``(height, width)``.
    :param dtype: Torch dtype of the operator, e.g. ``torch.float32``. The
      HDF5 arrays will be of same type.
    :param partition_size: Each created HDF5 will be split into chunks of this
      many vectors (see :meth:`DistributedHDF5Tensor.create` for more details).
    :param lo_meas: If given, a dataset of shape ``(lo_meas, w)`` will be
      created under the name given by ``lo_fmt``.
    :param ro_meas: If given, a dataset of shape ``(h, ro_meas)`` will be
      created under the name given by ``ro_fmt``.
    :param inner_meas: If given, a dataset of shape
      ``(inner_meas, inner_meas)`` will be created under the name given by
      ``inner_fmt``.
    """
    h, w = lop_shape
    #
    lo_pth, lo_subpaths, lo_begs_ends = None, None, None
    ro_pth, ro_subpaths, ro_begs_ends = None, None, None
    in_pth, in_subpaths, in_begs_ends = None, None, None
    #
    if lo_meas is not None:
        lo_pth, lo_subpaths, lo_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, lo_fmt), (lo_meas, w), partition_size, dtype
        )
    #
    if ro_meas is not None:
        ro_pth, ro_subpaths, ro_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, ro_fmt), (ro_meas, w), partition_size, dtype
        )
    #
    if inner_meas is not None:
        in_pth, n_subpaths, in_begs_ends = DistributedHDF5Tensor.create(
            os.path.join(root, inner_fmt),
            (inner_meas, inner_meas),
            partition_size,
            dtype,
        )
    #
    return (
        (lo_pth, lo_subpaths, lo_begs_ends),
        (ro_pth, ro_subpaths, ro_begs_ends),
        (in_pth, in_subpaths, in_begs_ends),
    )
