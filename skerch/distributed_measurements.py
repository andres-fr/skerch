#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for distributed sketch measurements.

One typically expensive step when we want to do sketched decompositions is
taking measurements from the linear operator we want to decompose, either
in terms of runtime, memory, or both. This module provides functionality to
parallelize and distribute said measurements.
"""


import os

import h5py
import torch

from .ssrft import SSRFT


# ##############################################################################
# # DECOMPOSITION HELPERS
# ##############################################################################
class DistributedHDF5:
    """Class to manage HDF5 files holding distributed sketch measurements.

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

    Usage example (see :mod:`.distributed_decompositions` for more examples)::

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
    SUBPATHS_FORMAT = "{0:010d}"
    DATA_NAME = "data"
    FLAG_NAME = "flags"
    FLAG_DTYPE = h5py.string_dtype()
    INITIAL_FLAG = "initialized"

    @classmethod
    def create(
        cls,
        base_path,
        num_files,
        shape,
        dtype,
        compression="lzf",
        filedim_last=True,
    ):
        """Create a distributed HDF5 measurement dataset.

        :param str base_path: Format string with the path to store created HFG5
          files, in the form ``<DIR>/my_dataset_{}.h5``.
        :param num_files: Number of HDF5 files to be created, each
          corresponding to one measurement.
        :param shape: Shape of the array inside each individual file. For
          linear measurements, this is a vector, e.g. ``(1000,)``.
        :param dtype: Datatype of the HDF5 arrays to be created.
        :param filedim_last: If true, the virtual dataset result of merging
          all files will be of shape ``shape + (num_files,)``. Otherwise,
          ``(num_files,) + shape``.
        :returns: The pair ``(all_path, subpaths)``, where ``all_path`` is the
          path of the virtual dataset encompassing all subpaths, which
          correspond to the individual files.
        """
        all_path = base_path.format(cls.MAIN_PATH)
        subpaths = [
            base_path.format(cls.SUBPATHS_FORMAT.format(i))
            for i in range(num_files)
        ]
        # create virtual dataset to hold everything together via softlinks
        # use relative paths to just assume everything is in the same dir
        if filedim_last:
            data_shape = shape + (num_files,)
        else:
            data_shape = (num_files,) + shape
        data_lyt = h5py.VirtualLayout(shape=data_shape, dtype=dtype)
        flag_lyt = h5py.VirtualLayout(shape=(num_files,), dtype=cls.FLAG_DTYPE)
        for i, p in enumerate(subpaths):
            p = os.path.basename(p)  # relative path
            vs = h5py.VirtualSource(p, cls.DATA_NAME, shape=shape)
            if filedim_last:
                data_lyt[..., i] = vs
            else:
                data_lyt[i] = vs
            flag_lyt[i] = h5py.VirtualSource(p, cls.FLAG_NAME, shape=(1,))
        all_h5 = h5py.File(all_path, "w")  # , libver="latest")
        all_h5.create_virtual_dataset(cls.DATA_NAME, data_lyt)
        all_h5.create_virtual_dataset(cls.FLAG_NAME, flag_lyt)
        all_h5.close()
        # create separate HDF5 files
        for p in subpaths:
            h5f = h5py.File(p, "w")
            h5f.create_dataset(
                cls.DATA_NAME,
                shape=shape,
                maxshape=shape,
                dtype=dtype,
                compression=compression,
                chunks=shape,
            )
            h5f.create_dataset(
                cls.FLAG_NAME,
                shape=(1,),
                maxshape=(1,),
                compression=compression,
                dtype=cls.FLAG_DTYPE,
                chunks=(1,),
            )
            h5f[cls.FLAG_NAME][:] = cls.INITIAL_FLAG
            h5f.close()
        #
        return all_path, subpaths

    @classmethod
    def load(cls, path, filemode="r+"):
        """Load an individual dataset created via :meth:`.create`.

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
    def load_virtual(cls, all_path, filemode="r+"):
        """Load a merged dataset created via :meth:`.create`.

        :param path: The ``all_path`` returned by :meth:`.create`.
        :param filemode: Default is 'r+', read/write, file must preexist. See
          documentation of ``h5py.File`` for more details.
        :returns: ``(data, flag, h5f)``, where ``data`` is the dataset
          for the numerical measurements, ``flag`` is the dataset for state
          tracking, and ``h5f`` is the (open) HDF5 file handle.

        .. note::

          Remember to ``h5f.close()`` once done with this file.
        """
        all_h5f = h5py.File(all_path, filemode)
        data = all_h5f[cls.DATA_NAME]
        flags = all_h5f[cls.FLAG_NAME]
        return data, flags, all_h5f

    @classmethod
    def analyze_virtual(cls, all_path, check_success_flag=None):
        """Analyze shapes and relations of files created via :meth:`.create`.

        Extract relevant information from a given virtual HDF5 dataset, which
        can be used e.g. to allow merging it into a monolithic one (see
        implementation of :meth:`merge_all` for an example).

        :param all_path: The ``all_path`` of a virtual HDF5 dataset like the
          ones created via :meth:`.create`.
        :param check_success_flag: If given, this method will check that all
          HDF5 flags equal this value, raise an ``AssertionError`` otherwise.
        :returns: the tuple ``(data_shape, data_dtype, data_subshapes,
          data_map, flags_shape, flags_dtype, flag_subshapes, flag_map,
          filedim_idx)``.
        """
        # load virtual HDF5 and grab main infos. Note that we avoid
        # using data and flags for anything else, since they may be affected by
        # the "too many open files" OS issue.
        data, flags, h5f = DistributedHDF5.load_virtual(all_path, filemode="r")
        data_shape, flags_shape = data.shape, flags.shape
        data_dtype, flags_dtype = data.dtype, flags.dtype
        abspath = h5f.filename
        rootdir = os.path.dirname(abspath)
        # figure out involved paths and their respective indices in virtual
        data_map, flag_map = {}, {}
        data_subshapes, flag_subshapes = [], []
        for vs in h5f[cls.DATA_NAME].virtual_sources():
            subpath = os.path.join(rootdir, vs.file_name)
            begs_ends = tuple(
                (b, e + 1) for b, e in zip(*vs.vspace.get_select_bounds())
            )
            shape = tuple(e - b for b, e in begs_ends)
            data_map[begs_ends] = subpath
            data_subshapes.append(shape)
        for vs in h5f[cls.FLAG_NAME].virtual_sources():
            subpath = os.path.join(rootdir, vs.file_name)
            begs_ends = tuple(
                (b, e + 1) for b, e in zip(*vs.vspace.get_select_bounds())
            )
            shape = tuple(e - b for b, e in begs_ends)
            assert shape == (1,), "Flags expected to have shape (1,)!"
            flag_map[begs_ends] = subpath
            flag_subshapes.append(shape)
        subpaths = set(data_map.values())
        # figure out position of filedim was first or last
        is_filedim = [
            (a - b) == 0 for a, b in zip(data_shape, data_subshapes[0])
        ]
        assert sum(is_filedim) == 1, "Only one running dimension supported!"
        filedim_idx = is_filedim.index(False)
        # virtual sanity check and close virtual
        data_beginnings = {k[filedim_idx][0] for k in data_map.keys()}
        assert len(data_beginnings) == len(
            h5f[cls.DATA_NAME].virtual_sources()
        ), "Repeated file_idx beginnings in data?"
        assert len(flag_map) == len(
            h5f[cls.FLAG_NAME].virtual_sources()
        ), "Repeated indices in flags?"
        for sp in subpaths:
            assert os.path.isfile(sp), f"Subpath doesn't exist! {sp}"
        for sp2 in flag_map.values():
            assert sp2 in subpaths, "Flag subpaths different to data subpath!"
        assert (
            len(set(data_subshapes)) == 1
        ), "Heterogeneous data shapes in virtual dataset not supported!"
        assert (
            len(set(flag_subshapes)) == 1
        ), "Heterogeneous flag shapes in virtual dataset not supported!"
        h5f.close()
        # if success flag was given, check every individual sub-HDF5 was
        # successful
        if check_success_flag is not None:
            for sp in subpaths:
                _, flag, h5 = cls.load(sp, filemode="r")
                flag = flag[0].decode()
                assert (
                    flag == check_success_flag
                ), f"Unsuccessful flag in {sp}! {flag}"
                h5.close()
        #
        return (
            data_shape,
            data_dtype,
            data_subshapes,
            data_map,
            flags_shape,
            flags_dtype,
            flag_subshapes,
            flag_map,
            filedim_idx,
        )

    @classmethod
    def merge_all(
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
        # grab relevant infos from virtual, we still didn't modify anything
        (
            data_shape,
            data_dtype,
            data_subshapes,
            data_map,
            flags_shape,
            flags_dtype,
            flag_subshapes,
            flag_map,
            filedim_idx,
        ) = cls.analyze_virtual(all_path, check_success_flag)
        # create HDF5 that we will iteratively expand. If no out_path,
        # we will overwrite virtual.
        if out_path is None:
            out_path = all_path
        h5f = h5py.File(out_path, "w")
        #
        init_shape = list(data_shape)
        init_shape[filedim_idx] = 0
        h5f.create_dataset(
            cls.DATA_NAME,
            shape=init_shape,
            maxshape=data_shape,
            dtype=data_dtype,
            compression=compression,
            chunks=data_subshapes[0],
        )
        h5f.create_dataset(
            cls.FLAG_NAME,
            shape=0,
            maxshape=flags_shape,
            dtype=flags_dtype,
            compression=compression,
            chunks=flag_subshapes[0],
        )
        # iterate over contents in sorted order and extend h5f with them
        sorted_data = sorted(data_map, key=lambda x: x[filedim_idx][0])
        for begs_ends in sorted_data:
            subpath = data_map[begs_ends]
            subdata, subflag, h5 = cls.load(subpath, filemode="r")
            if check_success_flag is not None:
                assert (
                    subflag[0].decode() == check_success_flag
                ), f"Subfile flag not equal {check_success_flag}!"
            # increment size of h5f by 1 entry
            data_shape = list(h5f[cls.DATA_NAME].shape)
            data_shape[filedim_idx] += 1
            h5f[cls.DATA_NAME].resize(data_shape)
            h5f[cls.FLAG_NAME].resize((len(h5f[cls.FLAG_NAME]) + 1,))
            # write subdata and subflags to h5f, flush and close subfile
            target_slices = tuple(slice(*be) for be in begs_ends)
            h5f[cls.DATA_NAME][target_slices] = subdata[:].reshape(
                data_subshapes[0]
            )
            h5f[cls.FLAG_NAME][-1:] = subflag[:]
            h5f.flush()
            h5.close()
            # optionally, delete subfile
            if delete_subfiles_while_merging:
                os.remove(subpath)
        #
        h5f.close()
        return out_path


# ##############################################################################
# # DISTRIBUTED MEASUREMENTS
# ##############################################################################
def linmeas_idx_torch(idx, lop, meas_lop, lop_device, lop_dtype, adjoint=False):
    """Distributed-able linear measurement from given linear operators.

    Convenience function to perform a single linear measurement from two
    (potentially matrix-free) linear operators. Specifically, it picks the
    ``[idx]`` row of ``meas_lop``, and applies it to ``lop``.

    :param lop: Linear operator to measure from, typically matrix-free or very
      costly to store or compute. Assumed to implement right- and left matrix
      multiplication via the ``@`` operator, and also the ``shape = (h, w)``
      attribute, but no transposition.
    :param meas_lop: Linear operator to be applied to ``lop``. It must also
      support ``@`` operator and ``shape`` attribute, but not transposition is
      required. For this reason, its shape is ``(*, h)`` if adjoint is true,
      and ``(*, w)`` otherwise, where ``*`` is arbitrary.
    :returns: If adjoint is True, ``meas_lop[idx, :] @ lop``. Otherwise
      ``lop @ meas_lop[idx, :].T``.
    """
    rh, rw = meas_lop.shape
    v = torch.zeros(rh, dtype=lop_dtype).to(lop_device)
    v[idx] = 1
    try:
        if adjoint:
            result = (v @ meas_lop) @ lop
        else:
            result = lop @ (v @ meas_lop)
    except TypeError as te:
        msg = "Error during matmul! LinOp expected to work with torch tensors."
        raise TypeError(msg) from te
    return result


def ssrft_idx_torch(
    idx,
    total_measurements,
    lop,
    lop_device,
    lop_dtype,
    out_vals,
    seed=0b1110101001010101011,
    flag=None,
    processing_flag="-1",
    success_flag="1",
    adjoint=False,
):
    """Distributed-able SSRFT measurement from given linear operators.

    .. note::

      See :mod:`.ssrft` for more details on the SSRFT and its implementation.

    In-place SSRFT measurement, it projects the ``idx`` row of a SSRFT operator
    onto the given ``lop``, writing results to ``out_vals``. If adjoint is
    true, computes ``ssrft[idx, :] @ lop``, otherwise ``lop @ ssrft[idx, :].T``,
    where ``ssrft``is always of shape ``(total_measurements, *)`` with ``*``
    being the fitting dimension of ``lop``.

    :param out_vals: Array where the results will be writen in-place, of shape
      ``(*, )`` for either the height or width of ``lop``. It can be a ``torch``
      tensor or a ``numpy`` or a ``HDF5`` array, accepting in-place  writing
      via ``out_vals[:] = ...``.
    :param flag: An array-like object of shape (1,) and string datatype,
      optionally used by this function to keep track of current state by
      setting it to the ``processing_flag, success_flag`` values via
      the ``flag[0] = "..."`` assignment.
    :param bool adjoint: See above.
    :returns: Nothing, result is written to ``out_vals``, and optionally
      ``flag`` is also written.
    """
    assert (
        0 <= idx <= total_measurements
    ), "OOB index for number of measurements!"
    vals_shape = out_vals.shape
    lop_h, lop_w = lop.shape
    if adjoint:
        ssrft = SSRFT((total_measurements, lop_h), seed=seed)
        assert vals_shape == (lop_w,), "Mismatching measurement and out_shapes!"
    else:
        ssrft = SSRFT((total_measurements, lop_w), seed=seed)
        assert vals_shape == (lop_h,), "Mismatching measurement and out_shapes!"
    #
    if flag is not None:
        flag[0] = processing_flag
    #
    result = linmeas_idx_torch(idx, lop, ssrft, lop_device, lop_dtype, adjoint)
    try:
        # try to assign as-is (covers torch-to-torch and CPU cases)
        out_vals[:] = result
    except Exception:
        # if anything happens, try to copy to CPU
        out_vals[:] = result.cpu()
    if flag is not None:
        flag[0] = success_flag


# ##############################################################################
# # DISTRIBUTED INNER MEASUREMENTS
# ##############################################################################
def innermeas_idx_torch(
    idx,
    num_measurements,
    lop,
    lop_device,
    lop_dtype,
    out,
    left_seed,
    right_seed,
    flag=None,
    processing_flag="0",
    success_flag="1",
):
    """Distributed-able SSRFT inner measurement for sketched decompositions.

    .. note::

      See :mod:`.ssrft` for more details on the SSRFT and its implementation.

    Like :func:`ssrft_idx_torch`, but it applies the SSRFT on both sides of
    ``lop``, i.e. it computes: ``ssrft_l @ lop @ ssrft_r[idx, :].T``, where
    ``ssrft_l`` has ``left_seed`` and shape ``(num_measurements, lop.h)``, and
    ``ssrft_r`` has ``right_seed`` and shape ``(num_measurements, lop.w)``.

    :param out: Array where the results will be writen in-place, of shape
      ``(num_measurements, )``. It can be a ``torch`` tensor or a ``numpy`` or
      a ``HDF5`` array, accepting in-place  writing via ``out[:] = ...``.
    :param flag: See :func:`ssrft_idx_torch`.
    :returns: The in-between result ``lop @ ssrft_r[idx, :].T``, as a torch
      tensor of shape ``(lop_height,)``. It will be on CUDA device if
      ``lop_device`` is CUDA, otherwise CPU.
    """
    h, w = lop.shape
    #
    li_ssrft = SSRFT((num_measurements, h), seed=left_seed)
    buff = torch.zeros(h, dtype=lop_dtype).to(lop_device)
    ssrft_idx_torch(
        idx,
        num_measurements,
        lop,
        lop_device,
        lop_dtype,
        buff[:],
        right_seed,
        flag,
        processing_flag=processing_flag,
        success_flag=success_flag,
        adjoint=False,
    )
    if flag is not None:
        flag[0] = processing_flag
    try:
        # try to assign as-is (covers torch-to-torch and CPU cases)
        out[:] = li_ssrft @ buff
    except Exception:
        # if anything happens, try to copy to CPU
        out[:] = (li_ssrft @ buff).cpu()
        buff = buff.cpu()
    if flag is not None:
        flag[0] = success_flag
    return buff
