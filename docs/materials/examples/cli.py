# -*- coding: utf-8 -*-

r"""Command Line Interface
==========================

This example demonstrates ``skerch`` functionality directly accessible
from the command line interface (CLI).

Note that all CLI functionality can also be directly accessed from Python.
Check the API docs for comprehensive information, e.g.:

* :mod:`skerch.a_posteriori`
* :mod:`skerch.hdf5`


To run these commands, we simply assume that ``skerch`` is accessible to the
available ``python`` interpreter. The following (standard) imports are only
needed to run this example:
"""


import os
from tempfile import TemporaryDirectory

from skerch.__main__ import main_wrapper as skerch_main

# %%
#
# ##############################################################################
#
# CLI help documentation
# ----------------------
#
# The following command provides information about positional and optional
# CLI arguments::
#
#   python -m skerch -h
#
# And the corresponding output looks like this:


skerch_main(["-h"])


# %%
#
# ##############################################################################
#
# A-posteriori error bounds
# -------------------------
#
# It is possible to efficiently estimate the Frobenius distance between
# any two linear operators via sketches (see e.g. :mod:`skerch.a_posteriori`
# or e.g. :ref:`Sketched Low-Rank Decompositions` for more information and
# examples on how to run this estimation with ``skerch``).
#
# In a nutshell, we apply the same random "test" sketch to both operators, and
# compare the distance between measurements, which becomes is a proxy for the
# distance between the operators.
#
# Since this is a randomized estimation, it is subject to error, and there
# are probabilistic bounds that allow us to know the probability that a given
# error may have occurred. Interestingly, this probability does *not* depend
# on the size of the operators, but on the number of "test" measurements
# performed (and whether the operators are real-valued or complex).
#
# The following CLI call allows us to quickly check these probabilistic bounds
# for a given configuration (30 complex measurements)::
#
#   python -m skerch post_bounds --apost_n=30 --apost_err=0.5 --is_complex
#
# The following Python code is equivalent:

skerch_main(["post_bounds", "--apost_n=30", "--apost_err=0.5", "--is_complex"])

# %%
# This can be interpreted as follows: If we performed 30 test measurements and
# got an error estimate of :math:`\hat{\varepsilon}`, the probability
# of the *actual* error :math:`\varepsilon` being outside of the
# :math:`(0.5\hat{\varepsilon}, 1.5\hat{\varepsilon})` range is as provided.


# %%
#
# ##############################################################################
#
# Creating HDF5 layout for distributed sketches
# ---------------------------------------------
#
# `HDF5 <https://www.h5py.org/>`_ files allow to efficiently read and write
# large numerical arrays in an out-of-core, distributed fashion.
# This is useful to perform sketched decompositions of (very) large linear
# operators, since both storage and measurements can be distributed across
# different processes or machines (see :mod:`skerch.hdf5` and
# :ref:`Out-of-core Operations via HDF5` for details on how to work with
# these files using ``skerch``).
#
# The following ``skerch`` CLI call allows to conveniently create a HDF5
# layout to store sketched measurements from a linear operator of given
# ``lop_shape`` and ``dtype``::
#
#   python -m skerch create_hdf5_layout_lop --lop_shape=100,200 \
#          --dtype=complex128 --partsize=10 --lo=30 --ro=30 --inner=60
#
# Equivalent python code (up to use of ``tmpdir``):

tmpdir = TemporaryDirectory()
skerch_main(
    [
        "create_hdf5_layout_lop",
        f"--hdf5dir={tmpdir.name}",
        "--lop_shape=100,200",
        "--dtype=complex128",
        "--partsize=10",
        "--lo=30",
        "--ro=30",
        "--inner=60",
    ]
)


# %%
# Note that if ``hdf5dir`` is given, it must exist and be empty (if not
# given, a temporary directory will be suggested).
# The optional ``lo, ro, inner`` parameters determine whether layouts for
# (respectively) left-outer, right-outer and inner sketches will be created.
#
# Another important detail is that, in order to facilitate concurrent writing,
# the overall HDF5 layout is divided in smaller chunks, in what is known
# as a `HDF5 virtual dataset <https://docs.h5py.org/en/stable/vds.html>`_.
# In this example, each chunk contains ``partsize=10`` measurements, so we
# end up with 3 chunks for ``lo, ro`` and 6 for ``inner``.

# %%
#
# ##############################################################################
#
# Merging distributed HDF5 sketches
# ---------------------------------
#
# Although decentralized measurement and storage via HDF5 virtual datasets has
# many advantages, some operations may require to process the measurements in
# a centralized fashion. For instance, many operative systems do not allow a
# single process to keep thousands of files open at the same time. Also,
# many numerical routines may not feature an out-of-core, in-place
# implementation.
#
# The ``skerch`` solution is to merge all individual HDF5 chunks from the
# virtual dataset into a single, centralized HDF5 file of the same size.
# It will still have the same contents, but instead of being a collection of
# HDF5 files bundled into a virtual dataset, it will be a single, monolithic
# HDF5 file with contiguous data. The following command merges the previously
# created left-outer measurement layout::
#
#   python -m skerch merge_hdf5 --delete_subfiles --ok_flag=initialized \
#          --in_path /tmp/tmp4fswvvk2/leftouter_ALL.h5
#
# Equivalent python code (up to ``tmpdir``):

skerch_main(
    [
        "merge_hdf5",
        "--delete_subfiles",
        "--ok_flag=initialized",
        "--in_path",
        os.path.join(tmpdir.name, "leftouter_ALL.h5"),
    ]
)

tmpdir.cleanup()

# %%
# Note the following:
#
# * If the ``--delete_subfiles`` flag is provided, each "chunk" file
#   file will be deleted after being merged. This ensures disk usage remains
#   almost constant.
# * If ``--ok_flag`` is provided, the script will check that all HDF5 flags
#   match this value before proceeding. This can be used to ensure that all
#   distributed measurements have been performed before merging/deleting.
# * The ``--out_path`` flag can also be provided to set the location of the
#   merged HDF5 file. If none provided, the path of the former virtual dataset
#   is used (and it becomes an actual monolithic dataset instead of a bundle
#   of virtual references to the chunk files). In either case, this CLI
#   call returns the path of the merged dataset.
