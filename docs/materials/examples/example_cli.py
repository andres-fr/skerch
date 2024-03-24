# -*- coding: utf-8 -*-

r"""Command Line Interface
==========================

This example demonstrates the functionality in ``skerch`` that can be directly
accessed from the command line interface (CLI).

We simply assume that ``skerch`` is accessible to the available ``python``
interpreter.

The following imports are only needed to exemplify the Python code:
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
# The following Python code is equivalent:


skerch_main(["-h"])


# %%
#
# ##############################################################################
#
# A-priori hyperparameter estimation
# ----------------------------------
#
# Sketched decompositions have hyperparameters for the number of inner and outer
# measurements to be performed. A natural question is, given a limited amount
# of memory (provided in number of matrix entries), how many inner and outer
# measurements should we perform to get the best results?
#
# For asymmetric matrices, this can be quickly checked as follows::
#
#   python -m skerch prio_hpars --shape=100,200 --budget=12345
#
# The following Python code is equivalent:

skerch_main(["prio_hpars", "--shape=100,200", "--budget=1234"])

# %%
# The ``[CHECK]`` field shows that the memory requirements are optimally
# satisfied for the given solution and constraints (see respective
# documentation for more details)

# %%
#
# ##############################################################################
#
# A-posteriori error bounds
# -------------------------
#
# Once a linear operator is approximated via sketched methods, it is possible
# to efficiently estimate how good of an approximation it is (see respective
# documentation for more details).
#
# The quality of this estimation depends only on the number of *a-posteriori*
# measurements performed and it can be quickly checked as follows::
#
#   python -m skerch post_bounds --apost_n=30 --apost_err=0.5
#
# The following Python code is equivalent:

skerch_main(["post_bounds", "--apost_n=30", "--apost_err=0.75"])

# %%
# We see in this example that, if we performed 30 measurements and we got an
# error estimate of :math:`\hat{\varepsilon}`, the probability of the *actual*
# error :math:`\varepsilon` being outside of the
# :math:`(0.25\hat{\varepsilon}, 1.75\hat{\varepsilon})` range is provided.


# %%
#
# ##############################################################################
#
# Creating HDF5 layout for distributed sketches
# ---------------------------------------------
#
# HDF5 files allow to efficiently read and write large numerical arrays in
# an out-of-core, distributed fashion. This is useful to perform sketched
# decompositions of (very) large linear operators, since both storage
# and measurements can be distributed (see respective documentation for details
# on how to work with these files using ``skerch``.
#
# A HDF5 layout can be conveniently created as follows::
#
#   python -m skerch create_hdf5_layout --hdf5dir=/tmp/test_skerch \
#          --dtype=float64 --shape=123,234 --outer=30 --inner=60
#
# The following Python code is equivalent (up to the use of ``tmpdir``):

tmpdir = TemporaryDirectory()
skerch_main(
    [
        "create_hdf5_layout",
        f"--hdf5dir={tmpdir.name}",
        "--dtype=float64",
        "--shape=123,234",
        "--outer=30",
        "--inner=60",
    ]
)

# %%
# The script prints the paths and main properties of the generated HDF5 files,
# which comprise several
# `HDF5 virtual datasets <https://docs.h5py.org/en/stable/vds.html>`_ which act
# as layouts to hold the left-outer, right-outer and inner measurements.
# In this example, a sketched decomposition for a ``torch.float64`` operator
# of height 123 and width 234 is being prepared, comprising 30 outer and 60
# inner measurements (which can be good if e.g. the effective rank of the
# operator is around 10).
#
# If the ``--sym`` flag is provided, ``rightouter`` files won't be created.


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
# single process to keep thousands of files open at the same time.
#
# A solution is to merge all individual HDF5 files from the virtual dataset
# into a single, centralized HDF5 file of the same size. The following command
# merges the previously created left-outer measurement layout::
#
#   python -m skerch merge_hdf5 --delete_subfiles --ok_flag=initialized \
#          --in_path /tmp/tmp4fswvvk2/leftouter_ALL.h5
#
# The following Python code is equivalent (up to the use of ``tmpdir``):

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
# * If the ``--delete_subfiles`` flag is provided, each individual distributed
#   HDF5 file will be deleted after being merged.
# * If ``--ok_flag`` is provided, the script will check that all HDF5 flags
#   match this value before proceeding. This can be used to ensure that all
#   distributed measurements have been performed before merging/deleting.
# * The ``--out_path`` flag can also be provided to set the location of the
#   merged HDF5 file. If none provided, the path of the former virtual dataset
#   is used. In any case the path of the merged dataset is returned.
