# -*- coding: utf-8 -*-

r"""Out-of-core Operations via HDF5
===================================

In-core operations are generally faster and convenient, but they have
limited scalability: if we distribute our operations across several machines,
we can make use of more memory and parallelize computations.
This is particularly relevant for sketched methods, in the cases where
linear operators have intractable sizes and/or linop evaluations take a
long time.

`HDF5 <https://www.h5py.org/>`_ is a popular way to store numerical data
persistently. From Python, it looks mostly like a NumPy array, but it is
stored in disk, and it can be partitioned across multiple sub-files in the
filesystem. This allows us to work with very large arrays while satisfying
our memory constraints. Also, the different sub-files can be processed
by different machines independently, with the resulting speedup.

In this example we illustrate the functionality provided in :mod:`skerch.hdf5`
in order to facilitate out-of-core operations. We first create a distributed
HDF5 numerical array, and then simulate multiple independent processes to
populate it with data. Finally, we test its correctness. Note that ``skerch``
also privdes access to some of the HDF5 functionality via CLI, see
:ref:`Command Line Interface`.
"""

import os
import tempfile
import matplotlib.pyplot as plt
import torch

from skerch.utils import torch_dtype_as_str
from skerch.measurements import GaussianNoiseLinOp
from skerch.hdf5 import DistributedHDF5Tensor


# %%
#
# ##############################################################################
#
# Setup
# -----
#
# We start creating a matrix-free linear operator, which could be of very
# large dimensionality and hence require distributed memory and/or
# computation.

SEED = 9876531
SHAPE = (1000, 2000)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.complex128
BLOCKSIZE = 100

mop = GaussianNoiseLinOp(SHAPE, SEED, blocksize=BLOCKSIZE, by_row=True)


# %%
#
# Now we create a distributed HDF5 database, and write chunks of the linear
# operator onto the respective HDF5 chunks. Since the chunks are indidivual
# files and the linear operator is seed-reproducible, this loop
# can be distributed across independent process and machines:

tmpdir = tempfile.TemporaryDirectory()
h5_pth, h5_subpaths, h5_begs_ends = DistributedHDF5Tensor.create(
    os.path.join(tmpdir.name, mop.__class__.__name__ + "_{}"),
    SHAPE,
    BLOCKSIZE,
    torch_dtype_as_str(DTYPE),
)
h5_map = dict(zip(h5_begs_ends, h5_subpaths))

for block, idxs in mop.get_blocks(DTYPE, DEVICE):
    # each one of these iterations could be in a parallel process/machine
    subpath = h5_map[(idxs.start, idxs.stop)]
    data, flags, h5 = DistributedHDF5Tensor.load(subpath)
    data[:] = block.cpu()
    flags[:] = "OK"
    h5.close()


# %%
#
# Finally, we test our HDF5 database, verifying that all flags have been
# set to OK and the contents match our linear operator.
# The exact moment in which we load the data from disk to memory is
# when calling ``data[:]``. The ``data`` reference is just a pointer to the
# filesystem and can be used to efficiently access portions of the array
# via ``data[idxs...]``, without having to load the whole array at once.

data, flags, h5 = DistributedHDF5Tensor.load(h5_pth)
is_ok = bool((flags.asstr()[:] == "OK").all())
same_data = bool((data[:] == mop.to_matrix(DTYPE, "cpu")).all())

print("All flags set to OK:", is_ok)
print("HDF5 data matches linear operator:", same_data)
