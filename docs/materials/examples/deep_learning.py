# -*- coding: utf-8 -*-

r"""Deep Learning
=================

Recall that ``skerch`` operations admit any linear operator that implements
left- and right matrix multiplication via the ``@`` operator, and the
``.shape = (height, width)`` attribute.

The `CurvLinOps library <https://github.com/f-dangel/curvlinops>`_
provides curvature linear operators that satisfy this requirement, for a
variety of very useful objects such as the Hessian, the Jacobian and the
Generalized Gauss-Newton (GGN). It is also implemented with PyTorch as a
backend, but the ``CurvLinOps`` operators actually implement SciPy's
`LinearOperator <https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
interface, so they can also be used with most of the LinAlg routines available
in SciPy.

In this example, we show how to approximate a GPU-accelerated Hessian
full eigendecomposition from a deep neural network, using ``skerch``'s
sketched EIGH combined with CurvLinOps.

To verify the quality of approximation, we apply the *a-posteriori* test
method already discussed in :ref:`Sketched Low-Rank Decompositions`.

The result is a pretty good and quick approximation, at scales that would
otherwise be very slow/large, or even intractable. This procedure also
scales to much larger networks (with parameters up to the millions) and
datasets (with thousands of samples) see e.g.
`this paper <https://openreview.net/forum?id=yGGoOVpBVP>`_.
"""

import matplotlib.pyplot as plt
import torch
from curvlinops import HessianLinearOperator

from skerch.utils import gaussian_noise, rademacher_noise
from skerch.linops import CompositeLinOp, DiagonalLinOp, TorchLinOpWrapper
from skerch.algorithms import ssvd, seigh
from skerch.a_posteriori import apost_error


# %%
#
# ##############################################################################
#
# Setup
# -----
#
# Curvature matrices are a function of a dataset, model and loss function.
# For this example, we create a synthetic dataset and a model with >30000
# parameters, resulting in a Hessian of >1 billion entries.

SEED = 12345780
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
DATASET_SHAPE = (2, 50, 784)  # num_batches, batch_size, xdim
MLP_DIMS = (784, 50, 50, 10)
OUTER_MEAS, INNER_MEAS, TEST_MEAS = 6_000, 9_000, 100
MEAS_BLOCKSIZE = 5_000

# synthetic dataset, model and loss function
X = gaussian_noise(DATASET_SHAPE, seed=SEED, dtype=DTYPE, device=DEVICE)
Y = rademacher_noise(
    DATASET_SHAPE[:-1] + (MLP_DIMS[-1],), seed=SEED + 1, device=DEVICE
).to(DTYPE)
dataloader = list(zip(X, Y))
model = torch.nn.Sequential(
    *sum(
        [
            [torch.nn.Linear(i, o), torch.nn.ReLU()]
            for i, o in zip(MLP_DIMS[:-1], MLP_DIMS[1:])
        ],
        [],
    )[:-1]
).to(DEVICE)
loss_function = torch.nn.MSELoss(reduction="mean").to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
num_params = sum(p.numel() for p in params)

print(model)
print("Number of trainable parameters:", num_params)
print("Number of Hessian entries:", num_params**2)

# %%
#
# ##############################################################################
#
# Sketched Hessian Eigendecomposition
# -----------------------------------
#
# While ``CurvLinOps`` interfaces with ``SciPy/NumPy``, ``skerch`` does expect
# that the linear operators input and output ``PyTorch`` tensors. To overcome
# this discrepancy, ``skerch`` provides a
# :class:`skerch.linops.TorchLinOpWrapper` class. With this, ``seigh``
# can be directly applied to the wrapped ``CurvLinOps`` operator!


class TorchHessianLinearOperator(TorchLinOpWrapper, HessianLinearOperator):
    pass


# Now we can create the Hessian LinOp and perform the ``skerch``
# eigendecomposition. Some considerations:
# * If ``SKERCH_MEAS`` is too small, recovery quality may suffer drastically
# * Reducing ``meas_blocksize`` helps against out-of-memory errors, but
#   is slower (less parallel measurements at once)

H = TorchHessianLinearOperator(
    model, loss_function, params, dataloader, progressbar=True
)
print(H)

ews, evs = seigh(
    H,
    DEVICE,
    DTYPE,
    OUTER_MEAS,
    seed=SEED + 2,
    noise_type="rademacher",
    recovery_type=f"oversampled_{INNER_MEAS}",
    lstsq_rcond=1e-6,
    meas_blocksize=MEAS_BLOCKSIZE,
)
sH = CompositeLinOp((("Q", evs), ("Lbd", DiagonalLinOp(ews)), ("Qt", evs.T)))

# %%
#
# ##############################################################################
#
# Error estimation
# ----------------
#
# Looks good, but how good is our recovery? We now to estimate the error via
# a-posteriori test measurements, confirming it is very low:
(frob_sq, f_sq, err_sq), _ = apost_error(
    H, sH, DEVICE, DTYPE, num_meas=TEST_MEAS, seed=SEED + num_params
)

rel_err = (err_sq / frob_sq) ** 0.5
print("Estimated Hessian Norm:", frob_sq.item() ** 0.5)
print("Estimated Frobenius Error:", err_sq.item() ** 0.5)
print("RELATIVE ERROR:", (err_sq / frob_sq).item() ** 0.5)


# Finally plot recovered eigenvalues and a fragment of the recovered Hessian,
# showcasing the usual "tartan" pattern

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.plot(ews.cpu())
ax2.imshow(((evs[-200:] * ews) @ evs[-200:].T).abs().log().cpu())
fig.suptitle("Recovered Hessian eigenvalues and fragment of $log |H|$")
