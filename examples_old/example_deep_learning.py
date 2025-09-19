# -*- coding: utf-8 -*-

r"""Deep Learning
=================

As shown in e.g. :ref:`Sketched Decompositions`, this library computes sketched
decompositions with any linear operator that implements left- and right matrix
multiplication via the ``@`` operator, and the ``shape = (height, width)``
attribute.

The amazing `CurvLinOps library <https://github.com/f-dangel/curvlinops>`_
provides linear operators that satisfy this property (among many others), for a
variety of very useful DL-related linear operators, like the Hessian, the
Jacobian and the Generalized Gauss-Newton (GGN). It is also implemented
with PyTorch as a backend, but the ``CurvLinOps`` operators actually implement
the ``SciPy LinearOperator`` interface, and can be therefore used with most of
the LinAlg routines available in SciPy.

But, to the date, SciPy does not provide sketched eigendecompositions for
such operators (see `here <https://github.com/scipy/scipy/issues/16049>`_) and
classical algorithms have potentially much worse runtime and memory
requirements.

In this tutorial, we show how to approximate a GPU-accelerated Hessian
full eigendecomposition from a deep neural network, using ``skerch``'s very own
SEIGH together with CurvLinOps. Note that this procedure can easily scale to
much larger networks (with parameters up to the millions) and datasets (with
thousands of samples) than traditional methods.

We also show via *a posteriori* methods that the recovered eigendecomposition
is pretty close to the original, **full** Hessian. No more compromises!
"""

import matplotlib.pyplot as plt
import torch
from curvlinops import HessianLinearOperator

from skerch.a_posteriori import a_posteriori_error
from skerch.decompositions import seigh, truncate_core
from skerch.linops import CompositeLinOp, DiagonalLinOp, TorchLinOpWrapper

# %%
#
# ##############################################################################
#
# Setup
# -----
#
# Mirroring CurvLinOps tutorials, we will use synthetic data, consisting of two
# mini-batches, a small MLP, and mean-squared error as loss function. But we
# will use a larger scale :)

# GLOBALS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# dl globals
N = 20
D_in = 50
D_hidden = 100
D_out = 10
# skerch globals
OUTER, INNER, TRUNCATE = 500, 1000, 400
NUM_A_POSTERIORI = 20


# synthetic dataset
X1, y1 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)
X2, y2 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)
data = [(X1, y1), (X2, y2)]

# neural network
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(D_hidden, D_hidden),
    torch.nn.Sigmoid(),
    torch.nn.Linear(D_hidden, D_out),
).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
loss_function = torch.nn.MSELoss(reduction="mean").to(DEVICE)

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


# Create the Hessian LinOp, which is a function of model, loss and data
H = TorchHessianLinearOperator(model, loss_function, params, data)
print(H)

# Perform sketched Hermitian eigendecomposition!
Q, U, S = seigh(
    H,
    op_device=DEVICE,
    op_dtype=getattr(torch, str(H.dtype)),
    outer_dim=OUTER,
    inner_dim=INNER,
)

# Compose low-rank approximation
U, S = truncate_core(TRUNCATE, U, S)
appr_H = CompositeLinOp(
    (
        ("Q", Q),
        ("U", U),
        ("S", DiagonalLinOp(S)),
        ("Ut", U.T),
        ("Qt", Q.T),
    )
)


# Plot a fragment of the recovered Hessian
frag_n = 50
appr_H_fragment = torch.empty_like(Q[:frag_n, :frag_n])
buff = torch.zeros_like(Q[:, 0])
for i in range(frag_n):
    buff *= 0
    buff[-(i + 1)] = 1
    appr_H_fragment[:, -(i + 1)] = (appr_H @ buff)[-frag_n:]

plt.imshow(appr_H_fragment.cpu().numpy())
plt.show()

# %%
#
# ##############################################################################
#
# Error estimation
# ----------------
#
# Now all is left is to compute an *a posteriori* estimate of the Frobenius
# error between the original Hessian :math:`H` and our sketched, low-rank
# approximation :math:`\hat{H}`. The ``RELATIVE ERROR`` :math:`\rho` is given
# as the ratio
# :math:`\rho := \frac{\lVert H - \hat{H} \rVert_F}{\lVert H \rVert_F}`.

(f1, f2, frob_err) = a_posteriori_error(
    H,
    appr_H,
    NUM_A_POSTERIORI,
    dtype=getattr(torch, str(H.dtype)),
    device=DEVICE,
)[0]

print("Estimated Hessian Norm:", f1**0.5)
print("Estimated Frobenius Error:", frob_err**0.5)
print("RELATIVE ERROR:", (frob_err / f1) ** 0.5)
