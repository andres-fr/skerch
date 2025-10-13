:code:`skerch`
==================================

**Sketched linear operations for** `PyTorch <https://pytorch.org/>`_.

Consider a matrix or linear operator :math:`A \in \mathbb{C}^{M \times N}`,
typically of intractable size and/or very costly measurements :math:`v \to Av`.

In many cases, such large operators feature a much smaller but hidden
sub-structure (such as low-rank or banded), which allows for an
approximation :math:`\hat{A}` of scalable size.
Typical examples of this are kernel matrices for large datasets, Hessian
matrices for deep learning, large-scale datasets and the throughput
of high-resolution simulations.

But obtaining :math:`\hat{A}` through traditional compression methods is
not feasible, since we would need to fully store or scan :math:`\hat{A}`
first. Instead, we *directly obtain* :math:`\hat{A}` from just
a few random :math:`y_i = A v_i` measurements, or *sketches*
(i.e. :math:`v_i` follows some random distribution). Luckily, this is
possible for a variety of :math:`\hat{A}` structures, and the
:math:`A v_i` measurements are typically parallelizable, allowing us
to work at large scales.

From an operational point of view, sketched methods only require the
ability to draw a few matrix-vector measurements in the form
:math:`Av, vA`. In Python, and for finite dimensions, this means
providing an ``A.shape`` attribute and implementing the
*matrix-multiplication* ``@`` operation.

One core advantage of ``skerch`` is that this is the *only* requirement
that :math:`A` needs to fulfill (unlike other libraries which require
``A`` to implement more attributes and/or operations). In code, we just
need to ensure that ``A`` satisfies the following interface:

.. code-block:: python

   class MyLinOp:
    def __init__(self, shape):
        self.shape = shape

    def __matmul__(self, x):
        return "... implement A @ x ..."

    def __rmatmul__(self, x):
        return "... implement x @ A ..."

Any operator implementing this interface will run on ``skerch`` routines
such as diagonalizations, operator norms and triangular approximations.
Other advantages of ``skerch``:

* Built on top of PyTorch, naturally supports CPU and CUDA, as well as complex datatypes. Very few dependencies otherwise
* Rich API for matrix-free linear operators, including matrix-free noise sources (Rademacher, Gaussian, SSRFT...)
* Efficient parallelized and distributed computations
* Support for out-of-core operations via HDF5
* A-posteriori verification tools to test accuracy of sketched approximations modular and extendible design, for easy adaption to new settings and operations
* Modular and extendible design

See the API docs and examples for illustrations of the above points.


.. seealso::

  * `[HMT2009] <https://arxiv.org/abs/0909.4061>`_: Nathan Halko, Per-Gunnar
    Martinsson, Joel A. Tropp. 2011. *“Finding Structure with Randomness:
    Probabilistic Algorithms for Constructing Approximate Matrix Decompositions”*.
    SIAM Review, 53 (2): 217-288.

  * `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_: Joel A. Tropp, Alp
    Yurtsever, Madeleine Udell, Volkan Cevher. 2019. *“Streaming Low-rank
    Matrix Approximation with an Application to Scientific Simulation”*. SIAM
    Journal on Scientific Computing 41 (4): A2430–63.

  * `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston, Yuji
    Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds and
    an improved algorithm”*.  CoRR abs/2201.10684.

  * `[ETW2024] <https://arxiv.org/pdf/2301.07825>`_: Ethan N. Epperly,
    Joel A. Tropp, Robert J. Webber. 2024. *“XTrace: Making the most of every
    sample in stochastic trace estimation”*. SIAM Journal on Matrix Analysis
    and Applications.

  * `[FSMH2025] <https://openreview.net/forum?id=yGGoOVpBVP>`_ Andres Fernandez,
    Frank Schneider, Maren Mahsereci, Philipp Hennig. 2025. *“Connecting Parameter
    Magnitudes and Hessian Eigenspaces at Scale using Sketched Methods”*.
    Transactions on Machine Learning Research.

  * `[DEOFTK2025] <https://arxiv.org/abs/2501.19183>`_ Felix Dangel, Runa Eschenhagen,
    Weronika Ormaniec, Andres Fernandez, Lukas Tatzel, Agustinus Kristiadi. 2025.
    *“Position: Curvature Matrices Should Be Democratized via Linear Operators”*.
    arXiv 2501.19183.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API

   skerch

.. toctree::
   :maxdepth: 2
   :caption: For Developers

   for_developers

..
   .. toctree::
      :maxdepth: 2
      :caption: Indices

      modindex
      genindex
