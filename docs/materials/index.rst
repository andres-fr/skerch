:code:`skerch`
==================================

``skerch``: Sketched matrix decompositions for `PyTorch <https://pytorch.org/>`_.

Consider a matrix or linear operator :math:`A \in \mathbb{R}^{M \times N}`,
typically of intractable size and/or very costly measurement :math:`w = Av`,
but of low-rank structure. A typical example of this are
`kernel matrices for Gaussian Processes <https://arxiv.org/abs/2107.00243>`_
or the `Hessian matrices for Deep Learning <https://arxiv.org/abs/1706.04454>`_.

Furthermore, consider its `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_:

.. math::

  A := U \Sigma V^T


If :math:`A` has rank :math:`k`, sketched methods allow us to approximate it while requiring only in the order of:

* :math:`\mathcal{O}(\text{max}(M, N) \cdot k)` memory
* :math:`\mathcal{O}(\text{max}(M, N) \cdot k^2)` arithmetic
* :math:`\mathcal{O}(k)` *parallelizable* measurements

This is in stark contrast with traiditional methods (e.g. QR-based, orthogonal iterations, Arnoldi...), which entail more memory requirements, arithmetic overhead, sequential measurements and/or numerical instability (see `[HMT2009] <https://arxiv.org/abs/0909.4061>`_ for extensive discussion). With the help of sketched methods, **explicit representation and SVD of otherwise intractable operators becomes now practical at unprecedented scales**.

This package implements functionality to perform such sketched decompositions
for any arbitrary matrix or linear operator :math:`A`.
**This includes non-square, non-PSD, and matrix-free operators**. Furthermore,
operations are implemented using ``PyTorch``, which means that **CPU/CUDA can
be used in a device-agnostic way and automatic differentiation is available**.

It also implements:

* Efficient *a priori* methods to choose meaningful hyperparameters for the sketched algorithms
* Efficient *a posteriori* methods to estimate the quality and rank of the sketched approximations
* Matrix-free estimation of (sub-)diagonals for square linear operators
* Matrix-free estimation of matrix-vector products for upper- and lower-triangular portions of square linear operators.

.. seealso::

  The contents of this repository are based on the following publications:

  * `[HMT2009] <https://arxiv.org/abs/0909.4061>`_: Nathan Halko, Per-Gunnar
    Martinsson, Joel A. Tropp. 2011. *“Finding Structure with Randomness:
    Probabilistic Algorithms for Constructing Approximate Matrix Decompositions”*.
    SIAM Review, 53 (2): 217-288.

  * `[TYUC2019] <https://arxiv.org/abs/1902.08651>`_: Joel A. Tropp, Alp
    Yurtsever, Madeleine Udell, and Volkan Cevher. 2019. *“Streaming Low-rank
    Matrix Approximation with an Application to Scientific Simulation”*. SIAM
    Journal on Scientific Computing 41 (4): A2430–63.

  * `[BN2022] <https://arxiv.org/abs/2201.10684>`_: Robert A. Baston and Yuji
    Nakatsukasa. 2022. *“Stochastic diagonal estimation: probabilistic bounds and
    an improved algorithm”*.  CoRR abs/2201.10684.

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
