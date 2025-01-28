<p align="center">
    <img alt="Skerch logo, light mode" src="docs/materials/assets/skerch_horizontal.svg#gh-light-mode-only" width="50%"/>
    <img alt="Skerch logo, dark mode" src="docs/materials/assets/skerch_horizontal_inv.svg#gh-dark-mode-only" width="50%"/>
</p>


<h3 align="center">
<code>skerch</code>: Sketched matrix decompositions for PyTorch
</h3>


<div align="center">

|                                                             PyPI                                                                 |                                                                             Docs                                                                             |                                                                                                  CI                                                                                                  |                                                                                    Tests                                                                                    |
|:--------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![PyPI - Downloads](https://img.shields.io/pypi/dm/skerch?style=flat&label=skerch)](https://pypi.org/project/skerch/) | [![Documentation Status](https://readthedocs.org/projects/skerch/badge/?version=latest)](https://skerch.readthedocs.io/en/latest/?badge=latest) | [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/andres-fr/skerch/ci.yaml)](https://github.com/andres-fr/skerch/actions) | [![Coverage Status](https://coveralls.io/repos/github/andres-fr/skerch/badge.svg?branch=main)](https://coveralls.io/github/andres-fr/skerch?branch=main) |

</div>


`skerch` is a Python package to compute different decompositions (SVD, Hermitian Eigendecomposition, diagonal, subdiagonal, triangular, block-triangular) of linear operators via sketched methods.

* Built on top of PyTorch, with natural support for CPU and CUDA interoperability, and very few dependencies otherwise
* Works on matrices and matrix-free operators of potentially very large dimensionality
* Support for sketched measurements in a fully distributed fashion via [HDF5 databases](https://www.h5py.org/)


References:

* [Streaming Low-Rank Matrix Approximation with an Application to Scientific Simulation](https://arxiv.org/abs/1902.08651) Joel A. Tropp, Alp Yurtsever, Madeleine Udell, and Volkan Cevher. 2019. SIAM Journal on Scientific Computing 41 (4): A2430–63.
* [Stochastic diagonal estimation: probabilistic bounds and an improved algorithm](https://arxiv.org/abs/2201.10684) Robert A. Baston and Yuji Nakatsukasa. 2022. CoRR abs/2201.10684.


See the [documentation](https://skerch.readthedocs.io/en/latest/index.html) for more details, including examples for other decompositions and use cases.


# Installation and basic usage

Install via:

```bash
pip install skerch
```

The sketched SVD of a linear operator `op` of shape `(h, w)` can be then computed simply via:


```python
from skerch.decompositions import ssvd

q, u, s, vt, pt = ssvd(
    op,
    op_device=DEVICE,
    op_dtype=DTYPE,
    outer_dim=NUM_OUTER,
    inner_dim=NUM_INNER,
)
```

Where the number of outer and inner measurements for the sketch is specified, and `q @ u @ diag(s) @ vt @ pt` is a PyTorch matrix that approximates `op`, where `q, p` are *thin* orthonormal matrices of shape `(h, NUM_OUTER)` and `(NUM_OUTER, w)` respectively, and `u, vt` are *small* orthogonal matrices of shape `(NUM_OUTER, NUM_OUTER)`.

The `op` object must simply satify the following criteria:

* It must have a `op.shape = (height, width)` attribute
* It must implement the `w = op @ v` right-matmul operator, receiving and returning PyTorch vectors/matrices
* It must implement the `w = v @ op` left-matmul operator, receiving and returning PyTorch vectors/matrices

`skerch` provides a convenience PyTorch wrapper for the cases where `op` interacts with NumPy arrays instead (e.g. [SciPy linear operators](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) like the ones used in [CurvLinOps](https://github.com/f-dangel/curvlinops)).

To get a good suggestion of the number of measurements required for a given shape and budget, simply run:

```bash
python -m skerch prio_hpars --shape=100,200 --budget=12345
```

The library also implements cheap *a-posteriori* methods to estimate the error of the obtained sketched approximation:


```python
from skerch.a_posteriori import a_posteriori_error
from skerch.linops import CompositeLinOp, DiagonalLinOp

# (q, u, s, vt, pt) previously computed via ssvd
sketched_op = CompositeLinOp(
    (
        ("Q", q),
        ("U", u),
        ("S", DiagonalLinOp(s)),
        ("Vt", vt),
        ("Pt", pt),
    )
)

(f1, f2, frob_err) = a_posteriori_error(
    op, sketched_op, NUM_A_POSTERIORI, dtype=DTYPE, device=DEVICE
)[0]
print("Estimated Frob(op):", f1**0.5)
print("Estimated Frob(sketched_op):", f2**0.5)
print("Estimated Frobenius Error:", frob_err**0.5)
```

For a given `NUM_A_POSTERIORI` measurements (30 is generally OK), the probability of `frob_err**0.5` being wrong by a certain amount can be queried as follows:

```bash
python -m skerch post_bounds --apost_n=30 --apost_err=0.5
```

See [Getting Started](https://skerch.readthedocs.io/en/latest/getting_started.html), [Examples](https://skerch.readthedocs.io/en/latest/examples/index.html), and [API docs](https://skerch.readthedocs.io/en/latest/skerch.html) for more details.

# Developers

Contributions are most welcome under this repo's [LICENSE](LICENSE).
Feel free to open an [issue](https://github.com/andres-fr/skerch/issues) with bug reports, feature requests, etc.

The documentation contains a [For Developers](https://skerch.readthedocs.io/en/latest/for_developers.html) section with useful guidelines to interact with this repo.


# Researchers

If this library is useful for your work, please consider citing it:

```
@manual{fernandez2024skerch,
  title={{S}kerch: Sketched matrix decompositions for {PyTorch}},
  author={Andres Fernandez},
  year={2024},
  url={https://github.com/andres-fr/skerch},
}
```
