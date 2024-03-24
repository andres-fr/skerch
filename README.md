<p align="center">
    <img alt="Skerch logo, light mode" src="docs/materials/assets/skerch_horizontal.svg#gh-light-mode-only" width="50%"/>
    <img alt="Skerch logo, dark mode" src="docs/materials/assets/skerch_horizontal_inv.svg#gh-dark-mode-only" width="50%"/>
</p>


<h3 align="center">
<code>skerch</code>: Sketched matrix decompositions for PyTorch
</h3>


<div align="center">

|                                                             PyPI Installation                                                                 |                                                                             Documentation                                                                             |                                                                                                  CI                                                                                                  |                                                                                    Coverage                                                                                    |
|:--------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![PyPI - Downloads](https://img.shields.io/pypi/dm/skerch?style=flat&label=skerch)](https://pypi.org/project/skerch/) | [![Documentation Status](https://readthedocs.org/projects/skerch/badge/?version=latest)](https://skerch.readthedocs.io/en/latest/?badge=latest) | [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/andres-fr/skerch/ci.yaml)](https://github.com/andres-fr/skerch/actions) | [![Coverage Status](https://coveralls.io/repos/github/andres-fr/skerch/badge.svg?branch=main)](https://coveralls.io/github/andres-fr/skerch?branch=main) |

</div>


`skerch` is a Python package to compute diagonal decompositions (SVD, Hermitian Eigendecomposition) of linear operators via sketched methods.

* Built on top of PyTorch, with natural support for CPU and CUDA interoperability, and very few dependencies otherwise
* Works on matrices and matrix-free operators of potentially very large dimensionality
* Support for sketched measurements in a fully distributed fashion via HDF5 databases


See the [documentation](https://skerch.readthedocs.io/en/latest/index.html) for more details.


# Installation and basic usage

Install via:

```bash
pip install skerch
```

The sketched SVD of a linear operator `op` can be then computed simply via:


```python
q, u, s, vt, pt = ssvd(
    op,
    op_device=DEVICE,
    op_dtype=DTYPE,
    outer_dim=NUM_OUTER,
    inner_dim=NUM_INNER,
)
```

Where `q @ u @ diag(s) @ vt @ pt` approximates `linop` and the number of outer and inner measurements for the sketch is specified.

See [Getting Started](https://skerch.readthedocs.io/en/latest/getting_started.html), [Examples](https://skerch.readthedocs.io/en/latest/examples/index.html), and [API docs](https://skerch.readthedocs.io/en/latest/skerch.html) for more details.

# Developers

Contributions are most welcome under this repo's [LICENSE](LICENSE).
Feel free to open an [issue](https://github.com/andres-fr/skerch/issues) with bug reports, features requests, etc.

The documentation contains a [For Developers](https://skerch.readthedocs.io/en/latest/for_developers.html) section with useful guidelines to interact with this repo.
