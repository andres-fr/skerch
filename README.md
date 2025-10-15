<p align="center">
    <img alt="Skerch logo, light mode" src="docs/materials/assets/skerch_horizontal.svg#gh-light-mode-only" width="50%"/>
    <img alt="Skerch logo, dark mode" src="docs/materials/assets/skerch_horizontal_inv.svg#gh-dark-mode-only" width="50%"/>
</p>


<h3 align="center">
<code>skerch</code>: Sketched linear operations for PyTorch
</h3>


<div align="center">

|                                                             PyPI                                                                 |                                                                             Docs                                                                             |                                                                                                  CI                                                                                                  |                                                                                    Tests                                                                                    |
|:--------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![PyPI - Downloads](https://static.pepy.tech/badge/skerch/month)](https://pypi.org/project/skerch/) | [![Documentation Status](https://readthedocs.org/projects/skerch/badge/?version=latest)](https://skerch.readthedocs.io/en/latest/?badge=latest) | [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/andres-fr/skerch/ci.yaml)](https://github.com/andres-fr/skerch/actions) | [![Coverage Status](https://coveralls.io/repos/github/andres-fr/skerch/badge.svg?branch=main)](https://coveralls.io/github/andres-fr/skerch?branch=main) |

</div>


# What is `skerch`?

`skerch` is a Python package to compute different sketched linear operations, such as SVD/EIGH, diagonal/triangular approximations and operator norms. See the [documentation](https://skerch.readthedocs.io/en/latest/index.html) for more details and usage examples. Main features:

* Built on top of PyTorch, naturally supports CPU and CUDA, as well as complex datatypes. Very few dependencies otherwise
* Rich API for matrix-free linear operators, including matrix-free noise sources (Rademacher, Gaussian, SSRFT...)
* Efficient parallelized and distributed computations
* Support for out-of-core operations via [HDF5](https://www.h5py.org/)
* A-posteriori verification tools to test accuracy of sketched approximations
* modular and extendible design, for easy adaption to new settings and operations

# Why sketches?

Sketched methods are a good choice whenever we are dealing with *large* objects that can be approximated by *smaller* substructures (e.g. a low-rank approximation of a large matrix).
Thanks to the random measurements (i.e. the "sketches"), we can directly obtain the *small* approximations, without having to store or compute the *large* object. This works with very few assumptions about how the smaller substructure looks like.

For example, if we have a large linear operator of dimensionality `(N, N)` that doesn't fit in memory, but has rank `k`, we can directly retrieve its top-`k` singular components with only `O(Nk)` storage, as opposed to the intractable `O(N^2)` (see picture below for an intuition). As a bonus, this technique is numerically stable and can be parallelized, which often results in substantial speedups.

<p align="center">
  <img alt="Intuition for low-rank sketch-and-solve." src="docs/materials/assets/sketch_and_solve.png" width="15%"/>
</p>

# Why `skerch`?

On paper, sketched methods only require our linear operators to satisfy the following *bare bones interface*:

```
class MyLinOp:
    def __init__(self, shape):
        self.shape = shape

    def __matmul__(self, x):
        return "... implement A @ x ..."

    def __rmatmul__(self, x):
        return "... implement x @ A ..."
```

Anything more than this is not really required. In most cases, libraries do require more complicated interfaces, and this limits the application scenarios, or introduces substantial overhead to developers.

`skerch` is specifically designed to work on this bare-bones interface. Furthermore, its highly modular architecture allows users to exchange and modify different components of the sketched methods.

As a bonus, `skerch` is built on top of PyTorch, and with very few dependencies otherwise, so it supports a broad variety of platforms and datatypes (including e.g. complex datatypes on GPU).
`skerch` also supports in-core and out-of-core parallelizations (e.g. via [HDF5](https://www.h5py.org/) tensor databases), providing good scalability in memory and runtime. The documentation [examples](https://skerch.readthedocs.io/en/latest/examples/index.html) illustrate all of the above points.

In summary, `skerch` brings sketched methods to you with minimal overhead, and retaining good performance, resulting in overall faster development times. Give it a try!


# Installation and basic usage

Install via:

```bash
pip install skerch
```

Then, given any linear operator `lop` that implements the bare-minimum interface `.shape = (h, w)` and `lop @ x, x @ lop`, we can compute the sketched SVD as follows (``skerch`` also provides functionality to estimate EIGH, norm, diagonals...):


```python
from skerch.algorithms import ssvd

U, S, Vh = ssvd(lop, device, dtype, num_outer, seed=12345, recovery_type="nystrom")
```

With the `nystrom` recovery, this method requires a total of `2 * num_outer` measurements, and yields a *thin* SVD estimation where `lop` is approximated by `(U * S) @ Vh`, and  `U.shape = (h, num_outer)`, `S.shape = (num_outer,)`, `Vh.shape = (num_outer, w)`.

If `num_outer` is close enough to covering the rank of `lop`, this yields an accurate recovery (see documentation [examples](https://skerch.readthedocs.io/en/latest/examples/index.html)). But how can we make sure?

The library also implements cheap *a-posteriori* methods to estimate the error of the obtained sketched approximation, without requiring to know `lop`:


```python
from skerch.a_posteriori import apost_error, apost_error_bounds
from skerch.linops import CompositeLinOp, DiagonalLinOp

lop_approx = CompositeLinOp([("U", U), ("S", DiagonalLinOp(S)), ("Vh", Vh)])

(lop_f2, approx_f2, err2), _ = apost_error(lop, lop_approx, device, dtype, num_meas=30, seed=54321)

print("Estimated Frob(op):", lop_f2**0.5)
print("Estimated Frob(sketched_op):", approx_f2**0.5)
print("Estimated Frobenius Error:", err2**0.5)
```


This technique makes use of a number of test measurements that must be independent from `lop` and the sketch measurements (make sure to use a different seed). For 30 measurements and complex-valued `lop`, the probability of `err2**0.5` being wrong by at most 50% can be queried as follows:

```bash
python -m skerch post_bounds --apost_n=30 --apost_err=0.5 --is_complex
# returns {'LOWER: P(err<=0.5x)': 0.0030445096757934554, 'HIGHER: P(err>=1.5x)': 0.05865709397802224}
```


See the documentation [examples](https://skerch.readthedocs.io/en/latest/examples/index.html) and [API docs](https://skerch.readthedocs.io/en/latest/skerch.html) for more details.

# Developers

Contributions are welcome under this repo's [LICENSE](LICENSE).

Feel free to open an [issue](https://github.com/andres-fr/skerch/issues) with bug reports, feature requests, etc.

The documentation also contains a [For Developers](https://skerch.readthedocs.io/en/latest/for_developers.html) section with useful guidelines to interact with this repo and propose pull requests.


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
