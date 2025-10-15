## 1.0.1 (2025-10-15)

## 1.0.0 (2025-10-15)

## 0.12.0 (2025-10-15)

### Feat

- major release, full refactoring, renaming and addition of new functionality, not backwards compatible.

Main features:
* Better test coverage -> less bugs
* Clearer and more comprehensive docs
* support for complex datatypes
* Support for (approximately) low-rank plus diagonal synthetic matrices
* Linop API:
  - New core functionality: Transposed, Signed Sum, Banded, ByBlock
  - Support for parallelization of matrix-matrix products
  - New measurement noise linops: Rademacher, Gaussian, Phase, SSRFT
* Data API:
  - Batched support for arbitrary tensors in distributed HDF5 arrays
  - Modular and extendible HDF5 layouts
* Sketching API:
  - Modular measurement API supporting multiprocessing and HDF5
  - Modular recovery methods (hmt, singlepass, Nystrom, oversampled)
* Algorithm API:
  - Algorithms: Hutch++ XDiag, SSVD, Triangular, SNorm (also Hermitian)
  - Modular and extendible design for noise sources and recovery types
  - Matrix-free a-posteriori error verification and rank estimation

## 0.11.0 (2025-04-25)

### Feat

- noOp commit to force commitizen to do a minor bump

## 0.10.1 (2025-03-17)

### Fix

- updated method name

## 0.10.0 (2025-03-17)

### Feat

- Finished unit tests for added linops. All tests and precommits passing, ready to merge
- Added NoiseLinOp, SumLinOp, BandedLinOp. With docstrings

## 0.9.1 (2025-03-16)

### Fix

- Upgraded sigstore action, was also crashing CI

## 0.9.0 (2025-03-16)

### Feat

- Upgraded version of GH actions. Obsolete version was crashing CI

## 0.8.1 (2025-03-16)

### Fix

- Also updated readthedocs builder to 3.10 (doesn't affect code)

## 0.8.0 (2025-03-16)

### Feat

- Upgraded version requirements to python 3.10 and curvlinops >=2
- Implemented (sub-)diagonal and (block-)triangular sketched estimators (#1)

## 0.7.0 (2024-05-08)

### Feat

- Implemented (sub-)diagonal and (block-)triangular sketched estimators (#1)

## 0.6.1 (2024-03-25)

### Fix

- updated CI pipeline

## 0.6.0 (2024-03-24)

### Feat

- updated version in changelog and pyproject for consistent migration
- ready for beta release

## 0.5.0 (2024-03-24)

### Feat

- migrated repo from private
