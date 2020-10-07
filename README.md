# numba-stream

Numba implementations of event stream preprocessing algorithms.

## Installation

```bash
git clone https://github.com/jackd/numba-stream.git
pip install -e numba-stream
```

For examples:

```bash
pip install git+git://github.com/jackd/events-tfds.git
python numba-stream/examples/lif.py
```

## Projects using numba-stream

- [Event Convolution Networks](https://github.com/jackd/ecn.git)

## Overview

### Grid Neighbors [grid.py](numba_stream/grid.py)

Provides [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))  connectedness information between grids as the result of a convolution-like operation (i.e. one with a kernel, stride and padding). For example, a 4x4 grid with kernel size and stride of 2 in each dimension and no padding as an output grid with shape 2x2. Output (0, 0) is neighbors with inputs [(0, 0), (0, 1), (1, 0), (1, 1)], i.e.

```python
neighbors[(0, 0)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
```

`grid.sparse_neighborhood` gives the ravelled indices, i.e.

```python
neighbors_ravelled[0] == [0, 1, 4, 5]
```

In this example, all elements of the output grid have the same number of neighbors, though this is not always the same - e.g. if the input was 5x5 and we added some padding, those padded neighbors would never have associated events, so we wouldn't need to consider them. For this reason we use a `ragged` representation, adopting terminology from [tensorflow](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor).

### Leaky Integrate and Fire [lif.py](numba_stream/lif.py)

We provide two leaky integrate and fire implementations:

- `leaky_integrate_and_fire`: base implementation without spatial differentiation. This can be thought of as the 0-d implementation.
- `spatial_leaky_integrate_and_fire`: implementation that can take arbitrary connectedness. Coordinates must be 1D, e.g. the transpose of those provided by `grid.py`. If your coordinates / spatial connectedness is multi-dimensional, use `grid.ravel_multi_index[_transpose]`.

### Event Neighbors [neighbors.py](numba_stream/neighbors.py)

An event `e0 == (x0, t0)` from stream `E0` is said to be in the event neighborhood of event `e1 == (x1, t1)` with duration `tau` from stream `E1` if `x0` is in the spatial neighborhood of `x1` and `t1 - t0 < tau`. Note this is not commutative, i.e. if `e0` is in the neighborhood of `e1` it is not necessarily the case that `e1` is in the neighborhood of `e0`. In other words, the event graph between two streams induced by this definition is directed.

[neighbors.py](numba_stream/neighbors.py) provides tools for computing connectedness between two streams.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
