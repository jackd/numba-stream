from typing import Tuple

import numba as nb
import numpy as np

from . import ragged, utils

IntArray = np.ndarray


@nb.njit(inline="always")
def ravel_multi_index(indices, dims):
    """
    See `np.ravel_multi_index`.

    Note this version accepts negative indices without raising.
    """
    ndim = len(dims)
    assert len(indices) == ndim
    acc = indices[-1].copy()
    stride = dims[-1]
    for i in range(ndim - 2, -1, -1):
        acc += indices[i] * stride
        stride *= dims[i]
    return acc


@nb.njit(inline="always")
def unravel_index(indices: IntArray, shape, dtype=None) -> IntArray:
    """See `np.ravel_index`."""
    ndim = len(shape)
    out = np.empty(
        (ndim, indices.size), dtype=indices.dtype if dtype is None else dtype
    )
    indices = indices.copy()
    for n in range(ndim - 1, 0, -1):
        s = shape[n]
        out[n] = indices % s
        indices //= shape[n]
    out[0] = indices

    return out


@nb.njit(inline="always")
def unravel_index_transpose(indices: IntArray, shape, dtype=None) -> IntArray:
    """Equivalent to `unravel_index(indices.T, shape)."""
    ndim = len(shape)
    out = np.empty(
        (indices.size, ndim), dtype=indices.dtype if dtype is None else dtype
    )
    indices = indices.copy()
    for n in range(ndim - 1, 0, -1):
        s = shape[n]
        out[..., n] = indices % s
        indices //= s
    out[..., 0] = indices
    return out


@nb.njit(inline="always")
def ravel_multi_index_transpose(indices: IntArray, dims):
    """
    Equivalent to `ravel_multi_index(indices.T, dims)`.

    Args:
        indices: [n, ndim] ints
        dims: tuple of ints, shape of the nd array.

    Returns:
        [n] int array of 1D indices.
    """
    ndim = len(dims)
    assert indices.shape[-1] == ndim
    ndim = indices.shape[1]
    acc = indices[..., -1].copy()
    stride = dims[-1].item()
    for i in range(ndim - 2, -1, -1):
        acc += indices[..., i] * stride
        stride *= dims[i]
    return acc


@nb.njit()
def base_grid_coords(shape: IntArray):
    return unravel_index_transpose(np.arange(utils.prod(shape)), shape)


@nb.njit()
def grid_coords(
    in_shape: IntArray, kernel_shape: IntArray, strides: IntArray, padding: IntArray
) -> Tuple[IntArray, IntArray]:
    """
    Get the coordinates of the top left of each kernel region in input coords.

    Args:
        in_shape: [ndims] int
        kernel_shape: [ndims] int
        strides: [ndims] int
        padding: [ndims] int

    Returns:
        coords_nd: [out_size, ndims] ints
        out_shape: [ndims] ints
    """
    out_shape = (in_shape + 2 * padding - kernel_shape) // strides + 1
    coords_nd = base_grid_coords(out_shape) * strides - padding
    return coords_nd, out_shape


@nb.njit()
def sparse_neighborhood(
    in_shape: IntArray, kernel_shape: IntArray, strides: IntArray, padding: IntArray
) -> Tuple[IntArray, IntArray, IntArray, IntArray]:
    """
    Get information associated with a grid-based sparse neighborhood.

    Args:
        in_shape: [nd] input grid shape
        kernel_shape: [nd]
        strides: [nd]
        padding: [nd]

    Returns:
        partitions: [ne] indices of kernel elements.
        indices: [ne] indices into 1d output coordinates.
        row_splits: [nn+1] ragged row_splits associated with partitions/indices.
        out_shape: [nd] output shape.
    """
    ndim = len(in_shape)
    coords_nd, out_shape = grid_coords(in_shape, kernel_shape, strides, padding)
    offset = base_grid_coords(kernel_shape)
    coords_expanded = np.expand_dims(coords_nd, axis=-2) + offset
    flat_coords = np.reshape(coords_expanded, (-1, ndim))
    num_partitions = offset.shape[0]
    num_coords = flat_coords.shape[0]
    partitions = np.arange(num_coords)
    partitions %= num_partitions
    valid = np.ones((num_coords,), dtype=np.bool_)
    for i in range(ndim):
        ci = flat_coords[:, i]
        valid[ci < 0] = False
        valid[ci >= in_shape[i]] = False

    valid_coords = flat_coords[valid]
    valid_partitions = partitions[valid]
    coords_1d = ravel_multi_index_transpose(valid_coords, in_shape)
    valid = np.reshape(valid, (-1, offset.shape[0]))
    lengths = np.count_nonzero(valid, axis=1)
    splits = ragged.lengths_to_splits(lengths)
    return valid_partitions, coords_1d, splits, out_shape
