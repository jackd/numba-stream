from typing import Optional, Tuple

import numba as nb
import numpy as np

from .utils import PARALLEL

IntArray = np.ndarray
BoolArray = np.ndarray


@nb.njit(inline="always")
def lengths_to_ids(row_lengths: IntArray) -> IntArray:
    """Row lengths to row ids."""
    return np.repeat(np.arange(row_lengths.size), row_lengths)


@nb.njit(inline="always")
def ids_to_lengths(ids: IntArray) -> IntArray:
    """Row ids to row lengths. ids are assumed to be sorted."""
    return np.bincount(ids)


@nb.njit(inline="always")
def splits_to_lengths(row_splits: IntArray) -> IntArray:
    """Row splits to row lengths."""
    return row_splits[1:] - row_splits[:-1]


@nb.njit(inline="always")
def splits_to_ids(row_splits: IntArray) -> IntArray:
    """Row splits to row ids."""
    return lengths_to_ids(splits_to_lengths(row_splits))


@nb.njit(inline="always")
def ids_to_splits(ids: IntArray) -> IntArray:
    """Row ids to row splits."""
    return lengths_to_splits(ids_to_lengths(ids))


@nb.njit(inline="always")
def lengths_to_splits(row_lengths: IntArray) -> IntArray:
    """Row lengths to row splits."""
    out = np.empty((row_lengths.size + 1,), dtype=row_lengths.dtype)
    # np.cumsum(row_lengths, out=out[1:])  # not supported yet
    out[1:] = np.cumsum(row_lengths)
    out[0] = 0
    return out


@nb.njit(inline="always", parallel=PARALLEL)
def row_sum(values: IntArray, splits: IntArray, out=None) -> IntArray:
    """Get the sum along the ragged dimension of the ragged components."""
    nrows = splits.size - 1
    if out is None:
        out = np.zeros((nrows,), dtype=values.dtype)
    # for i in range(nrows):
    for i in nb.prange(nrows):  # pylint: disable=not-an-iterable
        out[i] = values[splits[i] : splits[i + 1]].sum()
    return out


@nb.njit(inline="always")
def row_sorted(row_indices: IntArray, col_indices: IntArray, values: IntArray):
    """coo-indices sorted in row order."""
    order = np.argsort(row_indices)
    return row_indices[order], col_indices[order], values[order]


@nb.njit(inline="always")
def row(values, splits, row_index) -> IntArray:
    """Get the row from the given ragged components."""
    return values[splits[row_index] : splits[row_index + 1]]


# @nb.njit(inline='always')
# def rows(values, splits):
#     s0 = splits[0]
#     for s in splits[1:]:
#         yield values[s0:s]
#         s0 = s


@nb.njit(inline="always", parallel=PARALLEL)
def col_sort(values, splits) -> None:
    """Sort each row in-place by value."""
    # for i in range(splits.size - 1):
    for i in nb.prange(splits.size - 1):  # pylint: disable=not-an-iterable
        row(values, splits, i).sort()


@nb.njit(inline="always")
def transpose_csr(
    indices: IntArray, splits: IntArray, values: np.ndarray
) -> Tuple[IntArray, IntArray, np.ndarray]:
    col_indices = splits_to_ids(splits)  # initial row indices
    row_indices, col_indices, values = row_sorted(indices, col_indices, values)
    return col_indices, ids_to_splits(row_indices), values


# @nb.njit(inline="always", parallel=PARALLEL)
# def ragged_broadcast(
#     values: np.ndarray, row_splits: IntArray, out: Optional[np.ndarray] = None
# ):
#     out_ = np.empty((row_splits[-1],), values.dtype) if out is None else out
#     # for i in range(row_splits.size - 1):
#     for i in nb.prange(row_splits.size - 1):  # pylint: disable=not-an-iterable
#         out_[row_splits[i] : row_splits[i + 1]] = values[i]
#     return out_


# @nb.njit()
# def mask_rows(
#     values: np.ndarray, row_splits: IntArray, mask: BoolArray
# ) -> Tuple[np.ndarray, IntArray]:
#     flat_mask = ragged_broadcast(mask, row_splits)
#     values = values[flat_mask]
#     row_lengths = splits_to_lengths(row_splits)
#     splits = lengths_to_splits(row_lengths[mask])
#     return values, splits


# class RaggedArray(object):

#     def __init__(self, values, row_splits):
#         self.values = values
#         self.row_splits = row_splits

#     def row(self, i: int):
#         return self.values[self.row_splits[i]:self.row_splits[i + 1]]

#     def value_rowids(self):
#         row_lengths = self.row_lengths()
#         return np.repeat(np.arange(row_lengths.size), row_lengths)

#     def row_lengths(self):
#         return self.row_splits[1:] - self.row_splits[:-1]
