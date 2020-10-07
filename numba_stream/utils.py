import os
from typing import Optional, Tuple

import numba as nb
import numpy as np

PARALLEL = os.environ.get("NUMBA_PARALLEL", "1") != "0"
IntArray = np.ndarray


@nb.njit(inline="always")
def double_length(x):
    return np.concatenate((x, np.empty_like(x)), axis=0)


@nb.njit(inline="always")
def max_on_axis(x: np.ndarray, axis=0, out: Optional[np.ndarray] = None):
    if axis != 0 or x.ndim != 2:
        raise NotImplementedError("TODO")
    nd = x.shape[1]
    if out is None:
        out = np.empty((nd,), dtype=x.dtype)

    # errors using nb.prange...
    for i in range(nd):
        out[i] = np.max(x[:, i])
    return out


@nb.njit(inline="always")
def min_on_axis(x: np.ndarray, axis=0, out: Optional[np.ndarray] = None):
    if axis != 0 or x.ndim != 2:
        raise NotImplementedError("TODO")
    nd = x.shape[1]
    if out is None:
        out = np.empty((nd,), dtype=x.dtype)

    # errors using nb.prange...
    for i in range(nd):
        out[i] = np.min(x[:, i])
    return out


@nb.njit(inline="always")
def min_on_leading_axis(x: np.ndarray):
    out = np.empty((x.shape[1],), dtype=x.dtype)
    # errors using nb.prange...
    nd = x.shape[1]
    for i in range(nd):
        out[i] = np.min(x[:, i])
    return out


@nb.njit()
def merge(
    times0: IntArray, coords0: IntArray, times1: IntArray, coords1: IntArray
) -> Tuple[IntArray, IntArray]:
    assert len(times0) == len(coords0)
    assert len(times1) == len(coords1)
    total = times0.size + times1.size
    i0 = 0
    i1 = 0
    t0 = times0[0]
    t1 = times1[0]
    out_times = np.empty((total,), dtype=times0.dtype)
    out_coords = np.empty((total, coords0.shape[1]), dtype=coords0.dtype)
    for i in range(total):
        if t1 < t0:
            out_times[i] = t1
            out_coords[i] = coords1[i1]
            i1 += 1
            if i1 < len(times1):
                t1 = times1[i1]
            else:
                out_times[i + 1 :] = times0[i0:]
                out_coords[i + 1 :] = coords0[i0:]
                break
        else:
            out_times[i] = t0
            out_coords[i] = coords0[i0]
            i0 += 1
            if i0 < len(times0):
                t0 = times0[i0]
            else:
                out_times[i + 1 :] = times1[i1:]
                out_coords[i + 1 :] = coords1[i1:]
                break
    return out_times, out_coords


@nb.njit(inline="always")
def prod(x):
    p = 1
    for xi in x:
        p *= xi
    return p
