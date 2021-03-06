# import os

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import unittest

import numpy as np

import numba_stream.utils as utils


class UtilsTest(unittest.TestCase):
    def test_max_on_axis(self):
        np.random.seed(123)
        n = 100
        m = 5
        x = np.random.uniform(size=(n, m))
        actual = utils.max_on_axis(x, axis=0)
        expected = np.max(x, axis=0)
        np.testing.assert_allclose(actual, expected)

        # test using out kwarg
        expected = np.empty((m,), dtype=x.dtype)
        np.max(x, axis=0, out=expected)

        actual = np.empty((m,), dtype=x.dtype)
        utils.max_on_axis(x, axis=0, out=actual)

        np.testing.assert_allclose(actual, expected)

    def test_min_on_axis(self):
        np.random.seed(123)
        n = 100
        m = 5
        x = np.random.uniform(size=(n, m))
        actual = utils.min_on_axis(x)
        expected = np.min(x, axis=0)
        np.testing.assert_allclose(actual, expected)

        # test using out kwarg
        expected = np.empty((m,), dtype=x.dtype)
        np.min(x, axis=0, out=expected)

        actual = np.empty((m,), dtype=x.dtype)
        utils.min_on_axis(x, out=actual)

        np.testing.assert_allclose(actual, expected)

    def test_merge(self):
        t0 = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        c0 = np.expand_dims(t0, axis=-1) * 2
        t1 = np.array([1, 2, 6, 12])
        c1 = -np.expand_dims(t1, axis=-1)

        expected_t = [0, 1, 2, 5, 6, 10, 12, 15, 20]
        expected_c = np.expand_dims([0, -1, -2, 10, -6, 20, -12, 30, 40], axis=-1)

        actual_t, actual_c = utils.merge(t0, c0, t1, c1)
        np.testing.assert_equal(actual_t, expected_t)
        np.testing.assert_equal(actual_c, expected_c)

        # should be commutative if there are not duplicates
        actual_t, actual_c = utils.merge(t1, c1, t0, c0)
        np.testing.assert_equal(actual_t, expected_t)
        np.testing.assert_equal(actual_c, expected_c)

    def test_prod(self):
        x = np.random.uniform(size=(10,))
        np.testing.assert_allclose(np.prod(x), utils.prod(x))
        x = np.random.uniform(size=(10,), high=10).astype(np.int64)
        np.testing.assert_equal(np.prod(x), utils.prod(x))


if __name__ == "__main__":
    unittest.main()
