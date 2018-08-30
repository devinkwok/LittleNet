import numpy as np
import random, string
import xarray as xr
import matplotlib.pyplot as plt
import unittest
import math
import nn.apd as apd

class ApdTest(unittest.TestCase):

    def test_cost(self):
        apds = xr.DataArray(np.arange(6).reshape((2,3)), dims=('cases', 'histogram_buckets'))
        cost = apd.cost(apds)
        self.assertDictEqual(dict(cost.sizes), {'cases': 2})
        np.testing.assert_allclose(cost, [1.5, (9+32+25)/4])
        cost = apd.cost(apds, dim='cases')
        self.assertDictEqual(dict(cost.sizes), {'histogram_buckets': 3})
        np.testing.assert_allclose(cost, [4.5, 8.5, 14.5])

    def test_triangular_to_linear_index(self):
        four = xr.DataArray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dims=('cases', 'index'))
        five = xr.DataArray([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            dims=('cases', 'index'))
        linear = apd.triangular_to_linear_index(four.isel(index=0), four.isel(index=1), 4)
        np.testing.assert_allclose(linear, np.arange(6))
        linear = apd.triangular_to_linear_index(five.isel(index=0), five.isel(index=1), 5)
        np.testing.assert_allclose(linear, np.arange(10))

    def test_linear_to_triangular_indexes(self):
        four = apd.linear_to_triangular_indexes(xr.DataArray(np.arange(6)), 4)
        five = apd.linear_to_triangular_indexes(xr.DataArray(np.arange(10)), 5)
        np.testing.assert_allclose(four, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2,3]])
        np.testing.assert_allclose(five, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2,3], [2, 4], [3, 4]])

    def test_apd_raw(self):
        NUM_BUCKETS = 4
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape((5,2)), dims=('cases', 'inputs'))
        apds = apd.apd_raw(activations, num_buckets=NUM_BUCKETS)
        self.assertDictEqual(dict(apds.sizes), {'cases': 5, 'inputs': 2, 'histogram_buckets': NUM_BUCKETS})
        np.testing.assert_allclose(apds,
            [[[1, 0, 0, 0], [0.7, 0.3, 0, 0]],
             [[0.4, 0.6, 0, 0], [0.1, 0.9, 0, 0]],
             [[0, 0.8, 0.2, 0], [0, 0.5, 0.5, 0]],
             [[0, 0.2, 0.8, 0], [0, 0, 0.9, 0.1]],
             [[0, 0, 0.6, 0.4], [0, 0, 0.3, 0.7]]])

    def test_apd_area(self):
        apds = xr.DataArray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            dims=('cases', 'histogram_buckets'))
        area = apd.apd_area(apds)
        self.assertDictEqual(dict(area.sizes), {'cases': 4})
        np.testing.assert_allclose(area, [0.5, 1, 2, 0.5])

    def test_diff(self):
        apd1 = xr.DataArray([[0, 1, 0, 1], [1, 1, 0, 0]], dims=('inputs', 'histogram_buckets'))
        apd2 = xr.DataArray([[1, 0, 0, 1], [0.5, 1, 0.5, 0]], dims=('inputs', 'histogram_buckets'))
        difference = apd.diff(apd1, apd2)
        self.assertDictEqual(dict(difference.sizes), {'inputs': 2, 'histogram_buckets': 4})
        np.testing.assert_allclose(difference, [[1, 1, 0, 0], [0.5, 0, 0.5, 0]])

    def test_merge(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
             dims=('cases', 'inputs', 'histogram_buckets'))
        apds = apd.merge(apd_raw)
        self.assertDictEqual(dict(apds.sizes), {'histogram_buckets': 4})
        np.testing.assert_allclose(apds, [0.25, 0.5, 0.25, 0.25])

        apd2 = apd.merge(apd_raw, 'cases', 'inputs')
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge(apd_raw, 'cases')
        self.assertDictEqual(dict(apds.sizes), {'inputs': 2, 'histogram_buckets': 4})
        np.testing.assert_allclose(apds, [[0.5, 0.5, 0.5, 0], [0, 0.5, 0, 0.5]])

    def test_merge_ex(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
             dims=('cases', 'inputs', 'histogram_buckets'))
        apds = apd.merge_ex(apd_raw)
        apd2 = apd.merge(apd_raw)
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge_ex(apd_raw, 'cases')
        apd2 = apd.merge(apd_raw, 'inputs')
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge_ex(apd_raw, 'cases', 'inputs')
        self.assertDictEqual(dict(apds.sizes), dict(apd_raw.sizes))
        np.testing.assert_allclose(apds, apd_raw)

    def test_subset(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
             dims=('cases', 'inputs', 'histogram_buckets'))
        criterion = [0, 1]
        apds = apd.subset(apd_raw, inputs=np.equal(criterion, 1))
        self.assertDictEqual(dict(apds.sizes), {'cases': 2, 'histogram_buckets': 4})
        np.testing.assert_allclose(apds, [[0, 1, 0, 0], [0, 0, 0, 1]])
        apds = apd.subset(apd_raw, inputs=np.equal(criterion, 1), cases=np.equal(criterion, 1))
        self.assertDictEqual(dict(apds.sizes), {'histogram_buckets': 4})
        np.testing.assert_allclose(apds, [0, 0, 0, 1])

    def test_apd_by_label(self):
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape((5, 2)), dims=('cases', 'inputs'))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=('cases'))
        symbols = [0, 1]
        apds = apd.subset_by_label(apd.apd_raw(activations, num_buckets=4), labels, symbols)
        self.assertDictEqual(dict(apds.sizes), {'inputs': 2, 'labels': 2, 'histogram_buckets': 4})
        apd0 = apd.merge_ex(apd.apd_raw(activations.isel(cases=0), num_buckets=4), 'inputs')
        apd1 = apd.merge_ex(apd.apd_raw(activations.isel(cases=slice(1, 4)), num_buckets=4), 'inputs')
        np.testing.assert_allclose(apds[0], apd0)
        np.testing.assert_allclose(apds[1], apd1)

    def test_diff_by_label(self):
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape((5, 2)), dims=('cases', 'inputs'))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=('cases'))
        symbols = [0, 1]
        apds = apd.diff_by_label(activations, labels, symbols, num_buckets=4)
        self.assertDictEqual(dict(apds.sizes), {'inputs': 2, 'labels': 2, 'histogram_buckets': 4})
        apd_total = apd.merge_ex(apd.apd_raw(activations, num_buckets=4), 'inputs')
        apd0 = apd.merge_ex(apd.apd_raw(activations.isel(cases=0), num_buckets=4), 'inputs')
        apd1 = apd.merge_ex(apd.apd_raw(activations.isel(cases=slice(1, 4)), num_buckets=4), 'inputs')
        np.testing.assert_allclose(apds[0], apd.diff(apd_total, apd0))
        np.testing.assert_allclose(apds[1], apd.diff(apd_total, apd1))

    def test_diff_by_pairs(self):
        activations = xr.DataArray(np.arange(0, 2.5, 0.1).reshape((5, 5)), dims=('cases', 'inputs'))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=('cases'))
        symbols = [0, 1]
        raw_apds = apd.apd_raw(activations, num_buckets=4)
        apds = apd.diff_by_pairs(activations, labels, symbols, num_buckets=4)
        self.assertDictEqual(dict(apds.sizes), {'pairs': 10, 'labels': 2, 'histogram_buckets': 4})
        np.testing.assert_allclose(apds.isel(pairs=0, labels=0),
            apd.diff(raw_apds.isel(inputs=0, cases=0), raw_apds.isel(inputs=1, cases=0)))
        np.testing.assert_allclose(apds.isel(pairs=1, labels=0),
            apd.diff(raw_apds.isel(inputs=0, cases=0), raw_apds.isel(inputs=2, cases=0)))
        np.testing.assert_allclose(apds.isel(pairs=9, labels=0),
            apd.diff(raw_apds.isel(inputs=3, cases=0), raw_apds.isel(inputs=4, cases=0)))
        np.testing.assert_allclose(apds.isel(pairs=1, labels=1), apd.diff(
            apd.merge(raw_apds.isel(inputs=0, cases=slice(1, 4)), 'cases'),
            apd.merge(raw_apds.isel(inputs=2, cases=slice(1, 4)), 'cases')))
        np.testing.assert_allclose(apds.isel(pairs=9, labels=1), apd.diff(
            apd.merge(raw_apds.isel(inputs=3, cases=slice(1, 4)), 'cases'),
            apd.merge(raw_apds.isel(inputs=4, cases=slice(1, 4)), 'cases')))

if __name__ == '__main__':
    unittest.main()
