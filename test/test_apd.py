"""Unit tests for apd.py"""

import unittest
import numpy as np
import xarray as xr
from littlenet import apd
from littlenet import neural_net as nn


class ApdTest(unittest.TestCase):

    def test_cost(self):
        apds = xr.DataArray(np.arange(6).reshape((2, 3)),
                            dims=(nn.DIM_CASE, apd.DIM_HIST))
        cost = apd.cost(apds)
        self.assertDictEqual(dict(cost.sizes), {nn.DIM_CASE: 2})
        np.testing.assert_allclose(cost, [1.5, (9+32+25)/4])
        cost = apd.cost(apds, dim=nn.DIM_CASE)
        self.assertDictEqual(dict(cost.sizes), {apd.DIM_HIST: 3})
        np.testing.assert_allclose(cost, [4.5, 8.5, 14.5])

    def test_triangular_to_linear_index(self):
        four = xr.DataArray([[0, 1], [0, 2], [0, 3], [1, 2], [
                            1, 3], [2, 3]], dims=(nn.DIM_CASE, 'index'))
        five = xr.DataArray([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
                             [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
                            dims=(nn.DIM_CASE, 'index'))
        linear = apd.triangular_to_linear_index(
            four.isel(index=0), four.isel(index=1), 4)
        np.testing.assert_allclose(linear, np.arange(6))
        linear = apd.triangular_to_linear_index(
            five.isel(index=0), five.isel(index=1), 5)
        np.testing.assert_allclose(linear, np.arange(10))

    def test_linear_to_triangular_indexes(self):
        four = apd.linear_to_triangular_indexes(xr.DataArray(np.arange(6)), 4)
        five = apd.linear_to_triangular_indexes(xr.DataArray(np.arange(10)), 5)
        np.testing.assert_allclose(
            four, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        np.testing.assert_allclose(five, [[0, 1], [0, 2], [0, 3], [0, 4], [
                                   1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])

    def test_apd_raw(self):
        num_buckets = 4
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape(
            (5, 2)), dims=(nn.DIM_CASE, nn.DIM_IN))
        apds = apd.apd_raw(activations, num_buckets=num_buckets)
        self.assertDictEqual(dict(apds.sizes), {
                             nn.DIM_CASE: 5, nn.DIM_IN: 2, apd.DIM_HIST: num_buckets})
        np.testing.assert_allclose(apds,
                                   [[[1, 0, 0, 0], [0.7, 0.3, 0, 0]],
                                    [[0.4, 0.6, 0, 0], [0.1, 0.9, 0, 0]],
                                       [[0, 0.8, 0.2, 0], [0, 0.5, 0.5, 0]],
                                       [[0, 0.2, 0.8, 0], [0, 0, 0.9, 0.1]],
                                       [[0, 0, 0.6, 0.4], [0, 0, 0.3, 0.7]]])

    def test_apd_area(self):
        apds = xr.DataArray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
                            dims=(nn.DIM_CASE, apd.DIM_HIST))
        area = apd.apd_area(apds)
        self.assertDictEqual(dict(area.sizes), {nn.DIM_CASE: 4})
        np.testing.assert_allclose(area, [0.5, 1, 2, 0.5])

    def test_diff(self):
        apd1 = xr.DataArray([[0, 1, 0, 1], [1, 1, 0, 0]],
                            dims=(nn.DIM_IN, apd.DIM_HIST))
        apd2 = xr.DataArray([[1, 0, 0, 1], [0.5, 1, 0.5, 0]],
                            dims=(nn.DIM_IN, apd.DIM_HIST))
        difference = apd.diff(apd1, apd2)
        self.assertDictEqual(dict(difference.sizes), {
                             nn.DIM_IN: 2, apd.DIM_HIST: 4})
        np.testing.assert_allclose(
            difference, [[1, 1, 0, 0], [0.5, 0, 0.5, 0]])

    def test_merge(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
                               dims=(nn.DIM_CASE, nn.DIM_IN, apd.DIM_HIST))
        apds = apd.merge(apd_raw)
        self.assertDictEqual(dict(apds.sizes), {apd.DIM_HIST: 4})
        np.testing.assert_allclose(apds, [0.25, 0.5, 0.25, 0.25])

        apd2 = apd.merge(apd_raw, nn.DIM_CASE, nn.DIM_IN)
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge(apd_raw, nn.DIM_CASE)
        self.assertDictEqual(dict(apds.sizes), {
                             nn.DIM_IN: 2, apd.DIM_HIST: 4})
        np.testing.assert_allclose(
            apds, [[0.5, 0.5, 0.5, 0], [0, 0.5, 0, 0.5]])

    def test_merge_ex(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
                               dims=(nn.DIM_CASE, nn.DIM_IN, apd.DIM_HIST))
        apds = apd.merge_ex(apd_raw)
        apd2 = apd.merge(apd_raw)
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge_ex(apd_raw, nn.DIM_CASE)
        apd2 = apd.merge(apd_raw, nn.DIM_IN)
        self.assertDictEqual(dict(apds.sizes), dict(apd2.sizes))
        np.testing.assert_allclose(apds, apd2)

        apds = apd.merge_ex(apd_raw, nn.DIM_CASE, nn.DIM_IN)
        self.assertDictEqual(dict(apds.sizes), dict(apd_raw.sizes))
        np.testing.assert_allclose(apds, apd_raw)

    def test_subset(self):
        apd_raw = xr.DataArray([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 0, 0, 1]]],
                               dims=(nn.DIM_CASE, nn.DIM_IN, apd.DIM_HIST))
        criterion = [0, 1]
        apds = apd.subset(apd_raw, **{nn.DIM_IN: np.equal(criterion, 1)})
        self.assertDictEqual(dict(apds.sizes), {
                             nn.DIM_CASE: 2, apd.DIM_HIST: 4})
        np.testing.assert_allclose(apds, [[0, 1, 0, 0], [0, 0, 0, 1]])
        apds = apd.subset(apd_raw, **{nn.DIM_IN: np.equal(
            criterion, 1), nn.DIM_CASE: np.equal(criterion, 1)})
        self.assertDictEqual(dict(apds.sizes), {apd.DIM_HIST: 4})
        np.testing.assert_allclose(apds, [0, 0, 0, 1])

    def test_apd_by_label(self):
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape(
            (5, 2)), dims=(nn.DIM_CASE, nn.DIM_IN))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=(nn.DIM_CASE))
        symbols = [0, 1]
        apds = apd.subset_by_label(apd.apd_raw(
            activations, num_buckets=4), labels, symbols)
        self.assertDictEqual(dict(apds.sizes), {
                             nn.DIM_IN: 2, nn.DIM_LABEL: 2, apd.DIM_HIST: 4})
        apd0 = apd.merge_ex(apd.apd_raw(
            activations.isel({nn.DIM_CASE: 0}), num_buckets=4), nn.DIM_IN)
        apd1 = apd.merge_ex(apd.apd_raw(activations.isel(
            {nn.DIM_CASE: slice(1, 4)}), num_buckets=4), nn.DIM_IN)
        np.testing.assert_allclose(apds[0], apd0)
        np.testing.assert_allclose(apds[1], apd1)

    def test_diff_by_label(self):
        activations = xr.DataArray(np.arange(0, 1.0, 0.1).reshape(
            (5, 2)), dims=(nn.DIM_CASE, nn.DIM_IN))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=(nn.DIM_CASE))
        symbols = [0, 1]
        apds = apd.diff_by_label(activations, labels, symbols, num_buckets=4)
        self.assertDictEqual(dict(apds.sizes), {
                             nn.DIM_IN: 2, nn.DIM_LABEL: 2, apd.DIM_HIST: 4})
        apd_total = apd.merge_ex(apd.apd_raw(
            activations, num_buckets=4), nn.DIM_IN)
        apd0 = apd.merge_ex(apd.apd_raw(
            activations.isel({nn.DIM_CASE: 0}), num_buckets=4), nn.DIM_IN)
        apd1 = apd.merge_ex(apd.apd_raw(activations.isel(
            {nn.DIM_CASE: slice(1, 4)}), num_buckets=4), nn.DIM_IN)
        np.testing.assert_allclose(apds[0], apd.diff(apd_total, apd0))
        np.testing.assert_allclose(apds[1], apd.diff(apd_total, apd1))

    def test_diff_by_pairs(self):
        activations = xr.DataArray(np.arange(0, 2.5, 0.1).reshape(
            (5, 5)), dims=(nn.DIM_CASE, nn.DIM_IN))
        labels = xr.DataArray([0, 1, 1, 1, 2], dims=(nn.DIM_CASE))
        symbols = [0, 1]
        raw_apds = apd.apd_raw(activations, num_buckets=4)
        apds = apd.diff_by_pairs(activations, labels, symbols, num_buckets=4)
        self.assertDictEqual(dict(apds.sizes), {
                             apd.DIM_PAIR: 10, nn.DIM_LABEL: 2, apd.DIM_HIST: 4})
        np.testing.assert_allclose(apds.isel({apd.DIM_PAIR: 0, nn.DIM_LABEL: 0}), apd.diff(
            raw_apds.isel({nn.DIM_IN: 0, nn.DIM_CASE: 0}),
            raw_apds.isel({nn.DIM_IN: 1, nn.DIM_CASE: 0})))
        np.testing.assert_allclose(apds.isel({apd.DIM_PAIR: 1, nn.DIM_LABEL: 0}), apd.diff(
            raw_apds.isel({nn.DIM_IN: 0, nn.DIM_CASE: 0}),
            raw_apds.isel({nn.DIM_IN: 2, nn.DIM_CASE: 0})))
        np.testing.assert_allclose(apds.isel({apd.DIM_PAIR: 9, nn.DIM_LABEL: 0}), apd.diff(
            raw_apds.isel({nn.DIM_IN: 3, nn.DIM_CASE: 0}),
            raw_apds.isel({nn.DIM_IN: 4, nn.DIM_CASE: 0})))
        np.testing.assert_allclose(apds.isel({apd.DIM_PAIR: 1, nn.DIM_LABEL: 1}), apd.diff(
            apd.merge(raw_apds.isel(
                {nn.DIM_IN: 0, nn.DIM_CASE: slice(1, 4)}), nn.DIM_CASE),
            apd.merge(raw_apds.isel({nn.DIM_IN: 2, nn.DIM_CASE: slice(1, 4)}), nn.DIM_CASE)))
        np.testing.assert_allclose(apds.isel({apd.DIM_PAIR: 9, nn.DIM_LABEL: 1}), apd.diff(
            apd.merge(raw_apds.isel(
                {nn.DIM_IN: 3, nn.DIM_CASE: slice(1, 4)}), nn.DIM_CASE),
            apd.merge(raw_apds.isel({nn.DIM_IN: 4, nn.DIM_CASE: slice(1, 4)}), nn.DIM_CASE)))


if __name__ == '__main__':
    unittest.main()
