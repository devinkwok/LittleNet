"""Unit tests for utility.py"""

import unittest
import numpy as np
import xarray as xr
from littlenet import neural_net as nn
from littlenet import utility as util


class UtilityTest(unittest.TestCase):

    def test_make_onehot(self):
        int_labels = xr.DataArray(
            np.array([[0, 0], [1, 2]]), dims=('batches', nn.DIM_CASE))
        int_symbols = [0, 1]
        int_onehot = util.make_onehot(int_labels, int_symbols)
        int_expected = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 0]]])
        self.assertDictEqual(dict(int_onehot.sizes), {
                             'batches': int_labels.sizes['batches'],
                             nn.DIM_CASE: int_labels.sizes[nn.DIM_CASE],
                             nn.DIM_LABEL: len(int_symbols)})
        for i, j in zip(int_onehot, int_expected):
            np.testing.assert_array_equal(i, j)

    def test_rotate_images_90_deg(self):
        images = xr.DataArray(np.arange(27).reshape((3, 3, 3)),
                              dims=(nn.DIM_CASE, nn.DIM_Y, nn.DIM_X)).stack(
                                  inputs=(nn.DIM_Y, nn.DIM_X))
        img1 = util.rotate_images_90_deg(images)
        self.assertFalse(img1.equals(images))
        np.testing.assert_allclose(img1.unstack(nn.DIM_IN).coords[nn.DIM_Y], np.arange(3))
        np.testing.assert_allclose(img1[0], [6, 3, 0, 7, 4, 1, 8, 5, 2])
        img2 = util.rotate_images_90_deg(images, num_clockwise=-3)
        np.testing.assert_allclose(img1, img2)
        img1 = util.rotate_images_90_deg(images, num_clockwise=3)
        np.testing.assert_allclose(img1[0], [2, 5, 8, 1, 4, 7, 0, 3, 6])
        self.assertFalse(img1.equals(images))
        img2 = util.rotate_images_90_deg(images, num_clockwise=-1)
        np.testing.assert_allclose(img1, img2)
        img1 = util.rotate_images_90_deg(images, num_clockwise=2)
        np.testing.assert_allclose(img1[0], [8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertFalse(img1.equals(images))
        img2 = util.rotate_images_90_deg(images, num_clockwise=-2)
        np.testing.assert_allclose(img1, img2)

    def test_flip_images_on_angle(self):
        images = xr.DataArray(np.arange(9).reshape((3, 3)),
                              dims=(nn.DIM_Y, nn.DIM_X)).stack(inputs=(nn.DIM_Y, nn.DIM_X))
        img1 = util.flip_images_on_angle(images)
        img2 = util.flip_images_on_angle(images, topright_to_bottomleft=True)
        np.testing.assert_allclose(img1, [0, 3, 6, 1, 4, 7, 2, 5, 8])
        np.testing.assert_allclose(img2, [8, 5, 2, 7, 4, 1, 6, 3, 0])
        np.testing.assert_allclose(img1.unstack(nn.DIM_IN).coords[nn.DIM_Y], np.arange(3))

    def test_flip_images(self):
        images = xr.DataArray(np.arange(27).reshape((3, 3, 3)),
                              dims=(nn.DIM_CASE, nn.DIM_Y, nn.DIM_X)).stack(
                                  inputs=(nn.DIM_Y, nn.DIM_X))
        img1 = util.flip_images(images)
        np.testing.assert_allclose(img1.unstack(nn.DIM_IN).coords[nn.DIM_Y], np.arange(3))
        np.testing.assert_allclose(img1[0], [6, 7, 8, 3, 4, 5, 0, 1, 2])
        self.assertFalse(img1.equals(images))
        img2 = util.flip_images(img1)
        np.testing.assert_allclose(img2, images)
        img1 = util.flip_images(images, dim=nn.DIM_X)
        self.assertFalse(img1.equals(images))
        img2 = util.flip_images(img1, dim=nn.DIM_X)
        np.testing.assert_allclose(img2, images)

    def test_quarter_images(self):
        images = xr.DataArray(np.arange(16).reshape((4, 4)),
                              dims=(nn.DIM_Y, nn.DIM_X)).stack(inputs=(nn.DIM_Y, nn.DIM_X))
        img1 = util.quarter_images(images)
        np.testing.assert_allclose(img1.unstack(nn.DIM_IN).coords[nn.DIM_Y], np.arange(4))
        np.testing.assert_allclose(
            img1, [10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5])

    def test_compose_params(self):
        params1 = ['a', 'b']
        params2 = [0, 1, 2]
        params3 = [30, 40, 50, 60]
        params = util.compose_params(params1, params2, params3)
        np.testing.assert_allclose(
            [len(params1) * len(params2) * len(params3), 3], params.shape)
        np.testing.assert_array_equal(params[0], ['a', 0, 30])
        np.testing.assert_array_equal(params[-1], ['b', 2, 60])


if __name__ == '__main__':
    unittest.main()
