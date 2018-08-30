import numpy as np
import xarray as xr
import unittest
import nn.neural_net as neural_net
import nn.train as train
import nn.utility as utility
from nn.test.test_neural_net import NeuralNetTest

class TrainTest(unittest.TestCase):

    def test_shuffle_indexes(self):
        arr1 = xr.DataArray(np.arange(100).reshape((50, 2)), dims=('cases', 'inputs'))
        arr2 = xr.DataArray(np.arange(100).reshape((50, 2)), dims=('cases', 'inputs'))
        out1, out2 = train.shuffle_indexes(arr1, arr2)
        np.testing.assert_allclose(out1, out2)
        self.assertFalse(np.all(np.equal(arr2, out2)))

    def test_combine_arrays(self):
        arr1 = xr.DataArray(np.ones((3,2)), dims=('cases', 'inputs'))
        arr2 = xr.DataArray(np.arange(6).reshape((3, 2)), dims=('cases', 'inputs'))
        combined = train.combine_arrays(arr1, arr2)
        np.testing.assert_allclose(combined.coords['cases'], np.arange(6))
        self.assertDictEqual(dict(combined.sizes), {'cases': 6, 'inputs': 2})

    def test_empty_labels(self):
        arr1 = xr.DataArray(np.ones((3,2)), dims=('cases', 'inputs'))
        empty = train.empty_labels(arr1)
        np.testing.assert_allclose(empty, np.zeros((3, 10)))

    def test_tile_shuffled_cases(self):
        inputs = xr.DataArray(np.arange(100).reshape((50, 2)), dims=('cases', 'inputs'))
        labels = xr.DataArray(np.arange(100).reshape((50, 2)), dims=('cases', 'inputs'))
        truncated_inputs, truncated_labels = train.tile_shuffled_cases(inputs, labels, tile_size=29)
        tiled_inputs, tiled_labels = train.tile_shuffled_cases(inputs, labels, tile_size=229)
        self.assertDictEqual(dict(truncated_inputs.sizes), {'cases': 29, 'inputs': 2})
        self.assertDictEqual(dict(tiled_inputs.sizes), {'cases': 229, 'inputs': 2})
        np.testing.assert_allclose(truncated_inputs, truncated_labels)
        np.testing.assert_allclose(tiled_inputs, tiled_labels)
        self.assertFalse(np.all(np.equal(inputs.isel(cases=slice(29)), truncated_inputs)))
        self.assertFalse(np.all(np.equal(inputs, tiled_inputs.isel(cases=slice(50)))))

    def test_benchmark(self):
        images = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
        labels_onehot = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
        labels_onehot = neural_net.make_onehot(labels_onehot, np.arange(10))
        net = neural_net.NeuralNet((784, 30, 10))
        inputs = images.isel(cases=slice(1000))
        labels = labels_onehot.isel(cases=slice(1000))
        test_inputs = images.isel(cases=slice(50000, 50100))
        test_labels = labels_onehot.isel(cases=slice(50000, 50100))
        last, loss_arr = train.train_nn(net, inputs, labels, test_inputs, test_labels, do_print=False)
        last2, loss_arr2 = None, None
        for name_id, last2, loss_arr2 in train.benchmark(('test', net, inputs, labels),
            test_inputs=test_inputs,test_labels=test_labels,
            max_batches=100, num_cases=[1000], rates=[3.0], batch_sizes=[10], sample_rate=100,
            save_dir=None, do_print=False):
            pass
        #TODO why is benchmark not returning the same thing as test_nn?
        NeuralNetTest.assert_nn_equal(NeuralNetTest, last, last2)
        [np.testing.assert_allclose(a, b) for a, b in zip(loss_arr, loss_arr2)]
