"""Unit tests for neural_net.py"""

import math
import random
import string
import unittest
import numpy as np
import xarray as xr
from littlenet import neural_net as nn
from littlenet import utility

LAYER_SIZES = [10, 5, 2]
NUM_LAYERS = len(LAYER_SIZES)-1
NUM_CASES = 3
NUM_LABELS = 2
INPUT_SIZE = LAYER_SIZES[0]
SIGMOID_INPUT = np.array([0, 1, -1])
SIGMOID_OUTPUT = np.array([0.5, 0.731058579, 0.268941421])
SIGMOID_D_OUTPUT = np.array([0.25, 0.196611933, 0.196611933])
EXPECTED_OUTPUT = 1. / \
    (1.+math.pow(math.e, -1*(LAYER_SIZES[1]
                             * 1 / (1 + math.pow(math.e, -1)) + 1)))


class NeuralNetTest(unittest.TestCase):

    def assert_nn_equal(self, net1, net2):
        self.assertSetEqual(set(net1.matrices.keys()),
                            set(net2.matrices.keys()))
        for k in net1.matrices.keys():
            np.testing.assert_allclose(net1.matrices[k], net2.matrices[k])

    def nn_not_equal(self, net1, net2):
        if set(net1.matrices.keys()) != set(net2.matrices.keys()):
            return True
        return not np.all([np.all(np.equal(
            net1.matrices[k], net2.matrices[k])) for k in net1.matrices.keys()])

    def assert_dimensions(self, matrices_dict, matrix_keyword, reference_dict_arrays):
        for i in range(NUM_LAYERS):
            dims = dict(matrices_dict[nn.mkey(i, matrix_keyword)].sizes)
            reference = {key: values[i]
                         for key, values in reference_dict_arrays.items()}
            self.assertDictEqual(dims, reference)

    def test_nn_mkey(self):
        self.assertEqual(nn.mkey(123, 'asdf'), 'layer123 asdf')
        num_tests = 10
        names = [''.join(random.choice(string.ascii_lowercase) for i in range(
            np.random.randint(0, 30))) for j in range(num_tests)]
        layer_numbers = np.random.randint(1000, size=num_tests)
        for i, name in zip(layer_numbers, names):
            self.assertEqual(nn.mkey(i, name), 'layer' + str(i) + ' ' + name)

    def test_del_rows(self):
        arr = xr.DataArray(np.arange(12).reshape(
            (4, 3)), dims=(nn.DIM_IN, nn.DIM_OUT))
        arr = nn.del_rows(arr, nn.DIM_IN, [0, 2])
        arr = nn.del_rows(arr, nn.DIM_OUT, [2])
        self.assertDictEqual(dict(arr.sizes), {nn.DIM_IN: 2, nn.DIM_OUT: 2})
        np.testing.assert_array_equal(arr, [[3, 4], [9, 10]])

    def test_dict_subset(self):
        test_dict = {}
        for i in range(NUM_LAYERS):
            test_dict[nn.mkey(i, nn.KEY_WEIGHT)] = 0
            test_dict[nn.mkey(i, nn.KEY_BIAS)] = 0
        weights = nn.dict_subset(test_dict, 'weig').keys()
        biases = nn.dict_subset(test_dict, 'bias').keys()
        layers = nn.dict_subset(test_dict, 'layer').keys()
        subset = nn.dict_subset(test_dict, '0', nn.KEY_WEIGHT).keys()
        self.assertEqual(len(weights), NUM_LAYERS)
        self.assertEqual(len(biases), NUM_LAYERS)
        self.assertEqual(len(layers), NUM_LAYERS * 2)
        self.assertEqual(len(subset), 1)
        for i in range(NUM_LAYERS):
            self.assertTrue(nn.mkey(i, nn.KEY_WEIGHT) in weights)
            self.assertTrue(nn.mkey(i, nn.KEY_BIAS) in biases)
            self.assertTrue(nn.mkey(i, nn.KEY_WEIGHT) in layers)
            self.assertTrue(nn.mkey(i, nn.KEY_BIAS) in layers)
        self.assertTrue(nn.mkey(0, nn.KEY_WEIGHT) in subset)

    def test_sigmoid(self):
        output = nn.sigmoid(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_OUTPUT):
            self.assertAlmostEqual(i, j)

    def test_sigmoid_d(self):
        output = nn.sigmoid_d(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_D_OUTPUT):
            self.assertAlmostEqual(i, j)

    def test_accuracy(self):
        test_onehot = xr.DataArray([[0, 0.6], [0.6, 0], [0.4, 0], [
                                   0, 0.4]], dims=(nn.DIM_CASE, nn.DIM_IN))
        goal_onehot = xr.DataArray(
            [[0, 1], [0, 1], [0, 1], [0, 0]], dims=(nn.DIM_CASE, nn.DIM_LABEL))
        expected = xr.DataArray([True, False, False, True], dims=nn.DIM_CASE)
        actual = nn.accuracy(test_onehot, goal_onehot)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(actual.sum(), expected.sum())

    def test_accuracy_sum(self):
        test_onehot = xr.DataArray(
            [[0, 0.5], [0.5, 0]], dims=(nn.DIM_CASE, nn.DIM_IN))
        goal_onehot = xr.DataArray([[0, 1], [0, 1]], dims=(nn.DIM_CASE, nn.DIM_LABEL))
        np.testing.assert_array_equal(
            nn.accuracy_sum(test_onehot, goal_onehot), 1)

    def test_cost_mean_squared(self):
        test_onehot = xr.DataArray(np.arange(10), dims=nn.DIM_IN)
        goal_onehot = xr.DataArray(np.arange(10), dims=nn.DIM_LABEL)
        self.assertEqual(nn.cost_mean_squared(test_onehot, goal_onehot), 0)

        test_onehot = xr.DataArray(
            [[0, 0.5], [0.5, 0]], dims=(nn.DIM_CASE, nn.DIM_IN))
        goal_onehot = xr.DataArray([[0, 1], [0, 1]], dims=(nn.DIM_CASE, nn.DIM_LABEL))
        expected = xr.DataArray([0.25/2, 1.25/2], dims=(nn.DIM_CASE))
        summed_expected = (0.5*0.5*2+1) / 4
        actual = nn.cost_mean_squared(
            test_onehot, goal_onehot, sum_along_dim=nn.DIM_IN)
        summed_actual = nn.cost_mean_squared(test_onehot, goal_onehot)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(summed_expected, summed_actual)

    def test_init(self):
        with self.assertRaises(ValueError):
            nn.NeuralNet((2,))

        net = nn.NeuralNet(LAYER_SIZES)
        self.assert_dimensions(net.matrices, nn.KEY_WEIGHT, {
                               nn.DIM_IN: LAYER_SIZES[:-1], nn.DIM_OUT: LAYER_SIZES[1:]})
        self.assert_dimensions(net.matrices, nn.KEY_BIAS, {
                               nn.DIM_OUT: LAYER_SIZES[1:]})

    def test_pass_forward(self):
        net = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        with self.assertRaises(ValueError):
            net.pass_forward(xr.DataArray(
                np.zeros((NUM_CASES, INPUT_SIZE)), dims=(nn.DIM_CASE, 'asdf')))
        with self.assertRaises(ValueError):
            net.pass_forward(xr.DataArray(
                np.zeros((NUM_CASES, INPUT_SIZE+10)), dims=(nn.DIM_CASE, nn.DIM_IN)))

        inputs = xr.DataArray(
            np.zeros((NUM_CASES, INPUT_SIZE)), dims=(nn.DIM_CASE, nn.DIM_IN))
        outputs = net.pass_forward(inputs)
        self.assert_dimensions(outputs, nn.KEY_OUT_PRE, {
                               nn.DIM_CASE: [NUM_CASES]*len(LAYER_SIZES), nn.DIM_IN: LAYER_SIZES})
        self.assert_dimensions(outputs, nn.KEY_OUT_POST, {
                               nn.DIM_CASE: [NUM_CASES]*len(LAYER_SIZES), nn.DIM_IN: LAYER_SIZES})
        np.testing.assert_allclose(net.pass_forward_output_only(
            inputs).isel({nn.DIM_IN: 0}), EXPECTED_OUTPUT)

    def test_pass_forward_output_only(self):
        net = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        inputs = xr.DataArray(
            np.zeros((NUM_CASES, INPUT_SIZE)), dims=(nn.DIM_CASE, nn.DIM_IN))
        output = net.pass_forward_output_only(inputs)
        self.assertDictEqual(dict(output.sizes), {
                             nn.DIM_CASE: NUM_CASES, nn.DIM_IN: NUM_LABELS})

    def test_pass_back(self):
        net = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        activations = {}
        for i, l_size in zip(range(NUM_LAYERS+1), LAYER_SIZES):
            activations[nn.mkey(i, nn.KEY_OUT_PRE)] = xr.DataArray(
                np.zeros((NUM_CASES, l_size)), dims=(nn.DIM_CASE, nn.DIM_IN))
            activations[nn.mkey(i, nn.KEY_OUT_POST)] = xr.DataArray(
                np.ones((NUM_CASES, l_size)), dims=(nn.DIM_CASE, nn.DIM_IN))
        inputs = xr.DataArray(
            np.ones((NUM_CASES, INPUT_SIZE)), dims=(nn.DIM_CASE, nn.DIM_IN))
        activations = net.pass_forward(inputs)
        labels = utility.make_onehot(xr.DataArray(np.arange(NUM_CASES), dims=(
            nn.DIM_CASE)), np.arange(NUM_LABELS))  # labels are 0 to n
        gradients = net.pass_back(activations, labels)
        self.assert_dimensions(gradients, nn.KEY_WEIGHT, {nn.DIM_CASE: [
                               NUM_CASES]*NUM_LAYERS, nn.DIM_IN: LAYER_SIZES[:-1], nn.DIM_OUT: LAYER_SIZES[1:]})
        self.assert_dimensions(gradients, nn.KEY_BIAS, {
                               nn.DIM_CASE: [NUM_CASES] * NUM_LAYERS, nn.DIM_OUT: LAYER_SIZES[1:]})

    def test_train(self):
        net = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        net2 = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        self.assert_nn_equal(net, net2)
        num_batches = 4
        inputs = xr.DataArray(
            np.zeros((num_batches * NUM_CASES, INPUT_SIZE)), dims=[nn.DIM_CASE, nn.DIM_IN])
        labels = utility.make_onehot(xr.DataArray(
            np.zeros((num_batches * NUM_CASES,)), dims=[nn.DIM_CASE]), np.zeros(NUM_LABELS))
        trained = net.train(inputs, labels, batch_size=NUM_CASES)
        self.assert_dimensions(trained.matrices, nn.KEY_WEIGHT, {
                               nn.DIM_IN: LAYER_SIZES[:-1], nn.DIM_OUT: LAYER_SIZES[1:]})
        self.assert_dimensions(trained.matrices, nn.KEY_BIAS, {
                               nn.DIM_OUT: LAYER_SIZES[1:]})
        self.assertTrue(self.nn_not_equal(net, trained))

    def test_delete_neurons(self):
        net = nn.NeuralNet(LAYER_SIZES, func_fill=np.ones)
        inputs = xr.DataArray(
            np.zeros((NUM_CASES, INPUT_SIZE)), dims=[nn.DIM_CASE, nn.DIM_IN])
        activations = net.pass_forward(inputs)
        new_net = net.delete_neurons([[3, 1], [0]], activations=activations)
        sizes = [x for x in LAYER_SIZES]
        sizes[1] -= 2
        sizes[2] -= 1
        self.assert_dimensions(new_net.matrices, nn.KEY_WEIGHT, {
                               nn.DIM_IN: sizes[:-1], nn.DIM_OUT: sizes[1:]})
        self.assertEqual(new_net.matrices[nn.mkey(1, nn.KEY_BIAS)][0], activations[nn.mkey(
            1, nn.KEY_OUT_POST)][0, 0] * 2 + 1)


if __name__ == '__main__':
    unittest.main()
