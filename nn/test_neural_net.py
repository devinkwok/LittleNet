import numpy as np
import random, string
import xarray as xr
import matplotlib.pyplot as plt
import unittest
import math

from nn.neural_net import mkey, sigmoid, sigmoid_d, dict_subset, make_onehot, accuracy, accuracy_sum, cost_mean_squared, NeuralNet


LAYER_SIZES = [10, 5, 2]
NUM_LAYERS = len(LAYER_SIZES)-1
NUM_CASES = 3
NUM_LABELS = 2
INPUT_SIZE = LAYER_SIZES[0]
SIGMOID_INPUT = np.array([0, 1, -1])
SIGMOID_OUTPUT = np.array([0.5, 0.731058579, 0.268941421])
SIGMOID_D_OUTPUT = np.array([0.25, 0.196611933, 0.196611933])
EXPECTED_OUTPUT = 1./(1.+math.pow(math.e, -1*(LAYER_SIZES[1] * 1 / (1 + math.pow(math.e, -1)) + 1)))

class NeuralNetTest(unittest.TestCase):

    def assert_dimensions(self, matrices_dict, matrix_keyword, reference_dict_arrays):
        for i in range(NUM_LAYERS):
            dims = dict(matrices_dict[mkey(i, matrix_keyword)].sizes)
            reference = {key: values[i] for key, values in reference_dict_arrays.items()}
            self.assertDictEqual(dims, reference)

    def test_mkey(self):
        self.assertEqual(mkey(123, 'asdf'), 'layer123 asdf')
        NUM_TESTS = 10
        names = [''.join(random.choice(string.ascii_lowercase) for i in range(np.random.randint(0, 30))) for j in range(NUM_TESTS)]
        layer_numbers = np.random.randint(1000, size=NUM_TESTS)
        for i, name in zip(layer_numbers, names):
            self.assertEqual(mkey(i, name), 'layer' + str(i) + ' ' + name)
    
    def test_dict_subset(self):
        test_dict = {}
        for i in range(NUM_LAYERS):
            test_dict[mkey(i, 'weights')] = 0 
            test_dict[mkey(i, 'biases')] = 0
        weights = dict_subset(test_dict, 'weight').keys()
        biases = dict_subset(test_dict, 'bias').keys()
        layers = dict_subset(test_dict, 'layer').keys()
        subset = dict_subset(test_dict, '0', 'weights').keys()
        self.assertEqual(len(weights), NUM_LAYERS)
        self.assertEqual(len(biases), NUM_LAYERS)
        self.assertEqual(len(layers), NUM_LAYERS * 2)
        self.assertEqual(len(subset), 1)
        for i in range(NUM_LAYERS):
            self.assertTrue(mkey(i, 'weights') in weights)
            self.assertTrue(mkey(i, 'biases') in biases)
            self.assertTrue(mkey(i, 'weights') in layers)
            self.assertTrue(mkey(i, 'biases') in layers)
        self.assertTrue(mkey(0, 'weights') in subset)
    
    def test_make_onehot(self):
        int_labels = xr.DataArray(np.array([[0, 0], [1, 2]]), dims=('batches', 'cases'))
        int_symbols = [0, 1]
        int_onehot = make_onehot(int_labels, int_symbols)
        int_expected = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 0]]])
        self.assertDictEqual(dict(int_onehot.sizes), {'batches': int_labels.sizes['batches'],
            'cases': int_labels.sizes['cases'], 'labels': len(int_symbols)})
        for i, j in zip(int_onehot, int_expected):
            np.testing.assert_array_equal(i, j)
        #TODO: allow/disallow string labels, or maybe use coordinates instead
        # str_labels = xr.DataArray(np.array([['a', 'a'], ['bb', 'c']]), dims=('batches', 'cases'))
        # str_symbols = ['a', 'bb']
        # str_onehot = make_onehot(str_labels, str_symbols)
        # str_expected = np.array([[[1, 0],
        #     [1, 0]],
        #     [[0, 1],
        #     [0, 0]]])
        # self.assertDictEqual(dict(str_onehot.sizes), {'cases': len(str_labels), 'labels': len(str_symbols)})
        # for i, j in zip(str_onehot, str_expected):
        #     np.testing.assert_array_equal(i, j)
    
    def test_sigmoid(self):
        output = sigmoid(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_OUTPUT):
            self.assertAlmostEqual(i, j)
    
    def test_sigmoid_d(self):
        output = sigmoid_d(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_D_OUTPUT):
            self.assertAlmostEqual(i, j)

    def test_accuracy_sum(self):
        test_onehot = xr.DataArray([[0, 0.5], [0.5, 0]], dims=('cases', 'inputs'))
        goal_onehot = xr.DataArray([[0, 1], [0, 1]], dims=('cases', 'labels'))
        actual = accuracy_sum(test_onehot, goal_onehot)
        np.testing.assert_array_equal(accuracy_sum(test_onehot, goal_onehot), 1)

    def test_accuracy(self):
        test_onehot = xr.DataArray([[0, 0.5], [0.5, 0]], dims=('cases', 'inputs'))
        goal_onehot = xr.DataArray([[0, 1], [0, 1]], dims=('cases', 'labels'))
        expected = xr.DataArray([True, False], dims='cases')
        actual = accuracy(test_onehot, goal_onehot)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(actual.sum(), expected.sum())

    def test_cost_mean_squared(self):
        test_onehot = xr.DataArray(np.arange(10), dims='inputs')
        goal_onehot = xr.DataArray(np.arange(10), dims='labels')
        self.assertEqual(cost_mean_squared(test_onehot, goal_onehot), 0)

        test_onehot = xr.DataArray([[0, 0.5], [0.5, 0]], dims=('cases', 'inputs'))
        goal_onehot = xr.DataArray([[0, 1], [0, 1]], dims=('cases', 'labels'))
        expected = xr.DataArray([0.25/2, 1.25/2], dims=('cases'))
        summed_expected = (0.5*0.5*2+1) / 4
        actual = cost_mean_squared(test_onehot, goal_onehot, sum_along_dim='inputs')
        summed_actual = cost_mean_squared(test_onehot, goal_onehot)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(summed_expected, summed_actual)
    
    def test_init(self):
        with self.assertRaises(ValueError):
            NeuralNet((2,))

        net = NeuralNet(LAYER_SIZES)
        self.assert_dimensions(net.matrices, 'weights', {'inputs':LAYER_SIZES[:-1], 'neurons':LAYER_SIZES[1:]})
        self.assert_dimensions(net.matrices, 'biases', {'neurons':LAYER_SIZES[1:]})
    
    def test_pass_forward(self):
        net = NeuralNet(LAYER_SIZES, func_fill=np.ones)
        with self.assertRaises(ValueError):
            net.pass_forward(xr.DataArray(np.zeros((NUM_CASES, INPUT_SIZE)), dims=('cases', 'asdf')))
        with self.assertRaises(ValueError):
            net.pass_forward(xr.DataArray(np.zeros((NUM_CASES, INPUT_SIZE+10)), dims=('cases', 'inputs')))

        inputs = xr.DataArray(np.zeros((NUM_CASES, INPUT_SIZE)), dims=('cases', 'inputs'))
        outputs = net.pass_forward(inputs)
        self.assert_dimensions(outputs, 'pre_activation', {'cases':[NUM_CASES]*len(LAYER_SIZES), 'inputs':LAYER_SIZES})
        self.assert_dimensions(outputs, 'post_activation', {'cases':[NUM_CASES]*len(LAYER_SIZES), 'inputs':LAYER_SIZES})
        np.testing.assert_allclose(net.output_only(outputs).isel(inputs=0), EXPECTED_OUTPUT)

    def test_output_only(self):
        net = NeuralNet(LAYER_SIZES, func_fill=np.ones)
        inputs = xr.DataArray(np.zeros((NUM_CASES, INPUT_SIZE)), dims=('cases', 'inputs'))
        output = net.output_only(net.pass_forward(inputs))
        self.assertDictEqual(dict(output.sizes), {'cases': NUM_CASES, 'inputs': NUM_LABELS})
    
    def test_pass_back(self):
        net = NeuralNet(LAYER_SIZES, func_fill=np.ones)
        activations = {}
        for i, l_size in zip(range(NUM_LAYERS+1), LAYER_SIZES):
            activations[mkey(i, 'pre_activation')] = xr.DataArray(np.zeros((NUM_CASES, l_size)), dims=('cases', 'inputs'))
            activations[mkey(i, 'post_activation')] = xr.DataArray(np.ones((NUM_CASES, l_size)), dims=('cases', 'inputs'))
        inputs = xr.DataArray(np.ones((NUM_CASES, INPUT_SIZE)), dims=('cases', 'inputs'))
        activations = net.pass_forward(inputs)
        labels = make_onehot(xr.DataArray(np.arange(NUM_CASES), dims=('cases')), np.arange(NUM_LABELS))  # labels are 0 to n
        gradients = net.pass_back(activations, labels)
        self.assert_dimensions(gradients, 'weights', {'cases':[NUM_CASES]*NUM_LAYERS, 'inputs':LAYER_SIZES[:-1], 'neurons':LAYER_SIZES[1:]})
        self.assert_dimensions(gradients, 'biases', {'cases': [NUM_CASES] * NUM_LAYERS, 'neurons': LAYER_SIZES[1:]})
        # print('l', labels)
        # print('a', activations)
        # print('g', gradients)
        for i in range(NUM_LAYERS):
            w = gradients[mkey(i, 'weights')]
            b = gradients[mkey(i, 'biases')]
            # for i in range(NUM_CASES):
                # print('a', i, w.isel(cases=i))
                # print('b', i, activations[mkey(i, 'post_activation')].isel(cases=i))
            #     self.assertEqual(np.all(np.greater(w.isel(cases=i), 0)), i < EXPECTED_OUTPUT) # label less than output, gradient is positive
            #     self.assertEqual(np.all(np.less(w.isel(cases=i), 0)), i > EXPECTED_OUTPUT)
            #     self.assertEqual(np.all(np.greater(b.isel(cases=i), 0)), i < EXPECTED_OUTPUT)
            #     self.assertEqual(np.all(np.less(b.isel(cases=i), 0)), i > EXPECTED_OUTPUT)
    
    def test_train(self):
        net = NeuralNet(LAYER_SIZES, func_fill=np.ones)
        NUM_BATCHES = 4
        inputs = xr.DataArray(np.zeros((NUM_BATCHES * NUM_CASES, INPUT_SIZE)), dims=['cases', 'inputs'])
        labels = make_onehot(xr.DataArray(np.zeros((NUM_BATCHES * NUM_CASES,)), dims=['cases']), np.zeros(NUM_LABELS))
        inputs = inputs.groupby_bins('cases', NUM_BATCHES)
        labels = labels.groupby_bins('cases', NUM_BATCHES)
        trained = net.train(list(inputs), list(labels))
        self.assert_dimensions(trained.matrices, 'weights', {'inputs':LAYER_SIZES[:-1], 'neurons':LAYER_SIZES[1:]})
        self.assert_dimensions(trained.matrices, 'biases', {'neurons':LAYER_SIZES[1:]})
        gradient = net.pass_back(net.pass_forward(inputs.first()), labels.first())
        # for key in trained.matrices.keys():
        #     self.assertTrue(np.all(np.greater(gradient[key].isel(cases=0), 0), trained.matrices[key].less(0)))
        #     self.assertTrue(np.all(np.less(gradient[key].isel(cases=0), 0), trained.matrices[key].greater(0)))

if __name__ == '__main__':
    unittest.main()