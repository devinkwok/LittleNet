import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import unittest
import math

from neural_net import NeuralNet as Nn

def assert_dimensions(matrices_dict, keyword, **dims_dict):
    dims_size = len(dims_dict)
    for key, values in dims_dict.items():
        for i, value in zip(range(NUM_LAYERS), values):
            dims = matrices_dict['layer' + i + ' ' + keyword].sizes
            assertEqual(len(dims), dims_size)
            assertTrue(key in dims)
            assertEqual(dims[key], value)

# DEPRECIATED remove when assert_dimensions() is working
def assert_dict_contains(dict_to_search, reference_dict):
    for k in reference_dict.keys():
        assertTrue(k in dict_to_search)
        assertEqual(dict_to_search[k], reference_dict[k])


LAYER_SIZES = [10, 5, 2]
NUM_LAYERS = len(LAYER_SIZES)
NUM_CASES = 3
NUM_LABELS = 2
INPUT_SIZE = LAYER_SIZES[0]
SIGMOID_INPUT = np.arange(-1, 2, 1)
SIGMOID_OUTPUT = np.array([0.5, 0.731058579, 0.268941421])
SIGMOID_D_OUTPUT = np.array([0.25, 0.196611933, 0.196611933])
EXPECTED_OUTPUT = 1./(1.+math.pow(math.e, (LAYER_SIZES[1]*1/(1+math.pow(math.e, -1)) + 1) ))

class NeuralNetTest(unittest.TestCase):
    
    def test_sigmoid(self):
        output = Nn.sigmoid(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_OUTPUT):
            self.assertAlmostEqual(i, j)
    
    def test_sigmoid_d(self):
        output = Nn.sigmoid_d(SIGMOID_INPUT)
        for i, j in zip(output, SIGMOID_D_OUTPUT):
            self.assertAlmostEqual(i, j)
    

    def test_init(self):
        net = Nn(LAYER_SIZES)
        # DEPRECIATED remove when assert_dimensions() is working
        # for l_size, next_l in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])
        #     assert_dict_contains(net.tensor['layer'+i+' weights'].sizes,
        #         {'inputs': l_size,
        #         'neurons': next_l})
        #     assert_dict_contains(net.tensor['layer'+i+' biases'].sizes,
        #         {'neurons': next_l})
        assert_dimensions(net, 'weights', inputs=LAYER_SIZES[:-1], neurons=LAYER_SIZES[1:])
        assert_dimensions(net, 'biases', neurons=LAYER_SIZES[1:])
    
    def test_dict_subset(self):
        for i in range(NUM_LAYERS):
            test_dict['layer'+i+' weights'] = 0 
            test_dict['layer'+i+' biases'] = 0
        weights = Nn.dict_subset(test_dict, 'weight').keys()
        biases = Nn.dict_subset(test_dict, 'bias').keys()
        layers = Nn.dict_subset(test_dict, 'layer').keys()
        subset = Nn.dict_subset(test_dict, '0', 'weights').keys()
        assertEqual(len(weights), NUM_LAYERS)
        assertEqual(len(biases), NUM_LAYERS)
        assertEqual(len(layers), NUM_LAYERS * 2)
        assertEqual(len(subset), 1)
        for i in range(NUM_LAYERS):
            assertTrue('layer'+i+' weights' in weights)
            assertTrue('layer'+i+' biases' in biases)
            assertTrue('layer'+i+' weights' in layers)
            assertTrue('layer'+i+' biases' in layers)
        assertTrue('layer0 weights' in subset)
    
    # def test_pass_forward(self):
    #     net = Nn(LAYER_SIZES, func_fill=np.ones)
    #     inputs = xr.DataArray(np.zeros((NUM_CASES, INPUT_SIZE)), dims=('cases', 'inputs'))
    #     outputs = net.pass_forward(inputs)
    #     # DEPRECIATED remove when assert_dimensions() is working
    #     # for activation, l_size in zip(outputs['layer'+i+' activations'], LAYER_SIZES):
    #     #     assert_dict_contains(activation.sizes), {'cases': NUM_CASES, 'inputs': l_size})
    #     assert_dimensions(outputs, 'activations', cases=[NUM_CASES]*NUM_LAYERS, inputs=LAYER_SIZES)
    #     assertAlmostEqual(outputs['layer'+(NUM_LAYERS-1)+' activations'].isel(inputs=0, neurons=0), EXPECTED_OUTPUT)

    # def test_gradient(self):
    #     net = Nn(LAYER_SIZES, func_fill=np.ones)
    #     for i, l_size in zip(range(NUM_LAYERS), LAYER_SIZES):
    #         activations['layer'+i+' activations'] = xr.DataArray(np.ones((NUM_CASES, l_size)), dims=('cases', 'activations'))
    #     gradients = net.gradient(activations, np.arange(NUM_CASES))
    #     assert_dimensions(gradients, 'weights', cases=[NUM_CASES]*NUM_LAYERS, inputs=LAYER_SIZES[:-1], neurons=LAYER_SIZES[1:])
    #     assert_dimensions(gradients, 'biases', cases=[NUM_CASES]*NUM_LAYERS, neurons=LAYER_SIZES[1:])
    #     for i in range(NUM_LAYERS):
    #         w = gradients['layer' + i + ' weights']
    #         b = gradients['layer' + i + ' biases']
    #         for i in range(NUM_CASES):
    #             assertEqual(np.all(np.less(w.isel(cases=i), 0)), i < EXPECTED_OUTPUT)
    #             assertEqual(np.all(np.greater(w.isel(cases=i), 0)), i > EXPECTED_OUTPUT)
    #             assertEqual(np.all(np.less(b.isel(cases=i), 0)), i < EXPECTED_OUTPUT)
    #             assertEqual(np.all(np.greater(b.isel(cases=i), 0)), i > EXPECTED_OUTPUT)
    
    # def test_backpropagate(self):
    #     net = Nn(LAYER_SIZES, func_fill=np.ones)
    #     NUM_BATCHES = 4
    #     training_set = xr.Dataset(
    #         {'training_inputs': (['batches', 'cases', 'inputs'], np.zeros((NUM_BATCHES, NUM_CASES, INPUT_SIZE))),
    #         'training_labels': (['batches', 'cases', 'label_onehot'], np.zeros((NUM_BATCHES, NUM_CASES, NUM_LABELS)))},
    #         coords={'label_onehot': [i for i in range(NUM_LABELS)]} )
    #     trained = net.backpropagate(training_set['training_inputs'], training_set['training_labels'])
    #     assert_dimensions(trained, 'weights', cases=[NUM_CASES]*NUM_LAYERS, inputs=LAYER_SIZES[:-1], neurons=LAYER_SIZES[1:])
    #     assert_dimensions(trained, 'biases', cases=[NUM_CASES]*NUM_LAYERS, neurons=LAYER_SIZES[1:])
    #     gradient = net.gradient(net.pass_forward(training_set['training_inputs'].isel(batches=0)), training_set['training_labels'].isel(batches=0))
    #     for g, t in zip(gradient, trained):
    #         assertTrue(np.all(np.less(g, t)))

if __name__ == '__main__':
    unittest.main()