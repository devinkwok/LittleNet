import numpy as np
import xarray as xr

# tensor: an xarray.DataSet containing multiple instances of xarray.DataArray
# parameters are of format 'name': dimensions, where dimension = 0 is a single variable per layer, dimension = 1 is layer size, dimension = 2 is previous layer size x layer size
class NeuralNet(object):

    @staticmethod
    def sigmoid(np_array):
        return np_array #TODO

    @staticmethod
    def sigmoid_d(np_array):
        return np_array #TODO

    # returns subset of dictionary of all keys containing contents in each of *keywords
    @staticmethod
    def dict_subset(dictionary, *keywords):
        return None #TODO
        subset = {}
        for key, value in dictionary.items():
            do_add = True
            for word in keywords:
                if word not in key:
                    do_add = False
            if do_add:
                subset[key] = value
        return subset

    # parameters is a dictionary of 'name': dimensionality
    # layer_sizes 
    # every layer has >=1 parameter types, of dimension >=1
    # func_fill takes in 1 argument of numpy shape, returns initial values for neuron parameters
    # func_activation and func_activation_d each take in 1 argument as numpy array
    def __init__(self, layer_sizes, func_fill=np.random.randn, func_activation=sigmoid, func_activation_d=sigmoid_d):
        return #TODO

    # inputs are m x n matrix of m vectors of inputs of n size, n being the size of the first layer
    def pass_forward(self, inputs, apply_activation=True):
        return None  #TODO
        # returns MultilayerTensor object containing activations, as one m x n size parameter per layer

    # activations are from pass_forward where apply_activation=False
    def gradient(self, activations, goal):
        return None  #TODO
        # returns NeuralNet object containing gradients, same format as current NeuralNet but all parameters have additional dimension x m
        
    def backpropagate(self, inputs, labels, learn_rate=1):
        return None  #TODO
