import numpy as np
import xarray as xr

def mkey(layer, argument_name):
    return 'layer'+str(layer)+' '+argument_name

# returns subset of dictionary of all keys containing contents in each of *keywords
def dict_subset(dictionary, *keywords):
    subset = {}
    for key, value in dictionary.items():
        do_add = True
        for word in keywords:
            if word not in key:
                do_add = False
        if do_add:
            subset[key] = value
    return subset

def make_onehot(label_xarray, symbols_list):
    onehot_array = []
    func_equal = np.equal
    # TODO: allow/disallow strings as symbols
    # if isinstance(label_xarray[0], str):
    #     func_equal = np.core.defchararray.equal
    for i, symbol in zip(range(len(symbols_list)), symbols_list):
        onehot_array.append(func_equal(label_xarray, symbol))
    labels_onehot = xr.concat(onehot_array, dim='labels_onehot').astype(float)
    return labels_onehot.transpose(*label_xarray.dims, 'labels_onehot')

def sigmoid(np_array):
    return np.divide(1, np.add(1, np.exp(np.multiply(np_array, -1))))

def sigmoid_d(np_array):
    e_a = np.exp(np_array)
    return np.divide(e_a, np.square(np.add(e_a, 1)))

# tensor: an xarray.DataSet containing multiple instances of xarray.DataArray
# parameters are of format 'name': dimensions, where dimension = 0 is a single variable per layer
# dimension = 1 is layer size, dimension = 2 is previous layer size x layer size
class NeuralNet(object):

    # parameters is a dictionary of 'name': dimensionality
    # layer_sizes 
    # every layer has >=1 parameter types, of dimension >=1
    # func_fill takes in 1 argument of numpy shape, returns initial values for neuron parameters
    # func_activation and func_activation_d each take in 1 argument as numpy array
    def __init__(self, layer_sizes, func_fill=lambda x: np.random.randn(*x), func_activation=sigmoid, func_activation_d=sigmoid_d):
        self.num_layers = len(layer_sizes) - 1
        self.func_activation = func_activation
        self.func_activation_d = func_activation_d
        self.matrices = {}
        for i, l_size, next_l in zip(range(self.num_layers), layer_sizes[:-1], layer_sizes[1:]):
            self.matrices[mkey(i, 'weights')] = xr.DataArray(func_fill((l_size, next_l)), dims=('inputs', 'neurons'))
            self.matrices[mkey(i, 'biases')] = xr.DataArray(func_fill((next_l,)), dims=('neurons'))
        return None

    # inputs are m x n matrix of m vectors of inputs of n size, n being the size of the first layer
    def pass_forward(self, training_inputs, func_normalize=lambda x: x):
        activations = {mkey(0, 'pre_activation'): training_inputs,
            mkey(0, 'post_activation'): func_normalize(training_inputs)}
        for i in range(0, self.num_layers):
            pre_activation = np.add(xr.dot(activations[mkey(i, 'post_activation')], self.matrices[mkey(i, 'weights')],
                dims=('inputs', 'inputs')), self.matrices[mkey(i, 'biases')])
            activations[mkey(i+1, 'pre_activation')] = pre_activation.rename(neurons='inputs')
            activations[mkey(i+1, 'post_activation')] = self.func_activation(pre_activation).rename(neurons='inputs')
        return activations
        # returns matrices dictionary containing activations, as one m x n size parameter per layer

    # activations are from pass_forward where apply_activation=False
    def pass_back(self, activations, goal_label, func_loss_d=lambda output_v, goal_v: np.subtract(goal_v, output_v)):
        gradients = {}
        partial_d = func_loss_d(activations[mkey(self.num_layers, 'post_activation')], goal_label.rename(labels_onehot='inputs'))
        for i in reversed(range(0, self.num_layers)):
            activation_d = self.func_activation_d(partial_d).rename(inputs='neurons')
            gradients[mkey(i, 'biases')] = activation_d # times 1, the bias's derivative
            gradients[mkey(i, 'weights')] = np.multiply(activation_d, activations[mkey(i, 'post_activation')])  # times input
            partial_d = xr.dot(activation_d, self.matrices[mkey(i, 'weights')], dims=('neurons', 'neurons'))
            # pre_activation = activations[mkey(i, 'pre_activation')]
        return gradients
        # returns NeuralNet object containing gradients, same format as current NeuralNet but all parameters have additional dimension x m
        
    def train(self, batch_inputs, batch_labels, training_rate=1.0):
        newNet = self
        num_cases = batch_inputs.sizes['cases']
        for i, l in zip(batch_inputs.rolling(batches=1), batch_labels.rolling(batches=1)):
            gradients = self.pass_back(self.pass_forward(i[1].squeeze(dim='batches')), l[1].squeeze(dim='batches'))
            for key in gradients.keys():
                newNet.matrices[key] = np.add(newNet.matrices[key], np.multiply(gradients[key].sum(dim='cases'), training_rate / num_cases))
        return newNet
