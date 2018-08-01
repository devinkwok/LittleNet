import numpy as np
import xarray as xr

KEY_NET = 'net'
KEY_WEIGHT = 'weights'
KEY_BIAS = 'biases'
KEY_INPUT = 'inputs'
KEY_OUTPUT = 'neurons'
KEY_OUT_PRE = 'pre_activation'
KEY_OUT_POST = 'post_activation'
KEY_BATCH = 'batches'
KEY_LABEL = 'labels'
KEY_CASE = 'cases'

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
    labels = xr.concat(onehot_array, dim=KEY_LABEL).astype(float)
    return labels.transpose(*label_xarray.dims, KEY_LABEL)

def sigmoid(np_array):
    return np.divide(1, np.add(1, np.exp(np.multiply(np_array, -1))))

def sigmoid_d(np_array):
    e_a = np.exp(np_array)
    return np.divide(e_a, np.square(np.add(e_a, 1)))

# tensor: an xarray.DataSet containing multiple instances of xarray.DataArray
# parameters are of format 'name': dimensions, where dimension = 0 is a single variable per layer
# dimension = 1 is layer size, dimension = 2 is previous layer size x layer size
class NeuralNet(object):

    # layer_sizes is a tuple or list containing num_layers + 1 integers indicating size of each layer
    def __init__(self, layer_sizes, func_fill=lambda x: np.random.randn(*x), func_activation=sigmoid, func_activation_d=sigmoid_d):
        if len(layer_sizes) < 2:
            raise ValueError('Not enough layers: layer_sizes must contain at least 2 values')
        
        self.num_layers = len(layer_sizes) - 1
        self.func_activation = func_activation
        self.func_activation_d = func_activation_d
        self.matrices = {}
        for i, l_size, next_l in zip(range(self.num_layers), layer_sizes[:-1], layer_sizes[1:]):
            self.matrices[mkey(i, KEY_WEIGHT)] = xr.DataArray(func_fill((l_size, next_l)), dims=(KEY_INPUT, KEY_OUTPUT))
            self.matrices[mkey(i, KEY_BIAS)] = xr.DataArray(func_fill((next_l,)), dims=(KEY_OUTPUT))
        return None

    # training_inputs is an xarray with dimension KEY_INPUT, with same size as dimension KEY_INPUT in self.matrices[mkey(0, KEY_WEIGHT)]
    def pass_forward(self, training_inputs, func_normalize=lambda x: x):
        if not KEY_INPUT in training_inputs.dims:
            raise ValueError('Missing dimension \'' + KEY_INPUT + '\' in training_inputs')
        tsize = training_inputs.sizes[KEY_INPUT]
        msize = self.matrices[mkey(0, KEY_WEIGHT)].sizes[KEY_INPUT]
        if tsize != msize:
            raise ValueError('Size of \'' + KEY_INPUT + '\'=' + str(tsize) + ' does not match layer 0 size: ' + str(msize))
        
        activations = {mkey(0, KEY_OUT_PRE): training_inputs,
            mkey(0, KEY_OUT_POST): func_normalize(training_inputs)}
        for i in range(0, self.num_layers):
            pre_activation = np.add(xr.dot(activations[mkey(i, KEY_OUT_POST)], self.matrices[mkey(i, KEY_WEIGHT)],
                dims=(KEY_INPUT, KEY_INPUT)), self.matrices[mkey(i, KEY_BIAS)])
            activations[mkey(i+1, KEY_OUT_PRE)] = pre_activation.rename(neurons=KEY_INPUT)
            activations[mkey(i+1, KEY_OUT_POST)] = self.func_activation(pre_activation).rename(neurons=KEY_INPUT)
        return activations
        # returns (num_layers + 1) layers with 2 matrices each, with and without activation function applied

    def output_only(self, pass_forward):
        return pass_forward[mkey(self.num_layers, KEY_OUT_POST)]

    # activations are the return value of pass_forward()
    def pass_back(self, activations, goal_label, func_loss_d=lambda output_v, goal_v: np.subtract(goal_v, output_v)):
        gradients = {}
        partial_d = func_loss_d(activations[mkey(self.num_layers, KEY_OUT_POST)], goal_label.rename(labels=KEY_INPUT))
        for i in reversed(range(0, self.num_layers)):
            activation_d = np.multiply(partial_d, self.func_activation_d(activations[mkey(i+1, KEY_OUT_PRE)])).rename(inputs=KEY_OUTPUT)
            gradients[mkey(i, KEY_BIAS)] = activation_d # times 1, the bias's derivative
            gradients[mkey(i, KEY_WEIGHT)] = np.multiply(activation_d, activations[mkey(i, KEY_OUT_POST)])  # times input
            partial_d = xr.dot(activation_d, self.matrices[mkey(i, KEY_WEIGHT)], dims=(KEY_OUTPUT, KEY_OUTPUT))
            # pre_activation = activations[mkey(i, KEY_OUT_PRE)]
        return gradients
        # returns NeuralNet object containing gradients, same format as current NeuralNet but all parameters have additional dimension x m
        
    def train(self, batch_inputs, batch_labels, training_rate=1.0):
        items = None
        for items in self.train_yield(batch_inputs, batch_labels, training_rate):
            pass
        return items[0]

    def train_yield(self, batch_inputs, batch_labels, training_rate=1.0):
        if not KEY_BATCH in batch_inputs.dims:
            raise ValueError('Missing dimension \'' + KEY_BATCH + '\' in batch_inputs')
        if not KEY_BATCH in batch_labels.dims:
            raise ValueError('Missing dimension \'' + KEY_BATCH + '\' in batch_labels')
        
        newNet = self
        num_cases = batch_inputs.sizes[KEY_CASE]
        for i, l in zip(batch_inputs.rolling(batches=1), batch_labels.rolling(batches=1)):
            inputs = i[1].squeeze(dim=KEY_BATCH)
            labels = l[1].squeeze(dim=KEY_BATCH)
            gradients = self.pass_back(self.pass_forward(inputs), labels)
            for key in gradients.keys():
                newNet.matrices[key] = np.add(newNet.matrices[key], np.multiply(gradients[key].sum(dim=KEY_CASE), training_rate / num_cases))
            yield newNet, inputs, labels
