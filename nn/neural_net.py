import numpy as np
import xarray as xr
import copy

KEY_NET = 'net'
KEY_WEIGHT = 'weights'
KEY_BIAS = 'biases'
KEY_INPUT = 'inputs'
KEY_OUTPUT = 'neurons'
KEY_OUT_PRE = 'pre_activation'
KEY_OUT_POST = 'post_activation'
KEY_LABEL = 'labels'
KEY_CASE = 'cases'

def mkey(layer, argument_name):
    return 'layer' + str(layer) + ' ' + argument_name
    
def del_rows(xarray, del_dim, indexes):
    row_nums = sorted(indexes)
    new_rows = [xarray.isel({del_dim: slice(0, row_nums[0])})]
    for i, j in zip(row_nums[:-1], row_nums[1:]):
        new_rows.append(xarray.isel({del_dim: slice(i + 1, j)}))
    new_rows.append(xarray.isel({del_dim: slice(row_nums[-1] + 1, xarray.sizes[del_dim])}))
    return xr.concat(new_rows, dim=del_dim)

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

def accuracy_sum(test_onehot, goal_onehot, sum_along_dim=None):
    return accuracy(test_onehot, goal_onehot).sum(dim=sum_along_dim)

def accuracy(test_onehot, goal_onehot, threshold=0.5):
    by_threshold = np.logical_and(np.less(test_onehot.max(dim=KEY_INPUT), threshold),
        np.less_equal(goal_onehot.max(dim=KEY_LABEL), 0))
    by_max_value = test_onehot.argmax(dim=KEY_INPUT) == goal_onehot.argmax(dim=KEY_LABEL)
    return np.logical_or(by_threshold, by_max_value)

def cost_mean_squared(test_onehot, goal_onehot, sum_along_dim=None):
    return np.square(np.subtract(test_onehot, goal_onehot.rename({KEY_LABEL: KEY_INPUT}))).mean(dim=sum_along_dim)

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
        
        # ugly hack: remove coordinates for dimension 'inputs' if coordinates present
        if KEY_INPUT in training_inputs.coords:
            training_inputs = training_inputs.reset_index(KEY_INPUT, drop=True)
        activations = {mkey(0, KEY_OUT_PRE): training_inputs,
            mkey(0, KEY_OUT_POST): func_normalize(training_inputs)}
        for i in range(self.num_layers):
            pre_activation = np.add(xr.dot(activations[mkey(i, KEY_OUT_POST)], self.matrices[mkey(i, KEY_WEIGHT)],
                dims=(KEY_INPUT, KEY_INPUT)), self.matrices[mkey(i, KEY_BIAS)])
            activations[mkey(i+1, KEY_OUT_PRE)] = pre_activation.rename({KEY_OUTPUT: KEY_INPUT})
            activations[mkey(i+1, KEY_OUT_POST)] = self.func_activation(pre_activation).rename({KEY_OUTPUT: KEY_INPUT})
        return activations
        # returns (num_layers + 1) layers with 2 matrices each, with and without activation function applied

    def output_only(self, pass_forward):
        return pass_forward[mkey(self.num_layers, KEY_OUT_POST)]

    # activations are the return value of pass_forward()
    def pass_back(self, activations, goal_label, func_loss_d=lambda output_v, goal_v: np.subtract(goal_v, output_v)):
        gradients = {}
        partial_d = func_loss_d(activations[mkey(self.num_layers, KEY_OUT_POST)], goal_label.rename({KEY_LABEL: KEY_INPUT}))
        for i in reversed(range(self.num_layers)):
            partial_d = np.multiply(partial_d, self.func_activation_d(activations[mkey(i+1, KEY_OUT_PRE)])).rename({KEY_INPUT: KEY_OUTPUT})
            gradients[mkey(i, KEY_BIAS)] = partial_d # times 1, the bias's derivative
            gradients[mkey(i, KEY_WEIGHT)] = np.multiply(partial_d, activations[mkey(i, KEY_OUT_POST)])  # times input
            partial_d = xr.dot(partial_d, self.matrices[mkey(i, KEY_WEIGHT)], dims=(KEY_OUTPUT, KEY_OUTPUT))
            # pre_activation = activations[mkey(i, KEY_OUT_PRE)]
        return gradients
        # returns NeuralNet object containing gradients, same format as current NeuralNet but all parameters have additional dimension 'cases'
        
    def train(self, batch_inputs, batch_labels, training_rate=1.0, deep_copy=False):
        items = None
        for items in self.train_yield(batch_inputs, batch_labels, training_rate):
            pass
        return items[0]

    # iterable list of (label, xarray) for batch_inputs and batch_labels (use xr.groupby() to generate)
    def train_yield(self, batch_inputs, batch_labels, training_rate=1.0, deep_copy=True):
        new_net = self
        if deep_copy:
            new_net = copy.deepcopy(self)
        for i, l in zip(batch_inputs, batch_labels):
            inputs = i[1]
            labels = l[1]
            gradients = new_net.pass_back(new_net.pass_forward(inputs), labels)
            for key in gradients.keys():
                new_net.matrices[key] = np.add(new_net.matrices[key],
                    np.multiply(gradients[key].mean(dim=KEY_CASE), training_rate))
            yield new_net, inputs, labels
    
    def delete_neurons(self, activations, neuron_indexes_per_layer):
        new_net = copy.deepcopy(self)
        for l, neurons in zip(range(self.num_layers), neuron_indexes_per_layer):
            if len(neurons) == 0:
                continue
            new_net.matrices[mkey(l, KEY_WEIGHT)] = del_rows(
                new_net.matrices[mkey(l, KEY_WEIGHT)], KEY_OUTPUT, neurons)
            new_net.matrices[mkey(l, KEY_BIAS)] = del_rows(
                new_net.matrices[mkey(l, KEY_BIAS)], KEY_OUTPUT, neurons)
            if l + 1 < self.num_layers:
                for n in neurons:
                    bias = activations[mkey(l + 1, KEY_OUT_POST)].isel(
                        inputs=n).mean(dim=KEY_CASE)
                    bias_adjust = np.multiply(bias,
                        new_net.matrices[mkey(l + 1, KEY_WEIGHT)].isel(inputs=n))
                    new_net.matrices[mkey(l + 1, KEY_BIAS)] = np.add(
                        new_net.matrices[mkey(l + 1, KEY_BIAS)], bias_adjust)
                new_net.matrices[mkey(l + 1, KEY_WEIGHT)] = del_rows(
                    new_net.matrices[mkey(l + 1, KEY_WEIGHT)], KEY_INPUT, neurons)
        return new_net

    def __repr__(self):
        string_out = 'Neural net '
        for key, value in self.matrices.items():
            string_out += key + ' ' + repr(value.sizes)
        return string_out