"""Basic neural network implementation and helper functions"""
import copy
import numpy as np
import xarray as xr


DIM_LABEL = 'labels'
"""xarray dimension name for onehot vector"""

DIM_CASE = 'cases'
"""xarray dimension name along which inputs in the same batch are stacked"""

DIM_X = 'inputs_x'
"""xarray dimension name for 2-D images in horizontal axis"""

DIM_Y = 'inputs_y'
"""xarray dimension name for 2-D images in vertical axis"""

DIM_IN = 'inputs'
"""xarray dimension name for input dimension of matrix"""

DIM_OUT = 'neurons'
"""xarray dimension name for output dimension of matrix"""

KEY_WEIGHT = 'weights'
"""Dictionary key for weight arrays, used as argument_name in mkey()"""

KEY_BIAS = 'biases'
"""Dictionary key for bias arrays, used as argument_name in mkey()"""

KEY_OUT_PRE = 'pre_activation'
"""Dictionary key for output values without applying activation function,
used as argument_name in mkey()"""

KEY_OUT_POST = 'post_activation'
"""Dictionary key for output values after activation function is applied,
used as argument_name in mkey() """


def mkey(layer, argument_name):
    """Gets key for NeuralNet matrix dict

    Arguments:
        layer {int} -- layer number
        argument_name {str} -- one of the dictionary keys KEY_* in neural_net

    Returns:
        str -- dict key for NeuralNet.matrices
    """

    return 'layer' + str(layer) + ' ' + argument_name


def del_rows(xarray, del_dim, indexes):
    """Deletes rows from xarray object

    Arguments:
        xarray {xarray} -- target array
        del_dim {str} -- name of dimension to index
        indexes {list(int)} -- list of integer indexes to remove along del_dim

    Returns:
        xarray -- array with removed rows
    """

    row_nums = sorted(indexes)
    new_rows = [xarray.isel({del_dim: slice(0, row_nums[0])})]
    for i, j in zip(row_nums[:-1], row_nums[1:]):
        new_rows.append(xarray.isel({del_dim: slice(i + 1, j)}))
    new_rows.append(xarray.isel(
        {del_dim: slice(row_nums[-1] + 1, xarray.sizes[del_dim])}))
    return xr.concat(new_rows, dim=del_dim)


def dict_subset(dictionary, *keywords):
    """Returns subset of a dict with keys that contain contents from *keywords.
        Example: if dictionary contains keys ['foo 1', 'foo 2', 'bar 1'],
        keywords=['1'], returns dict['foo 1', 'bar 1']
        keywords=['bar'], returns dict['bar 1']
        keywords=['foo', '1'], returns dict['foo 1']

    Arguments:
        dictionary {dict} -- dictionary to subset
        *keywords {str} -- patterns to match in keys (uses python keyword 'in')

    Returns:
        dict -- subset of dictionary
    """

    subset = {}
    for key, value in dictionary.items():
        do_add = True
        for word in keywords:
            if word not in key:
                do_add = False
        if do_add:
            subset[key] = value
    return subset


def sigmoid(np_array):
    """Applies sigmoid function to array

    Arguments:
        np_array {np_array} -- target array

    Returns:
        np_array -- sigmoid(np_array)
    """

    return np.divide(1, np.add(1, np.exp(np.multiply(np_array, -1))))


def sigmoid_d(np_array):
    """Applies derivative of sigmoid function to array

    Arguments:
        np_array {np_array} -- target array

    Returns:
        np_array -- sigmoid_derivative(np_array)
    """

    e_a = np.exp(np_array)
    return np.divide(e_a, np.square(np.add(e_a, 1)))


def accuracy(test_onehot, goal_onehot, threshold=0.5):
    """Compares if onehot vectors are equal to a threshold

    Arguments:
        test_onehot {xarray[dims: DIM_IN]} -- first onehot vector,
            onehots encoded on dim=DIM_IN
        goal_onehot {xarray[dims: DIM_LABEL]} -- second onehot vector,
            onehots encoded on dim=DIM_LABEL

    Keyword Arguments:
        threshold {float} -- pivot point at which onehot value is considered 0 or 1:
            0 when value <= threshold, 1 when value > threshold (default: {0.5})

    Returns:
        xarray(bool) -- Array of booleans, one per onehot
    """

    by_threshold = np.logical_and(np.less(test_onehot.max(dim=DIM_IN), threshold),
                                  np.less_equal(goal_onehot.max(dim=DIM_LABEL), 0))
    by_max_value = test_onehot.argmax(
        dim=DIM_IN) == goal_onehot.argmax(dim=DIM_LABEL)
    return np.logical_or(by_threshold, by_max_value)


def accuracy_sum(test_onehot, goal_onehot, threshold=0.5, sum_along_dim=None):
    """Same as accuracy(), but returns number of equal vectors

    Arguments:
        test_onehot {xarray[dims: DIM_IN]} -- first onehot vector,
            onehots encoded on dim=DIM_IN
        goal_onehot {xarray[dims: DIM_LABEL]} -- second onehot vector,
            onehots encoded on dim=DIM_LABEL

    Keyword Arguments:
        threshold {float} -- pivot point at which onehot value is considered 0 or 1:
            0 when value <= threshold, 1 when value > threshold (default: {0.5})
        sum_along_dim {str} -- dimension along which to find sums,
            returns aggregate sum by default (default: {None})

    Returns:
        xarray(int) -- number of matching onehot vectors, or array of dim=sum_along_dim
    """

    return accuracy(test_onehot, goal_onehot, threshold=threshold).sum(
        dim=sum_along_dim)


def cost_mean_squared(test_onehot, goal_onehot, sum_along_dim=None):
    """Mean squared error of two vectors

    Arguments:
        test_onehot {xarray} -- first vector
        goal_onehot {xarray} -- second vector

    Keyword Arguments:
        sum_along_dim {str} -- dimension along which to find means,
            returns aggregate mean by default (default: {None})

    Returns:
        xarray(float) -- mean squared error as float, or array of dim=sum_along_dim
    """

    return np.square(np.subtract(
        test_onehot, goal_onehot.rename({DIM_LABEL: DIM_IN}))) \
        .mean(dim=sum_along_dim)


class NeuralNet(object):
    """Simple neural network: self.num_layers is the number of layers,
        self.matrices is a dict containing all parameters, use neural_net.mkey()
        with neural_net.KEY_* to find individual arrays of weights and biases.
        Arrays are xarrays where KEY_IN is the dimension for inputs and KEY_OUT is the
        dimension for outputs."""

    def __init__(self, layer_sizes, func_fill=lambda x: np.random.randn(*x),
                 func_activation=sigmoid, func_activation_d=sigmoid_d):
        """Creates NeuralNet object with dict in self.matrices containing 2 xarrays
            per layer at KEY_WEIGHT and KEY_BIAS. KEY_WEIGHT xarray has dim=inputs, neurons
            KEY_BIAS has dim=neurons, self.num_layers is number of layers

        Arguments:
            layer_sizes {tuple(int)} -- Tuple of layer sizes
                containing (num_layers + 1) ints,
                first int is size of inputs, last int is size of outputs

        Keyword Arguments:
            func_fill {function(tuple): np_array} -- Function which returns a np_array of
                dimensions (tuple) with values (default: {lambdax:np.random.randn(*x)})
            func_activation {function(np_array): np_array} -- Activation function
                (default: {sigmoid})
            func_activation_d {function(np_array): np_array} -- Derivative of activation
                function (default: {sigmoid_d})

        Raises:
            ValueError -- Not enough layers: layer_sizes must contain at least 2 values

        """

        if len(layer_sizes) < 2:
            raise ValueError(
                'Not enough layers: layer_sizes must contain at least 2 values')

        self.num_layers = len(layer_sizes) - 1
        self.func_activation = func_activation
        self.func_activation_d = func_activation_d
        self.matrices = {}
        for i, l_size, next_l in zip(range(self.num_layers),
                                     layer_sizes[:-1], layer_sizes[1:]):
            self.matrices[mkey(i, KEY_WEIGHT)] = xr.DataArray(
                func_fill((l_size, next_l)), dims=(DIM_IN, DIM_OUT))
            self.matrices[mkey(i, KEY_BIAS)] = xr.DataArray(
                func_fill((next_l,)), dims=(DIM_OUT))

    def pass_forward(self, inputs, func_normalize=lambda x: x):
        """Applies NeuralNet to inputs

        Arguments:
            inputs {xarray[dims: DIM_IN]} -- xarray with dimension
                DIM_IN, same size as DIM_IN in self.matrices[mkey(0, KEY_WEIGHT)]

        Keyword Arguments:
            func_normalize {function(np_array): np_array} -- function to
                apply to inputs before passing through neural network (default: {lambdax:x})

        Raises:
            ValueError -- Missing dimension DIM_IN in inputs
            ValueError -- Size of inputs dimension DIM_IN does not match layer 0 size

        Returns:
            dict(xarray[dims: KEY_OUT_PRE or KEY_OUT_POST]) -- Dictionary containing
                (num_layers + 1) xarrays of intermediate layer outputs,
                with 2 matrices: KEY_OUT_PRE (without activation function applied)
                and KEY_OUT_POST (with activation function applied)
        """

        if not DIM_IN in inputs.dims:
            raise ValueError('Missing dimension \'' +
                             DIM_IN + '\' in inputs')
        tsize = inputs.sizes[DIM_IN]
        msize = self.matrices[mkey(0, KEY_WEIGHT)].sizes[DIM_IN]
        if tsize != msize:
            raise ValueError('Size of \'' + DIM_IN + '\'=' + str(tsize) +
                             ' does not match layer 0 size: ' + str(msize))

        # ugly hack: remove coordinates for dimension 'inputs' if coordinates present
        if DIM_IN in inputs.coords:
            inputs = inputs.reset_index(DIM_IN, drop=True)
        activations = {mkey(0, KEY_OUT_PRE): inputs,
                       mkey(0, KEY_OUT_POST): func_normalize(inputs)}
        for i in range(self.num_layers):
            pre_activation = np.add(xr.dot(activations[mkey(i, KEY_OUT_POST)],
                                           self.matrices[mkey(i, KEY_WEIGHT)],
                                           dims=(DIM_IN)), self.matrices[mkey(i, KEY_BIAS)])
            activations[mkey(i+1, KEY_OUT_PRE)
                        ] = pre_activation.rename({DIM_OUT: DIM_IN})
            activations[mkey(i + 1, KEY_OUT_POST)] = self.func_activation(
                pre_activation).rename({DIM_OUT: DIM_IN})
        return activations

    def pass_forward_output_only(self, inputs, func_normalize=lambda x: x):
        """Same as self.pass_forward() but only returns the output layer's activations

        Arguments:
            inputs {xarray[dims: DIM_IN]} -- xarray with dimension
                DIM_IN, same size as DIM_IN in self.matrices[mkey(0, KEY_WEIGHT)]

        Keyword Arguments:
            func_normalize {function(np_array): np_array} -- function to
                apply to inputs before passing through neural network (default: {lambdax:x})

        Returns:
            xarray[dims: KEY_OUT_POST] -- xarray of output layer activations
        """

        return self.pass_forward(inputs, func_normalize=func_normalize)[mkey(
            self.num_layers, KEY_OUT_POST)]

    def pass_back(self, activations, goal_label,
                  func_loss_d=lambda output_v, goal_v: np.subtract(goal_v, output_v)):
        """Backpropagates activations to get gradients

        Arguments:
            activations {dict(xarray[dims: KEY_OUT_PRE or KEY_OUT_POST])} -- dict which is
                the return value of self.pass_forward()
            goal_label {xarray[dims: DIM_LABEL]} -- array of onehot vectors encoded
                along dim=DIM_LABEL

        Keyword Arguments:
            func_loss_d {function(xarray, xarray)} -- derivative of loss function,
                returns gradients of dim=DIM_OUT same size as final layer outputs
                (default: {lambda output_v, goal_v: np.subtract(goal_v, output_v)})

        Returns:
            dict(xarray[dims: DIM_IN, DIM_OUT]) -- dict of gradients, containing
                xarrays in same format as self.matrices
        """

        gradients = {}
        partial_d = func_loss_d(activations[mkey(self.num_layers, KEY_OUT_POST)],
                                goal_label.rename({DIM_LABEL: DIM_IN}))
        for i in reversed(range(self.num_layers)):
            partial_d = np.multiply(partial_d, self.func_activation_d(
                activations[mkey(i+1, KEY_OUT_PRE)])).rename({DIM_IN: DIM_OUT})
            # times 1, the bias's derivative
            gradients[mkey(i, KEY_BIAS)] = partial_d
            gradients[mkey(i, KEY_WEIGHT)] = np.multiply(
                partial_d, activations[mkey(i, KEY_OUT_POST)])  # times input
            partial_d = xr.dot(
                partial_d, self.matrices[mkey(i, KEY_WEIGHT)], dims=(DIM_OUT))
        return gradients

    def train_yield(self, train_inputs, train_labels, batch=1, rate=1.0, deep_copy=True):
        """Trains NeuralNet on a batch of inputs, iterating through each batch

        Arguments:
            train_inputs {iter(xarray[dims: DIM_CASE, DIM_IN])} -- multiple inputs
                along DIM_CASE, of same format as inputs to self.pass_forward(), 
            train_labels {iter(xarray[dims: DIM_CASE, DIM_LABEL])} -- multiple inputs
                along DIM_CASE, of same format as inputs to self.pass_back()

        Keyword Arguments:
            batch {int} -- number of inputs per batch, net is trained once per batch
                (default: {1})
            rate {float} -- training rate by which gradients are multiplied
                (default: {1.0})
            deep_copy {bool} -- whether to return a new object or modify existing object
                (default: {False})

        Yields:
            tuple(NeuralNet, int, int, xarray, xarray) -- tuple of NeuralNet in training, number
                of input cases trained, number of batches trained, inputs, and labels used in batch
        """

        new_net = self
        if deep_copy:
            new_net = copy.deepcopy(self)
        num_batches = int(train_inputs.sizes[DIM_CASE] / batch)
        batch_inputs = train_inputs.groupby_bins(DIM_CASE, num_batches)
        batch_labels = train_labels.groupby_bins(DIM_CASE, num_batches)
        for batch_num, inputs_t, labels_t in zip(range(1, num_batches + 1), batch_inputs, batch_labels):
            inputs = inputs_t[1]  # unpacking tuple returned by groupby object
            labels = labels_t[1]  # unpacking tuple returned by groupby object
            gradients = new_net.pass_back(new_net.pass_forward(inputs), labels)
            for key in gradients.keys():
                new_net.matrices[key] = np.add(new_net.matrices[key], np.multiply(
                    gradients[key].mean(dim=DIM_CASE), rate))
            yield new_net, batch_num * batch, batch_num, inputs, labels

    def train(self, train_inputs, train_labels, batch_size=1, training_rate=1.0, deep_copy=True):
        """Same as self.train_yield(), but returns only the final result.

        Arguments:
            train_inputs {iter(xarray[dims: DIM_CASE, DIM_IN])} -- multiple inputs
                along DIM_CASE, of same format as inputs to self.pass_forward(), 
            train_labels {iter(xarray[dims: DIM_CASE, DIM_LABEL])} -- multiple inputs
                along DIM_CASE, of same format as inputs to self.pass_back()

        Keyword Arguments:
            batch_size {int} -- number of inputs per batch, net is trained once per batch
                (default: {1})
            training_rate {float} -- training rate by which gradients are multiplied
                (default: {1.0})
            deep_copy {bool} -- whether to return a new object or modify existing object
                (default: {False})

        Returns:
            NeuralNet -- trained NeuralNet
        """

        new_net = None
        for new_net, _, _, _, _ in self.train_yield(train_inputs, train_labels, batch=batch_size,
                                      rate=training_rate, deep_copy=deep_copy):
            pass
        return new_net

    def delete_neurons(self, neuron_indexes_per_layer, activations=None):
        """Removes neurons from the neural network

        Arguments:
            neuron_indexes_per_layer {list(list(int))} -- List of lists of indexes,
                one list per layer in order from input to output, int indexes of
                neurons to remove

        Keyword Arguments:
            activations {dict(xarray[dims: KEY_OUT_PRE or KEY_OUT_POST])} --
                activations from self.pass_forward(), if present
                the mean of the activations of the deleted neurons
                are used to adjust the bias of the next layer to minimize
                change to the network (default: {None})

        Returns:
            NeuralNet -- Copy of self with neurons removed
        """

        new_net = copy.deepcopy(self)
        for i, neurons in zip(range(self.num_layers), neuron_indexes_per_layer):
            if not neurons:
                continue
            new_net.matrices[mkey(i, KEY_WEIGHT)] = del_rows(
                new_net.matrices[mkey(i, KEY_WEIGHT)], DIM_OUT, neurons)
            new_net.matrices[mkey(i, KEY_BIAS)] = del_rows(
                new_net.matrices[mkey(i, KEY_BIAS)], DIM_OUT, neurons)
            if i + 1 < self.num_layers:
                if not activations is None:
                    for neuron in neurons:
                        bias = activations[mkey(i + 1, KEY_OUT_POST)].isel(
                            {DIM_IN: neuron}).mean(dim=DIM_CASE)
                        bias_adjust = np.multiply(
                            bias, new_net.matrices[mkey(i + 1, KEY_WEIGHT)].isel({DIM_IN: neuron}))
                        new_net.matrices[mkey(i + 1, KEY_BIAS)] = np.add(
                            new_net.matrices[mkey(i + 1, KEY_BIAS)], bias_adjust)
                new_net.matrices[mkey(i + 1, KEY_WEIGHT)] = del_rows(
                    new_net.matrices[mkey(i + 1, KEY_WEIGHT)], DIM_IN, neurons)
        return new_net

    def __repr__(self):
        """Quick and dirty debugging string

        Returns:
            str -- Dimensions and size of every matrix in self.matrices
        """

        string_out = 'Neural net '
        for key, value in self.matrices.items():
            string_out += key + ' ' + repr(value.sizes)
        return string_out
