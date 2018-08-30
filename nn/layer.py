from abc import ABCMeta, abstractmethod
import numpy as np
import xarray as xr

KEY_LOSS = 'loss'
KEY_IN = 'inputs'
KEY_OUT = 'outputs'
KEY_OUT_PRE = 'pre_activation'
KEY_W = 'weights'
KEY_B = 'biases'

def sigmoid(np_array):
    return np.divide(1, np.add(1, np.exp(np.multiply(np_array, -1))))

def sigmoid_d(np_array):
    e_a = np.exp(np_array)
    return np.divide(e_a, np.square(np.add(e_a, 1)))

def loss_mean_squared_d(expected, actual)
    diff = np.square(np.subtract(actual, expected))
    c = diff.sum() - diff
    d = 2 * (actual - expected) + np.square(expected)
    num = 0 #TODO number of summed inputs
    return np.divide(d + c, num)

def loss_mean_squared(expected, actual, sum_along_dim=None):
    return np.square(np.subtract(actual, expected)).mean(dim=sum_along_dim)


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def pass_forward(self, inputs, past_inputs):
        raise NotImplementedError()

    @abstractmethod
    def pass_back(self, gradient, past_inputs, past_outputs):
        raise NotImplementedError()

    @property
    @abstractmethod
    def coords()
        raise NotImplementedError()


class InputLayer(Layer):

    def __init__(self, coords, func_normalize=lambda x: x):
        self.coords = coords
        self.layer_next = None  # NOTE: always create another layer with this object as layer_prev
        self.func_normalize = func_normalize

    def pass_forward(self, inputs, past_inputs=None):
        return func_normalize(inputs), []

    def pass_back(self, gradient, past_inputs, past_outputs):
        return layer_next.pass_back(gradient, past_inputs, past_outputs)


class OutputLayer(Layer):

    def __init__(self, layer_prev, func_loss=loss_mean_squared, func_loss_d=lambda expected, actual: np.subtract(expected, actual)):
        self.layer_prev = layer_prev
        self.func_loss = func_loss
        self.func_loss_d = func_loss_d

    def pass_forward(self, inputs, past_inputs=None):
        outputs, previous = layer_prev.pass_forward(inputs, past_inputs)
        return outputs, previous.append(KEY_OUT: outputs)

    def pass_back(self, gradient, past_inputs, past_outputs=None):
        outputs = past_inputs.pop()
        return func_loss_d(gradient, outputs[KEY_OUT]), past_inputs, []


class FullyConnectedLayer(Layer):

    def __init__(self, coords, layer_prev, func_fill=lambda x: np.random.randn(*x), func_activation=sigmoid, func_activation_d=sigmoid_d):
        sizes = [len(c) for c in coords.values()]
        layer_prev.layer_next = self
        self.layer_prev = layer_prev                # previous layer to get input coordinates from
        self.layer_next = None                      # NOTE: always create another layer with this object as layer_prev
        self.coords = coords                        # dict of dimension_name: coordinates (len = size of dim)
        self.func_activation = func_activation      # activation function
        self.func_activation_d = func_activation_d  # derivative of activation function
        self.sizes = sizes
        self.matrices[KEY_W] = xr.DataArray(func_fill([*layer_prev.sizes, *sizes]),
            dims=[*layer_prev.coords, *coords], coords={**layer_prev.coords, **coords})
        self.matrices[KEY_B] = xr.DataArray(func_fill(sizes),
            dims=[*coords], coords=coords)

    def pass_forward(self, inputs, past_inputs):
        i, p = self.layer_prev.pass_forward(inputs, past_inputs)
        pre_activation = xr.dot(i, self.m_weights,
            dims=([*self.layer_prev.coords]).add(self.m_biases)
        activation = self.func_activation(pre_activation)
        return activation, p.append({KEY_IN: i, KEY_OUT_PRE: pre_activation})

    def pass_back(self, gradient, past_inputs, past_outputs):
        grad, inputs, p = self.layer_next.pass_back(gradient, past_inputs, past_outputs)
        i = inputs.pop()
        grad_b = np.multiply(grad, self.func_activation_d(i[KEY_OUT_PRE]))
        grad_w = np.multiply(grad_b, i[KEY_IN])
        grad_next =  xr.dot(grad_b, self.m_weights, dims=[*coords])
        return grad_next, inputs, p.append({KEY_B: grad_b, KEY_W: grad_w})


class ConvolutionLayer(Layer):

    def pass_forward(self, inputs, past_inputs):
        pass #TODO
        
    def pass_back(self, gradient, past_inputs, past_outputs):
        pass #TODO

def test():
    NUM_CASES = 5
    NUM_INPUTS = 10
    NUM_OUTPUTS = 2
    KEY_INPUTS = 'inputs'
    KEY_OUTPUTS = 'neurons'
    KEY_LABELS = 'labels'
    KEY_CASES = 'cases'

    layer0 = InputLayer({KEY_INPUTS: NUM_INPUTS})
    layer1 = FullyConnectedLayer(layer0, {KEY_OUTPUTS: NUM_OUTPUTS})
    layer2 = OutputLayer(layer1)

    input_coords = {KEY_CASES: np.arange(NUM_CASES), KEY_INPUTS: np.arange(NUM_INPUTS)}
    inputs = xr.DataArray(np.ones((NUM_CASES,NUM_INPUTS)), dims=[*input_coords], coords=input_coords)
    label_coords = {KEY_CASES: np.arange(NUM_CASES), KEY_LABELS: np.arange(NUM_OUTPUTS)}
    labels = xr.DataArray(np.ones((NUM_CASES, NUM_OUTPUTS)), dims=[*label_coords], coords=label_coords)

    #TODO
    out, outputs = layer2.pass_forward(inputs)
    gradients = layer0.pass_back(labels, outputs)


if __name__ == '__main__':
    test()