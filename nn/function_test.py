import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import neural_net as nn
from neural_net import mkey
from nn_old import pass_back, create_tensor
import utility

# should draw a 2D surface with points of 2 colours, surface should orient to match colors of points
def test_2in_1out():
    def func_probability(xarray_input):
        STEEPNESS = 100
        W = 1 * STEEPNESS
        B = 0.7 * STEEPNESS
        return np.divide(1, np.add(1, np.exp(np.multiply(-1, np.subtract(np.multiply(xarray_input.isel(inputs=0), W), B)))))
    def plot_tensor_2D(tensor):
        NUM = 50
        x = np.linspace(0, 1, num=NUM)
        y = np.linspace(0, 1, num=NUM)
        xgrid, ygrid = np.meshgrid(x, y)
        input_grid = xr.DataArray(xr.concat([xr.DataArray(xgrid, dims=('input1', 'input2')), xr.DataArray(ygrid, dims=('input1', 'input2'))], dim='inputs'))
        input_grid = input_grid.assign_coords(input1=x, input2=y)
        outputs = tensor.output_only(tensor.pass_forward(input_grid))
        outputs.plot()
    NUM_BATCHES = 100
    NUM_CASES = 100
    NUM_DIMS = 2
    inputs = xr.DataArray(np.random.rand(NUM_BATCHES*NUM_CASES, NUM_DIMS), dims=('cases', 'inputs'))
    labels = xr.DataArray(np.greater(func_probability(inputs), np.random.rand(NUM_BATCHES * NUM_CASES)), dims=('cases'))
    labels = labels.expand_dims('labels')
    net = nn.NeuralNet((2, 1))
    count = 0
    for new_net, i, l in net.train_yield(inputs.groupby_bins('cases', NUM_BATCHES), labels.groupby_bins('cases', NUM_BATCHES), training_rate=5.0):
        if count % 20 == 0:
            plot_tensor_2D(new_net)
            plt.scatter(i.isel(inputs=0), i.isel(inputs=1), c=l.isel(labels=0))
            plt.show()
        count += 1

# should draw a 1D function and points, points should orient to match contour of function
def test_1in_1out():
    def target_function_1D(x):
        return np.abs(np.sin(np.multiply(x, 4)))  # inputs from 0 to 1, outputs from 0 to 1
    NUM_BATCHES = 4000
    NUM_CASES = 10
    SIZES = (1, 10, 1)
    net = nn.NeuralNet(SIZES)
    inputs = np.divide(np.arange(NUM_CASES * NUM_BATCHES), NUM_CASES * NUM_BATCHES)
    labels = target_function_1D(inputs)
    indexes = np.arange(NUM_CASES * NUM_BATCHES)
    np.random.shuffle(indexes)
    inputs = xr.DataArray(inputs[indexes].reshape((NUM_BATCHES*NUM_CASES, 1)), dims=('cases', 'inputs'))
    labels = xr.DataArray(labels[indexes].reshape((NUM_BATCHES*NUM_CASES, 1)), dims=('cases', 'labels'))
    num = 0
    for new_net, i, l in net.train_yield(inputs.groupby_bins('cases', NUM_BATCHES), labels.groupby_bins('cases', NUM_BATCHES), training_rate=3.0):
        if num % 500 == 0:
            x = np.linspace(0, 1, num=50)
            xgrid = xr.DataArray(x, dims=('cases'), coords={'cases': x}).expand_dims('inputs')
            net_outputs = new_net.output_only(new_net.pass_forward(xgrid)).squeeze()
            goal_outputs = target_function_1D(xgrid)
            plt.scatter(x, net_outputs)
            plt.scatter(i.squeeze().rename(cases='inputs'), l.squeeze().rename(cases='outputs'))
            goal_outputs.plot()
            net_outputs.plot()
            plt.show()
        num += 1
    NUM_TESTS = 100
    test_inputs = np.divide(np.arange(NUM_TESTS), NUM_TESTS)
    xarray_inputs = xr.DataArray(test_inputs.reshape((NUM_TESTS, 1)), dims=('cases', 'inputs'))
    goal_outputs = target_function_1D(xarray_inputs).squeeze()
    net_outputs = net.pass_forward(xarray_inputs)[mkey(len(SIZES)-1, 'post_activation')].squeeze()
    goal_outputs.assign_coords(cases=test_inputs)
    net_outputs.assign_coords(cases=test_inputs)
    goal_outputs.plot()
    net_outputs.plot()

# should output arrays of weights and gradients with identical values
def test_compare():
    SIZES = (3,2,1)
    net = nn.NeuralNet(SIZES, func_fill=np.ones)
    inputs = xr.DataArray(np.ones((3,)), dims=('inputs'))
    activations = net.pass_forward(inputs)
    pre_activations = [activations[mkey(i, 'pre_activation')].values.reshape((1, SIZES[i])) for i in range(len(SIZES))]
    gradient = np.subtract(np.ones((1,)), activations[mkey(len(SIZES) - 1, 'post_activation')]).values.reshape((1,1))
    old_output = pass_back(pre_activations, gradient, create_tensor(SIZES, fillFunction=np.ones))
    output = net.pass_back(activations, xr.DataArray([1], dims=('labels')))
    for layer in range(len(SIZES) - 1):
        print('LAYER', layer, 'WEIGHTS:\nold:', old_output['w'][layer].tolist(),
            '\nnew:', output[mkey(layer, 'weights')].values.tolist(),
            '\nLAYER', layer, 'BIASES:\nold:', old_output['b'][layer].tolist(),
            '\nnew:', output[mkey(layer, 'biases')].values.tolist(), '\n')

def train_and_save_nn(net, inputs, labels, filename='/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained.pyc', batch_size=10, test_fraction=0.1, report_frequency=100):
    num_cases = inputs.sizes['cases']
    num_tests = int(num_cases * test_fraction)
    num_batches = (num_cases - num_tests) / batch_size
    test_inputs = inputs.isel(cases=slice(num_tests))
    test_labels = labels.isel(cases=slice(num_tests))
    train_inputs = inputs.isel(cases=slice(num_tests, num_cases)).groupby_bins('cases', num_batches)
    train_labels = labels.isel(cases=slice(num_tests, num_cases)).groupby_bins('cases', num_batches)
    count = 0
    last_net = net
    for newNet, i, l in net.train_yield(train_inputs, train_labels, training_rate=3.0):
        if count % report_frequency == 0:
            test_outputs = newNet.output_only(newNet.pass_forward(test_inputs))
            print('number:', count * batch_size,
                'accuracy:', nn.accuracy_sum(test_outputs, test_labels).values / num_tests,
                'loss:', nn.cost_mean_squared(test_outputs, test_labels).values)
        count += 1
        last_net = newNet
    utility.write_object(last_net, filename)
    return last_net

# accuracy should rise to approx. 90 (depends on random initialization), loss should reduce to < 0.02
def test_mnist():
    inputs = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    labels = nn.make_onehot(labels, np.arange(10))

    indexes = np.arange(60000)
    net = nn.NeuralNet((784, 30, 10))
    for x in range(10):
        np.random.shuffle(indexes)
        inputs = inputs.isel(cases=indexes)
        labels = labels.isel(cases=indexes)
        net = train_and_save_nn(net, inputs, labels)
    
def test_mnist_with_random_images():
    inputs = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    labels = nn.make_onehot(labels, np.arange(10))

    random_inputs = utility.shuffle_pixels(inputs).isel(cases=slice(30000)).assign_coords(cases=np.arange(60000,90000))
    random_labels = nn.make_onehot(xr.DataArray(np.full((30000), None), dims=('cases'), coords={'cases': np.arange(60000,90000)}), np.arange(10))
    inputs = xr.concat([inputs, random_inputs], dim='cases')
    labels = xr.concat([labels, random_labels], dim='cases')

    indexes = np.arange(90000)
    net = nn.NeuralNet((784, 30, 10))
    for x in range(10):
        np.random.shuffle(indexes)
        inputs = inputs.isel(cases=indexes)
        labels = labels.isel(cases=indexes)
        net = train_and_save_nn(net, inputs, labels)


# test_2in_1out()
# test_1in_1out()
# test_compare()
test_mnist()
# test_mnist_with_random_images()