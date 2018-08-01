import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from neural_net import NeuralNet as Nn, mkey

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

    NUM_BATCHES = 20
    NUM_CASES = 100
    NUM_DIMS = 2
    inputs = xr.DataArray(np.random.rand(NUM_BATCHES, NUM_CASES, NUM_DIMS), dims=('batches', 'cases', 'inputs'))
    labels = xr.DataArray(np.greater(func_probability(inputs), np.random.rand(NUM_BATCHES, NUM_CASES)), dims=('batches', 'cases'))
    labels = labels.expand_dims('labels')

    net = Nn((2,1))
    for new_net, i, l in net.train_yield(inputs, labels, training_rate=5.0):
        plot_tensor_2D(new_net)
        plt.scatter(i.isel(inputs=0), i.isel(inputs=1), c=l.isel(labels=0))
        plt.show()


    
def test_1in_1out():

    def target_function_1D(x):
        return np.abs(np.sin(np.multiply(x, 1)))  # inputs from 0 to 1, outputs from 0 to 1

    NUM_BATCHES = 100
    NUM_CASES = 100
    SIZES = (1, 2, 1)
    net = Nn(SIZES)
    inputs = np.divide(np.arange(NUM_CASES * NUM_BATCHES), NUM_CASES * NUM_BATCHES)
    labels = target_function_1D(inputs)
    indexes = np.arange(NUM_CASES * NUM_BATCHES)
    np.random.shuffle(indexes)
    inputs = xr.DataArray(inputs[indexes].reshape((NUM_BATCHES, NUM_CASES, 1)), dims=('batches', 'cases', 'inputs'))
    labels = xr.DataArray(labels[indexes].reshape((NUM_BATCHES, NUM_CASES, 1)), dims=('batches', 'cases', 'labels'))
    for new_net, i, l in net.train_yield(inputs, labels, training_rate=3.0):
        x = np.linspace(0, 1, num=50)
        xgrid = xr.DataArray(x, dims=('cases'), coords={'cases': x}).expand_dims('inputs')
        print(new_net.matrices)
        net_outputs = new_net.output_only(new_net.pass_forward(xgrid)).squeeze()
        goal_outputs = target_function_1D(xgrid)
        plt.scatter(x, net_outputs)
        plt.scatter(i.squeeze().rename(cases='inputs'), l.squeeze().rename(cases='outputs'))
        goal_outputs.plot()
        # net_outputs.plot()
        plt.show()
        

    NUM_TESTS = 100
    test_inputs = np.divide(np.arange(NUM_TESTS), NUM_TESTS)
    xarray_inputs = xr.DataArray(test_inputs.reshape((NUM_TESTS, 1)), dims=('cases', 'inputs'))
    goal_outputs = target_function_1D(xarray_inputs).squeeze()
    net_outputs = net.pass_forward(xarray_inputs)[mkey(len(SIZES)-1, 'post_activation')].squeeze()
    goal_outputs.assign_coords(cases=test_inputs)
    net_outputs.assign_coords(cases=test_inputs)
    goal_outputs.plot()
    net_outputs.plot()

# test_2in_1out()
test_1in_1out()