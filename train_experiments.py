import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from littlenet import neural_net as nn
from littlenet import neural_net
from littlenet import utility
from littlenet import train

PLOT_COLORS = ['tab:blue', 'tab:cyan', 'tab:gray', 'tab:orange', 'tab:red', 'tab:green',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
PLOT_LINESTYLES = ["-", "--", "-.", ":"]

def train_with_shuffled(net, inputs, labels, test_inputs, test_labels, proportion=0.5):
    random_inputs = utility.shuffle_pixels(inputs)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    random_labels = train.empty_labels(random_inputs)
    inputs = train.combine_arrays(inputs, random_inputs)
    labels = train.combine_arrays(labels, random_labels)
    return train.train_nn(net, *train.shuffle_indexes(inputs, labels), test_inputs, test_labels)
    
def train_with_random(net, inputs, labels, test_inputs, test_labels, proportion=0.5):
    random_inputs = utility.random_noise(inputs, percent_noise=1, noise_stdev=0)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    random_labels = train.empty_labels(random_inputs)
    inputs = train.combine_arrays(inputs, random_inputs)
    labels = train.combine_arrays(labels, random_labels)
    return train.train_nn(net, *train.shuffle_indexes(inputs, labels), test_inputs, test_labels)

def train_with_noise(net, inputs, labels, test_inputs, test_labels, noise_percent=0.2):
    # noises = [utility.random_noise(inputs.isel(
    #     cases=0), percent_noise=i / 20, noise_stdev=0.1) for i in range(21)]
    # xr.concat(noises, dim='noises').unstack('inputs').plot(
    #     x='inputs_x', y='inputs_y', col='noises', col_wrap=5)
    # plt.show()
    inputs = utility.random_noise(inputs, percent_noise=noise_percent)
    return train.train_nn(net, inputs, labels, test_inputs, test_labels)

def train_with_shuffled_random_noise(net, inputs, labels, test_inputs, test_labels,
    shuffled_proportion=0.5, random_proportion=0.5, noise_percent=0.2):

    shuffled_inputs = utility.shuffle_pixels(inputs)
    shuffled_inputs = shuffled_inputs.isel(cases=slice(int(inputs.sizes['cases'] * shuffled_proportion)))
    shuffled_labels = train.empty_labels(shuffled_inputs)
    random_inputs = utility.random_noise(inputs, percent_noise=1, noise_stdev=0)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * random_proportion)))
    random_labels = train.empty_labels(random_inputs)
    inputs = utility.random_noise(inputs, percent_noise=noise_percent)
    inputs = train.combine_arrays(inputs, random_inputs, shuffled_inputs)
    labels = train.combine_arrays(labels, random_labels, shuffled_labels)
    return train.train_nn(net, *train.shuffle_indexes(inputs, labels), test_inputs, test_labels)

def build_kernel_net():
    first_layer = utility.tile_kernel(utility.square_kernel(3, 3), stride=(4, 4)).transpose('inputs', 'neurons')
    # first_layer.unstack('inputs').transpose('inputs_y', 'inputs_x', 'neurons').plot(
    #     x='inputs_x', y='inputs_y', col='neurons', col_wrap=10)
    # plt.show()
    net = nn.NeuralNet((784, first_layer.sizes['neurons'], 10))
    net.matrices[nn.mkey(0, 'weights')] = first_layer.reset_index('inputs', drop=True)
    return net

def plot_loss_arrays(*filename_filters, directory='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/', prefix='progress-'):
    arr, net_names = [], []
    for filename in filename_filters:
        loss_arrs = utility.read_all_objects(directory=directory, pattern=prefix + filename)
        for name, array in loss_arrs:
            temp = np.array(array)
            acc = xr.DataArray(temp[:, 1], dims=('cases'),
                coords={'cases': temp[:, 0]})
            loss = xr.DataArray(temp[:, 2], dims=('cases'),
                coords={'cases': temp[:, 0]})
            arr.append(acc)
            arr.append(loss)
            net_names.append(name + ' acc')
            net_names.append(name + ' loss')
    net_names_coords = pd.MultiIndex.from_arrays([net_names, np.arange(len(net_names))])
    arr = xr.concat(arr, dim='nets').assign_coords(nets=net_names_coords)
    linestyles = cycler('linestyle', PLOT_LINESTYLES) * cycler('lw', [2, 2, 1, 1])
    linestyles = linestyles[:len(filename_filters) * 2]
    new_prop_cycle = cycler('color', PLOT_COLORS) * linestyles
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=new_prop_cycle)
    plt.rc('legend')
    arr.plot(x='cases', hue='nets')
    plt.show()

def train_on_various_data(net, inputs, labels, test_inputs, test_labels, name='net', save_dir=''):
    train.write_nn(*train.train_nn(
        net, inputs, labels, test_inputs, test_labels), name=name + '_regular', save_dir=save_dir)
    train.write_nn(*train_with_shuffled(
        net, inputs, labels, test_inputs, test_labels), name=name + '_shuffled', save_dir=save_dir)
    train.write_nn(*train_with_random(
        net, inputs, labels, test_inputs, test_labels), name=name + '_random', save_dir=save_dir)
    train.write_nn(*train_with_noise(
        net, inputs, labels, test_inputs, test_labels), name=name + '_noise', save_dir=save_dir)
    train.write_nn(*train_with_shuffled_random_noise(
        net, inputs, labels, test_inputs, test_labels), name=name + '_shuffled_random_noise_combo', save_dir=save_dir)

# trains a randomly initialized network with differing types of noise in data
# then does same thing for a network initialized with kernels
def experiment_randomized_data_random_vs_kernel(inputs, labels, test_inputs, test_labels,
    save_dir='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment_randomized_data_random_vs_kernel/'):

    utility.write_object(nn.NeuralNet((784, 30, 10)), save_dir + 'untrained-rand.pyc')
    utility.write_object(build_kernel_net(), save_dir + 'untrained-kernel.pyc')

    random_net = utility.read_object(save_dir + 'untrained-rand.pyc')
    kernel_net = utility.read_object(save_dir + 'untrained-kernel.pyc')
    train_on_various_data(random_net, inputs, labels, test_inputs, test_labels,
        name='rand', save_dir=save_dir)
    train_on_various_data(kernel_net, inputs, labels, test_inputs, test_labels,
        name='kernel', save_dir=save_dir)

def experiment_benchmark_params():
    # for a, b, c in benchmark(('regular', random_net, inputs, labels),
    #     test_inputs=test_inputs, test_labels=test_labels,
    #     max_batches=5000, num_cases=[50000, 35000, 20000], rates=[3, 5, 1], batch_sizes=[10, 1, 25]):

    #     pass
    pass

def experiment_all_nets(directory, key='untrained_'):
    for name, net in utility.read_all_objects(directory=directory, pattern=key + '*'):
        name = name.split(key)[-1].split('.pyc')[0]
        print('TRAINING', name)
        train_on_various_data(net, inputs, labels, test_inputs, test_labels,
            name=name, save_dir=directory)

def plot_all_nets(directory, key='untrained_'):
    for name, net in utility.read_all_objects(directory=directory, pattern=key + '*'):
        name = name.split(key)[-1].split('.pyc')[0]
        print(name)
        utility.plot_layer_weights(net, layer=0, shape=(28, 28))
    plt.show()

if __name__ == '__main__':
    images = utility.read_idx_images('./mnist_data/train-images.idx3-ubyte')
    labels_onehot = utility.read_idx_labels('./mnist_data/train-labels.idx1-ubyte')
    labels_onehot = utility.make_onehot(labels_onehot, np.arange(10))
    num_input_cases = 50000
    num_tests = 1000
    inputs = images.isel(cases=slice(num_input_cases))
    labels = labels_onehot.isel(cases=slice(num_input_cases))
    test_inputs = images.isel(cases=slice(num_input_cases, num_input_cases + num_tests))
    test_labels = labels_onehot.isel(cases=slice(num_input_cases, num_input_cases + num_tests))

    test_dir = './models/experiment_reg_vs_culled_1/'
    # test_dir = '/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment_apd_nets/'
    # experiment_randomized_data_random_vs_kernel(inputs, labels, test_inputs, test_labels)

    plot_all_nets(test_dir, key='untrained_30x10_control')
    experiment_all_nets(test_dir)
    # plot_all_nets(test_dir, key='trained-')

    groups = [
        '*semiculled100_best*random',
        # '*rand_culled100_best*random',
        '*semiculled10_best*random',
        # '*rand_culled10_best*random',
        '*control*regular',
        '*semiculled10_worst*random',
        # '*rand_culled10_worst*random',
        '*semiculled100_worst*random'
        # '*rand_culled100_worst*random',
        ]
    plot_loss_arrays(*groups, directory=test_dir)