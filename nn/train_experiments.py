import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import neural_net as nn
import utility
from train import write_nn, train_nn, train_with_noise, train_with_random, \
    train_with_shuffled, train_with_shuffled_random_noise, build_kernel_net

def plot_loss_arrays(*filename_filters, directory='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/'):
    arr_groups = []
    print(filename_filters)
    for filename in filename_filters:
        arr, net_names = [], []
        loss_arrs = utility.read_all_objects(directory, filename)
        for name, array in loss_arrs:
            temp = np.array(array)
            # acc_loss_coords = pd.MultiIndex.from_arrays([['accuracy', 'loss'], [0, 1]], names=('type', 'id'))
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
        arr_groups.append(arr)
    net_group_coords = pd.MultiIndex.from_arrays([filename_filters, np.arange(len(filename_filters))])
    arr_groups = xr.concat(arr_groups, dim='net_groups').assign_coords(net_groups=net_group_coords)
    new_prop_cycle = cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']) * cycler('linestyle', ['-', '--'])
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=new_prop_cycle)
    arr_groups.plot(x='cases', hue='nets', col='net_groups', col_wrap=1)
    plt.show()

# trains a randomly initialized network with differing types of noise in data
# then does same thing for a network initialized with kernels
def experiment_randomized_data_random_vs_kernel():
    images = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels_onehot = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    labels_onehot = nn.make_onehot(labels_onehot, np.arange(10))
    save_dir = '/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/'

    utility.write_object(nn.NeuralNet((784, 30, 10)), save_dir + 'untrained-random.pyc')
    utility.write_object(build_kernel_net(), save_dir + 'untrained-kernel.pyc')

    random_net = utility.read_object(save_dir + 'untrained-random.pyc')
    kernel_net = utility.read_object(save_dir + 'untrained-kernel.pyc')

    num_input_cases = 50000
    num_tests = 1000
    inputs = images.isel(cases=slice(num_input_cases))
    labels = labels_onehot.isel(cases=slice(num_input_cases))
    test_inputs = images.isel(cases=slice(num_input_cases, num_input_cases + num_tests))
    test_labels = labels_onehot.isel(cases=slice(num_input_cases, num_input_cases + num_tests))

    write_nn(*train_nn(
        random_net, inputs, labels, test_inputs, test_labels), name='rand_regular')
    write_nn(*train_with_shuffled(
        random_net, inputs, labels, test_inputs, test_labels), name='rand_shuffled')
    write_nn(*train_with_random(
        random_net, inputs, labels, test_inputs, test_labels), name='rand_random')
    write_nn(*train_with_noise(
        random_net, inputs, labels, test_inputs, test_labels), name='rand_noise')
    write_nn(*train_with_shuffled_random_noise(
        random_net, inputs, labels, test_inputs, test_labels), name='rand_shuffled_random_noise_combo')
    
    write_nn(*train_nn(
        kernel_net, inputs, labels, test_inputs, test_labels), name='kernel_regular')
    write_nn(*train_with_shuffled(
        kernel_net, inputs, labels, test_inputs, test_labels), name='kernel_shuffled')
    write_nn(*train_with_random(
        kernel_net, inputs, labels, test_inputs, test_labels), name='kernel_random')
    write_nn(*train_with_noise(
        kernel_net, inputs, labels, test_inputs, test_labels), name='kernel_noise')
    write_nn(*train_with_shuffled_random_noise(
        kernel_net, inputs, labels, test_inputs, test_labels), name='kernel_shuffled_random_noise_combo')

    # for a, b, c in benchmark(('regular', random_net, inputs, labels),
    #     test_inputs=test_inputs, test_labels=test_labels,
    #     max_batches=5000, num_cases=[50000, 35000, 20000], rates=[3, 5, 1], batch_sizes=[10, 1, 25]):

    #     pass

if __name__ == '__main__':
    # experiment_randomized_data_random_vs_kernel()
    groups = ['-rand_*', '-kernel_*']
    groups = ['*regular', '*noise', '*random', '*shuffled', '*combo']
    plot_loss_arrays(*groups,
        directory='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment_randomized_data_random_vs_kernel/progress')