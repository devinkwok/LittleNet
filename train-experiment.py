import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from littlenet import neural_net as nn
from littlenet import utility
from littlenet import train

def make_random_data(inputs, labels, proportion=0.5):
    random_inputs = utility.random_noise(inputs, percent_noise=1, noise_stdev=0)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    return train.combine_arrays(inputs, random_inputs), \
           train.combine_arrays(labels, train.empty_labels(random_inputs))

def make_shuffled_data(inputs, labels, proportion=0.5):
    shuffled_inputs = utility.shuffle_pixels(inputs)
    shuffled_inputs = shuffled_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    return train.combine_arrays(inputs, shuffled_inputs), \
           train.combine_arrays(labels, train.empty_labels(shuffled_inputs))
    
def train_various(directory, inputs, labels, test_inputs, test_labels,
                  read_pattern='untrained_', write_prefix='trained_', hyperparams={'batch_size': 10, 'training_rate': 3.0}):
    for name, net in utility.read_all_objects(directory=directory, pattern=read_pattern + '*'):
        name = name.split(read_pattern)[-1].split('.pyc')[0] # remove directory and suffix
        print('TRAINING', name)
        name = write_prefix + name
        # regular training
        train.write_nn(*train.train_nn(net, inputs, labels, test_inputs, test_labels, hyperparams=hyperparams),
                    name=name + '_regular', save_dir=directory)
        # inputs have added noise
        noisy_inputs = utility.random_noise(inputs, percent_noise=0.2)
        noisy_net = train.train_nn(net, noisy_inputs, labels, test_inputs, test_labels, hyperparams=hyperparams)
        train.write_nn(*noisy_net, name=name + '_noisy', save_dir=directory)
        # extra 50% of control images that are random noise
        random_inputs, random_labels = make_random_data(inputs, labels)
        random_net = train.train_nn(net, random_inputs, random_labels, test_inputs, test_labels, hyperparams=hyperparams)
        train.write_nn(*random_net, name=name + '_random', save_dir=directory)
        # extra 50% of control images that are shuffled noise
        shuffled_inputs, shuffled_labels = make_shuffled_data(inputs, labels)
        shuffled_net = train.train_nn(net, shuffled_inputs, shuffled_labels, test_inputs, test_labels, hyperparams=hyperparams)
        train.write_nn(*shuffled_net, name=name + '_shuffled', save_dir=directory)
        # combination of noisy inputs, random and shuffled control images
        combo_inputs, combo_labels = make_random_data(inputs, labels)
        combo_inputs, combo_labels = make_shuffled_data(combo_inputs, combo_labels, proportion=0.333333333334)
        combo_net = train.train_nn(net, combo_inputs, combo_labels, test_inputs, test_labels, hyperparams=hyperparams)
        train.write_nn(*combo_net, name=name + '_combo', save_dir=directory)

# trains a randomly initialized network with differing types of noise in data
# then does same thing for a network initialized with kernels
def experiment_randomized_data_random_vs_kernel(inputs, labels, test_inputs, test_labels,
    save_dir='./models/experiment_randomized_data_random_vs_kernel/'):

    utility.write_object(nn.NeuralNet((784, 30, 10)), save_dir + 'untrained-rand.pyc')

    first_layer = utility.tile_kernel(utility.square_kernel(3, 3), stride=(4, 4))
    kernel_net = nn.NeuralNet((784, first_layer.sizes['neurons'], 10))
    kernel_net.matrices[nn.mkey(0, 'weights')] = first_layer.reset_index('inputs', drop=True)
    utility.write_object(kernel_net, save_dir + 'untrained-kernel.pyc')

    random_net = utility.read_object(save_dir + 'untrained-rand.pyc')
    kernel_net = utility.read_object(save_dir + 'untrained-kernel.pyc')
    train_various(save_dir, random_net, inputs, labels, test_inputs, test_labels, write_prefix='reg')
    train_various(save_dir, kernel_net, inputs, labels, test_inputs, test_labels, write_prefix='kernel')

PLOT_COLORS = ['tab:blue', 'tab:cyan', 'tab:gray', 'tab:orange', 'tab:red', 'tab:green',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
PLOT_LINESTYLES = ["-", "--", "-.", ":"]

def plot_loss_arrays(*pattern_groups, directory='./models/experiment', prefix='progress_'):
    arr, net_names = [], []
    for pattern in pattern_groups:
        loss_arrs = utility.read_all_objects(directory=directory, pattern=prefix + pattern)
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
    if not arr:
        return # nothing to plot
    arr = xr.concat(arr, dim='nets').assign_coords(nets=net_names_coords)
    linestyles = cycler('linestyle', PLOT_LINESTYLES) * cycler('lw', [2, 2, 1, 1])
    linestyles = linestyles[:len(pattern_groups) * 2]
    new_prop_cycle = cycler('color', PLOT_COLORS) * linestyles
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=new_prop_cycle)
    plt.rc('legend')
    arr.plot(x='cases', hue='nets')
    plt.show()

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
    num_input_cases = 1000
    num_tests = 100
    inputs = images.isel(cases=slice(num_input_cases))
    labels = labels_onehot.isel(cases=slice(num_input_cases))
    test_inputs = images.isel(cases=slice(num_input_cases, num_input_cases + num_tests))
    test_labels = labels_onehot.isel(cases=slice(num_input_cases, num_input_cases + num_tests))

    test_dir = './models/test/'
    # test_dir = './models/experiment_apd_nets/'
    # experiment_randomized_data_random_vs_kernel(inputs, labels, test_inputs, test_labels)

    # plot_all_nets(test_dir, key='untrained_30x10_control')
    # train_various(test_dir, inputs, labels, test_inputs, test_labels)
    # plot_all_nets(test_dir, key='trained-')

    groups = ['asdfasdf']
    # groups = [
    #     '*semiculled100_best*random',
    #     # '*rand_culled100_best*random',
    #     '*semiculled10_best*random',
    #     # '*rand_culled10_best*random',
    #     '*control*regular',
    #     '*semiculled10_worst*random',
    #     # '*rand_culled10_worst*random',
    #     '*semiculled100_worst*random'
    #     # '*rand_culled100_worst*random',
    #     ]
    plot_loss_arrays(*groups, directory=test_dir)