import itertools
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import neural_net as nn
from neural_net import mkey
import utility

def shuffle_indexes(*xr_arrays, shuffle_dim='cases'):
    indexes = np.arange(xr_arrays[0].sizes[shuffle_dim])
    np.random.shuffle(indexes)
    return [array.isel({shuffle_dim:indexes}) for array in xr_arrays]

def combine_arrays(*xr_arrays, combine_dim='cases'):
    indexes = [array.sizes[combine_dim] for array in xr_arrays]
    indexes = [i for i in itertools.accumulate([0] + indexes)]
    combined = []
    for i, j, array in zip(indexes[:-1], indexes[1:], xr_arrays):
        coords = {combine_dim: np.arange(i, j)}
        combined.append(array.assign_coords(**coords))
    return xr.concat(combined, dim=combine_dim)

def empty_labels(inputs, dim='cases', symbols=10):
    labels = xr.DataArray(np.full((inputs.sizes[dim]), None), dims=(dim))
    return nn.make_onehot(labels, np.arange(symbols))

# makes enough shuffled copies of arrays to have tile_size items in tile_dim
def tile_shuffled_cases(*xr_arrays, tile_size=0, tile_dim='cases'):
    output_arrays = [[] for x in xr_arrays]
    for i in range(0, tile_size + xr_arrays[0].sizes[tile_dim],  xr_arrays[0].sizes[tile_dim]):
        shuffled_arrays = shuffle_indexes(*xr_arrays)
        [x.append(y) for x, y in zip(output_arrays, shuffled_arrays)]
    combined_arrays = [combine_arrays(*x) for x in output_arrays]
    return [x.isel(cases=slice(tile_size)) for x in combined_arrays]

def evaluate_net(net, test_inputs, test_labels, num=0, do_print=True):
    if do_print:
        print('    testing', num, '... ', end='')
    test_outputs = net.pass_forward_output_only(test_inputs)
    accuracy = nn.accuracy_sum(test_outputs, test_labels).values / test_inputs.sizes['cases']
    # multiply by 1 to convert from array to float
    loss = nn.cost_mean_squared(test_outputs, test_labels).values * 1
    if do_print:
        print('accuracy:', accuracy, 'loss:', loss, end='')
    return num, accuracy, loss

def train_nn(net, train_inputs, train_labels, test_inputs, test_labels,
    batch_size=10, training_rate=3.0, sample_rate=100, do_print=True):

    # train_inputs.isel(cases=slice(20)).unstack('inputs').plot(x='inputs_x', y='inputs_y', col='cases', col_wrap=5)
    # print(train_labels.isel(cases=slice(20)))
    # plt.show()
    # test_inputs.isel(cases=slice(20)).unstack('inputs').plot(x='inputs_x', y='inputs_y', col='cases', col_wrap=5)
    # print(test_labels.isel(cases=slice(20)))
    # plt.show()
    # print(batch_size, training_rate)

    num_batches = int(train_inputs.sizes['cases'] / batch_size)
    inputs = train_inputs.groupby_bins('cases', num_batches)
    labels = train_labels.groupby_bins('cases', num_batches)
    last = net
    count = 0
    loss_arr = [evaluate_net(last, test_inputs, test_labels, num=count * batch_size, do_print=do_print)]
    if do_print:
        print(end='\n')
    for new_net, inputs, labels in net.train_yield(inputs, labels, training_rate=training_rate):
        last = new_net
        count += 1
        if count * batch_size % sample_rate == 0:
            loss_arr.append(evaluate_net(last, test_inputs, test_labels,
                num=int(count * batch_size), do_print=do_print))
            if do_print:
                print(end='\r')
    if do_print:
        print('\n ... done')
    return last, loss_arr

def write_nn(net, loss_array, name='net',
    save_dir='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/'):

    utility.write_object(net, save_dir + 'trained-' + name + '.pyc')
    utility.write_object(loss_array, save_dir + 'progress-' + name + '.pyc')

def benchmark(*name_net_input_label_tuples, test_inputs=None, test_labels=None,
    max_batches=5000, num_cases=[50000], rates=[3.0], batch_sizes=[10], sample_rate=100,
    save_dir='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/', do_print=True):

    for name, net, net_inputs, net_labels in name_net_input_label_tuples:
        if not save_dir is None:
            utility.write_object(net, save_dir + 'net_untrained-' + name + '.pyc')
        for case in num_cases:
            train_inputs, train_labels = tile_shuffled_cases(net_inputs.isel(cases=slice(case)),
                net_labels.isel(cases=slice(case)), tile_size=max_batches * max(batch_sizes))

            for batch, rate in utility.vectorize_params(batch_sizes, rates):
                id_dict = {'name': name, 'cases': case, 'batch_size': batch, 'training_rate': rate}
                name_id = name + '-' + str(int(case)) + 'cases-' + str(int(batch)) + \
                    'batchsize-' + str(rate) + 'rate'
                if do_print:
                    print('TRAINING', name_id)
                num_cases_needed = int(batch * max_batches)
                last, loss_arr = train_nn(net, train_inputs.isel(cases=slice(num_cases_needed)),
                    train_labels.isel(cases=slice(num_cases_needed)), test_inputs, test_labels,
                    batch_size=batch, training_rate=rate, sample_rate=sample_rate, do_print=do_print)
                if not save_dir is None:
                    utility.write_object(loss_arr, save_dir + 'progress-' + name_id + '.pyc')
                    utility.write_object(last, save_dir + 'net_trained-' + name_id + '.pyc')
                yield id_dict, last, loss_arr

def train_with_shuffled(net, inputs, labels, test_inputs, test_labels, proportion=0.5):
    random_inputs = utility.shuffle_pixels(inputs)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    random_labels = empty_labels(random_inputs)
    inputs = combine_arrays(inputs, random_inputs)
    labels = combine_arrays(labels, random_labels)
    return train_nn(net, *shuffle_indexes(inputs, labels), test_inputs, test_labels)
    
def train_with_random(net, inputs, labels, test_inputs, test_labels, proportion=0.5):
    random_inputs = utility.random_noise(inputs, percent_noise=1, noise_stdev=0)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * proportion)))
    random_labels = empty_labels(random_inputs)
    inputs = combine_arrays(inputs, random_inputs)
    labels = combine_arrays(labels, random_labels)
    return train_nn(net, *shuffle_indexes(inputs, labels), test_inputs, test_labels)

def train_with_noise(net, inputs, labels, test_inputs, test_labels, noise_percent=0.2):
    # noises = [utility.random_noise(inputs.isel(
    #     cases=0), percent_noise=i / 20, noise_stdev=0.1) for i in range(21)]
    # xr.concat(noises, dim='noises').unstack('inputs').plot(
    #     x='inputs_x', y='inputs_y', col='noises', col_wrap=5)
    # plt.show()
    inputs = utility.random_noise(inputs, percent_noise=noise_percent)
    return train_nn(net, inputs, labels, test_inputs, test_labels)

def train_with_shuffled_random_noise(net, inputs, labels, test_inputs, test_labels,
    shuffled_proportion=0.5, random_proportion=0.5, noise_percent=0.2):

    shuffled_inputs = utility.shuffle_pixels(inputs)
    shuffled_inputs = shuffled_inputs.isel(cases=slice(int(inputs.sizes['cases'] * shuffled_proportion)))
    shuffled_labels = empty_labels(shuffled_inputs)
    random_inputs = utility.random_noise(inputs, percent_noise=1, noise_stdev=0)
    random_inputs = random_inputs.isel(cases=slice(int(inputs.sizes['cases'] * random_proportion)))
    random_labels = empty_labels(random_inputs)
    inputs = utility.random_noise(inputs, percent_noise=noise_percent)
    inputs = combine_arrays(inputs, random_inputs, shuffled_inputs)
    labels = combine_arrays(labels, random_labels, shuffled_labels)
    return train_nn(net, *shuffle_indexes(inputs, labels), test_inputs, test_labels)

def build_kernel_net():
    first_layer = utility.tile_kernel(utility.square_kernel(3, 3), stride=(4, 4)).transpose('inputs', 'neurons')
    # first_layer.unstack('inputs').transpose('inputs_y', 'inputs_x', 'neurons').plot(
    #     x='inputs_x', y='inputs_y', col='neurons', col_wrap=10)
    # plt.show()
    net = nn.NeuralNet((784, first_layer.sizes['neurons'], 10))
    net.matrices[nn.mkey(0, 'weights')] = first_layer.reset_index('inputs', drop=True)
    return net