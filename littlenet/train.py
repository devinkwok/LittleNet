"""Functions for preparing training data, and training and evaluating neural nets."""

import itertools
import numpy as np
import xarray as xr
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from littlenet import neural_net as nn
from littlenet import utility


def shuffle_indexes(*xr_arrays, shuffle_dim=nn.DIM_CASE):
    """Reorders items in an array. Does not make a new copy.

    Keyword Arguments:
        shuffle_dim {str} -- dimension along which to reorder (default: {nn.DIM_CASE})

    Returns:
        xarray -- view of array that has been shuffled
    """

    indexes = np.arange(xr_arrays[0].sizes[shuffle_dim])
    np.random.shuffle(indexes)
    return [array.isel({shuffle_dim: indexes}) for array in xr_arrays]


def combine_arrays(*xr_arrays, combine_dim=nn.DIM_CASE):
    """Combines two arrays so that their coordinates are still sequential.
        This allows shuffle_indexes() to access each item with a unique coordinate.
        Example: combining arrays of size 3 and 10, the coordinates of the
        second array start at 3.

    Keyword Arguments:
        combine_dim {str} -- dimension along which to concatenate
        arrays (default: {nn.DIM_CASE})

    Returns:
        xarray -- combined array
    """

    indexes = [array.sizes[combine_dim] for array in xr_arrays]
    indexes = [i for i in itertools.accumulate([0] + indexes)]
    combined = []
    for i, j, array in zip(indexes[:-1], indexes[1:], xr_arrays):
        coords = {combine_dim: np.arange(i, j)}
        combined.append(array.assign_coords(**coords))
    return xr.concat(combined, dim=combine_dim)


def empty_labels(inputs, dim=nn.DIM_CASE, symbols=10):
    """Creates a onehot vector that is entirely zeros.

    Arguments:
        inputs {xarray} -- array of inputs to make labels for,
            used just to reference the dimensions and sizes.

    Keyword Arguments:
        dim {str} -- dimension along which to create labels (default: {nn.DIM_CASE})
        symbols {int} -- number of unique symbols, which is the size
            of the onehot dimension (default: {10})

    Returns:
        xarray -- same output as utility.make_onehot()
    """

    labels = xr.DataArray(np.full((inputs.sizes[dim]), None), dims=(dim))
    return utility.make_onehot(labels, np.arange(symbols))


def tile_shuffled_cases(*xr_arrays, tile_size=0, tile_dim=nn.DIM_CASE):
    """Makes array have tile_size number of items along tile_dim.
        Fills in missing items by repeating the array, shuffling all items
        so that the order of items does not repeat.

    Keyword Arguments:
        tile_size {int} -- target size of array along tile_dim (default: {0})
        tile_dim {str} -- dimension to fill (default: {nn.DIM_CASE})

    Returns:
        xarray -- array of tile_size along tile_dim
    """

    output_arrays = [[] for x in xr_arrays]
    for _ in range(0, tile_size + xr_arrays[0].sizes[tile_dim], xr_arrays[0].sizes[tile_dim]):
        shuffled_arrays = shuffle_indexes(*xr_arrays)
        _ = [x.append(y) for x, y in zip(output_arrays, shuffled_arrays)]
    combined_arrays = [combine_arrays(*x) for x in output_arrays]
    return [x.isel({nn.DIM_CASE: slice(tile_size)}) for x in combined_arrays]


def evaluate_net(net, test_inputs, test_labels, num=0, do_print=True):
    """Tests neural net for accuracy and mean square error.

    Arguments:
        net {NeuralNet} -- net to test
        test_inputs {xarray} -- list of testing inputs of appropriate
            dimensions for net.pass_forward()
        test_labels {xarray} -- list of onehot testing labels, same number as test_inputs

    Keyword Arguments:
        num {int} -- batch or case number for printing and plotting (default: {0})
        do_print {bool} -- prints to console if True, does not print
            endline (default: {True})

    Returns:
        int, float, float -- tuple of num, accuracy between [0., 1.], and mean squared loss
    """

    print_func = lambda *args, **kwargs: None
    if do_print:
        print_func = print
    print_func('    testing', num, '... ', end='')
    test_outputs = net.pass_forward_output_only(test_inputs)
    accuracy = nn.accuracy_sum(
        test_outputs, test_labels).values / test_inputs.sizes[nn.DIM_CASE]
    # multiply by 1 to convert from array to float
    loss = nn.cost_mean_squared(test_outputs, test_labels).values * 1
    print_func('accuracy:', accuracy, 'loss:', loss, end='')
    return num, accuracy, loss


def train_nn(net, train_inputs, train_labels, test_inputs, test_labels,
             sample_rate=100, do_print=True, hyperparams={}):
    """Trains a neural network and tracks the accuracy and loss.

    Arguments:
        net {NeuralNet} -- net to train
        train_inputs {xarray} -- training inputs
        train_labels {xarray} -- training labels
        test_inputs {[type]} -- testing inputs
        test_labels {[type]} -- testing labels

    Keyword Arguments:
        batch_size {int} -- number of inputs per batch to train on (default: {10})
        training_rate {float} -- rate to multiply gradient by in nn.train_yield()
            (default: {3.0})
        sample_rate {int} -- number of inputs to wait in between testing
            the net (default: {100})
        do_print {bool} -- prints tests to console if True (default: {True})

    Returns:
        NeuralNet, list(tuple(int, float, float)) -- trained net and list of
            outputs from evaluate_net()
    """

    print_func = lambda *args, **kwargs: None
    if do_print:
        print_func = print
    new_net = net
    progress = [evaluate_net(
        new_net, test_inputs, test_labels, num=0, do_print=do_print)]
    print_func(end='\n')
    for new_net, case_num, batch_num, _, _ in net.train_yield(train_inputs, train_labels, **hyperparams):
        if case_num % sample_rate == 0:
            progress.append(evaluate_net(new_net, test_inputs, test_labels,
                                         num=int(case_num), do_print=do_print))
            print_func(end='\r')
    print_func('\n ... done')
    return new_net, progress


def write_nn(net, loss_array, name='net',
             save_dir='./models/experiment/'):
    """Convenience function for writing the trained net and tests from train_nn() to file.

    Arguments:
        net {NeuralNet} -- net to write
        loss_array {list(tuple(int, float, float))} -- test results to write

    Keyword Arguments:
        name {str} -- filename (default: {'net'})
        save_dir {str} -- directory to save to
            (default: {'./models/experiment/'})
    """
    utility.write_object(net, 'trained_' + name, directory=save_dir)
    utility.write_object(loss_array, 'progress_' + name, directory=save_dir)


def train_all(name_net_tuples, name_input_label_tuples, hyperparam_dicts,
              test_inputs, test_labels, sample_rate=100, save_dir=None):
    trained_nets = []
    trained_progress = []
    for hyperparams in hyperparam_dicts:
        for data_name, inputs, labels in name_input_label_tuples:
            for net_name, net in name_net_tuples:
                result_name = net_name + 'X' + data_name + '_'
                for key, value in hyperparams.items():
                    result_name = result_name + key + str(value)
                print('TRAINING', result_name)
                net, progress = train_nn(net, inputs, labels, test_inputs, test_labels,
                                         sample_rate=sample_rate, hyperparams=hyperparams)
                if not save_dir is None:
                    write_nn(net, progress, name=result_name, save_dir=save_dir)
                trained_nets.append((result_name, net))
                trained_progress.append((result_name, progress))
    return trained_nets, trained_progress

def plot_progress(*name_progress_tuples_groups):
    plot_array, net_names = [], []
    num_per_group = 0  # temporary hack to make cycler work:
    # won't cycle styles properly if groups have different numbers of lines to plot
    num_groups = len(name_progress_tuples_groups)
    for name_progress_tuples in name_progress_tuples_groups:
        num_per_group = len(name_progress_tuples)
        for name, progress in name_progress_tuples:
            temp = np.array(progress)
            acc = xr.DataArray(temp[:, 1], dims=(nn.DIM_CASE),
                coords={nn.DIM_CASE: temp[:, 0]})
            loss = xr.DataArray(temp[:, 2], dims=(nn.DIM_CASE),
                coords={nn.DIM_CASE: temp[:, 0]})
            plot_array.append(acc)
            plot_array.append(loss)
            net_names.append(name + ' acc')
            net_names.append(name + ' loss')
    net_names_coords = pd.MultiIndex.from_arrays([net_names, np.arange(len(net_names))])
    if not plot_array:
        return # nothing to plot
    arr = xr.concat(plot_array, dim='nets').assign_coords(nets=net_names_coords)
    # temporary color cycling
    plot_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:gray', 'tab:orange', 'tab:purple', 'tab:brown',
                   'tab:pink', 'tab:cyan', 'tab:olive']
    plot_linestyles = cycler('linestyle', ["-", "--", "-.", ":"]) * cycler('lw', [2, 2, 1, 1])
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=cycler('color', plot_colors)[:num_groups] * plot_linestyles[:num_per_group * 2])
    plt.rc('legend')
    arr.plot(x=nn.DIM_CASE, hue='nets')
    plt.show()
