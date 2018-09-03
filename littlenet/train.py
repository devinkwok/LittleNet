"""Functions for preparing training data, and training and evaluating neural nets."""

import itertools
import numpy as np
import xarray as xr
import neural_net as nn
import utility


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
             batch_size=10, training_rate=3.0, sample_rate=100, do_print=True):
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
    num_batches = int(train_inputs.sizes[nn.DIM_CASE] / batch_size)
    inputs = train_inputs.groupby_bins(nn.DIM_CASE, num_batches)
    labels = train_labels.groupby_bins(nn.DIM_CASE, num_batches)
    last = net
    count = 0
    loss_arr = [evaluate_net(
        last, test_inputs, test_labels, num=count * batch_size, do_print=do_print)]
    print_func(end='\n')
    for new_net, inputs, labels in net.train_yield(inputs, labels, training_rate=training_rate):
        last = new_net
        count += 1
        if count * batch_size % sample_rate == 0:
            loss_arr.append(evaluate_net(last, test_inputs, test_labels,
                                         num=int(count * batch_size), do_print=do_print))
            print_func(end='\r')
    print_func('\n ... done')
    return last, loss_arr


def write_nn(net, loss_array, name='net',
             save_dir='../models/experiment/'):
    """Convenience function for writing the trained net and tests from train_nn() to file.

    Arguments:
        net {NeuralNet} -- net to write
        loss_array {list(tuple(int, float, float))} -- test results to write

    Keyword Arguments:
        name {str} -- filename (default: {'net'})
        save_dir {str} -- directory to save to
            (default: {'../models/experiment/'})
    """

    utility.write_object(net, save_dir + 'trained_' + name + '.pyc')
    utility.write_object(loss_array, save_dir + 'progress_' + name + '.pyc')

#TODO: fix
def benchmark(*name_net_input_label_tuples, test_inputs=None, test_labels=None,
              max_batches=5000, num_cases=[50000], rates=[3.0], batch_sizes=[10], sample_rate=100,
              save_dir='/home/devin/d/data/src/abstraction/neural_net_v2/models/experiment/', do_print=True):
    for name, net, net_inputs, net_labels in name_net_input_label_tuples:
        if not save_dir is None:
            utility.write_object(
                net, save_dir + 'net_untrained-' + name + '.pyc')
        for case in num_cases:
            train_inputs, train_labels = tile_shuffled_cases(net_inputs.isel(cases=slice(case)),
                                                             net_labels.isel(cases=slice(case)), tile_size=max_batches * max(batch_sizes))

            for batch, rate in utility.compose_params(batch_sizes, rates):
                id_dict = {'name': name, nn.DIM_CASE: case,
                           'batch_size': batch, 'training_rate': rate}
                name_id = name + '-' + str(int(case)) + 'cases-' + str(int(batch)) + \
                    'batchsize-' + str(rate) + 'rate'
                if do_print:
                    print('TRAINING', name_id)
                num_cases_needed = int(batch * max_batches)
                last, loss_arr = train_nn(net, train_inputs.isel(cases=slice(num_cases_needed)),
                                          train_labels.isel(cases=slice(
                                              num_cases_needed)), test_inputs, test_labels,
                                          batch_size=batch, training_rate=rate, sample_rate=sample_rate, do_print=do_print)
                if not save_dir is None:
                    utility.write_object(
                        loss_arr, save_dir + 'progress-' + name_id + '.pyc')
                    utility.write_object(
                        last, save_dir + 'net_trained-' + name_id + '.pyc')
                yield id_dict, last, loss_arr
