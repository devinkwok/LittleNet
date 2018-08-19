# calculate activation probability distributions
# put into a sql like database
# query APD between 2 sets of data
# example: apd(all, label=0)

import numpy as np
import xarray as xr
import neural_net as nn
import matplotlib.pyplot as plt
import utility

def apd_isel(histograms, indexes):
    return {key: histograms[key].isel(indexes) for key in histograms}

def apd_concat(histograms, hist_array={}):
    for key in histograms:
        if key not in hist_array:
            hist_array[key] = []
        hist_array[key].append(histograms[key])
    return hist_array

def apd_sum(histograms, dims='cases'):
    return {key: histograms[key].sum(dim=dims) for key in histograms}

def apd_diff(histograms1, histograms2, dims='cases'):
    return {key: apd_layer_diff(histograms1[key], histograms2[key]) for key in histograms1}

def apd(neural_net, cases, num_buckets=10):
    activations = neural_net.pass_forward(cases)
    histograms = {}
    for i in range(1, neural_net.num_layers + 1):  # don't start at 0 since this is input layer
        histograms[nn.mkey(i, 'post_activation')] = apd_layer(activations[nn.mkey(i,
            'post_activation')], num_buckets=num_buckets)
    return histograms

def apd_layer_diff(histogram1, histogram2, dims='cases'):
    diff = histogram1.mean(dim=dims) - histogram2.mean(dim=dims)
    return np.absolute(diff).sum(dim='histogram_buckets')

def apd_layer(layer_activations, num_buckets=10):
    scaled = layer_activations * (num_buckets - 1)
    histogram = [np.maximum(1 - np.abs(b - scaled), 0) for b in range(num_buckets)]
    return xr.concat(histogram, dim='histogram_buckets')

def plot_all_apds(apds):
    for key in apds:
        apds[key].mean(dim='cases').plot()
        plt.show()

def apd_layer_compare(inputs, truth_array):
    indexes = np.where(truth_array)[0]
    return apd_diff(apd_layer(inputs), apd_layer(inputs.isel(cases=indexes)))

def apd_compare(total_apd, truth_array):
    indexes = np.where(truth_array)[0]
    return apd_diff(total_apd, apd_isel(total_apd, {'cases': indexes}))

def apd_layer_by_label(apds_layer, labels):
    apds = []
    for i in range(10):
        indexes = np.where(np.equal(labels, i))[0]
        apds.append(apds_layer.isel(cases=indexes))
    return apds

def apd_layer_neuron_pairs(apds):
    coords = []
    grid = []
    cols = []
    num_neurons = apds[0].sizes['inputs']
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            rows = []
            for k in range(10):
                apd = apds[k]
                rows.append(apd_layer_diff(apd.isel(inputs=i),
                    apd.isel(inputs=j)))
            cols.append(xr.concat(rows, dim='labels'))
            coords.append(str(i) + ':' + str(j))
            if (len(cols) > num_neurons):
                grid.append(xr.concat(cols, dim='i'))
                cols = []
    return xr.concat(grid, dim='j')

def test_write_input_apds(net, images, labels):
    inputs_only = net.pass_forward(images)[nn.mkey(0, 'post_activation')]
    inputs_only = xr.DataArray(inputs_only.values.reshape((60000, 28, 28)), dims=('cases', 'inputs_x', 'inputs_y'))
    input_apds = []
    for i in range(10):
        input_apds.append(apd_layer_compare(inputs_only, np.equal(labels, i)))
    utility.write_object(input_apds,
        '/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-1-input-apd_diff-all-to-labels.pyc')

def test_del_neurons(net, activations, labels, score_threshold=1):
    del_neurons = []
    for i in range(net.num_layers - 1):
        apds = apd_layer(activations[nn.mkey(i + 1, 'post_activation')])
        apds_labels = apd_layer_by_label(apds, labels)
        diffs = xr.concat([apd_layer_diff(apds, label) for label in apds_labels], dim='labels')
        score = np.square(diffs).sum(dim='labels')
        del_neurons.append( np.where(score < score_threshold)[0])
    print('DELETED neurons:', del_neurons)
    return net.delete_neurons(activations, del_neurons)

def test_apd():
    untrained_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-0-untrained.pyc')
    trained_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained-without-random.pyc')
    images = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    labels_onehot = nn.make_onehot(labels, np.arange(10))
    NUM_CASES = images.sizes['cases']
    BATCH_SIZE = 10
    NUM_BATCHES = NUM_CASES / BATCH_SIZE
    test_images = images.isel(cases=slice(100))
    test_labels = labels_onehot.isel(cases=slice(100))

    random_images = utility.shuffle_pixels(images).isel(cases=slice(100))
    trained_net.output_only(trained_net.pass_forward(random_images)).plot()
    plt.show()
    
    untrained_outputs = untrained_net.output_only(untrained_net.pass_forward(test_images))
    print('UNTRAINED accuracy:', nn.accuracy_sum(untrained_outputs, test_labels).values,
        'loss:', nn.cost_mean_squared(untrained_outputs, test_labels).values)
    trained_outputs = trained_net.output_only(trained_net.pass_forward(test_images))
    print('TRAINED accuracy:', nn.accuracy_sum(trained_outputs, test_labels).values,
        'loss:', nn.cost_mean_squared(trained_outputs, test_labels).values)

    
    apd_untrained = apd(untrained_net, images)
    apd_trained = apd(trained_net, images)
    apds = {}
    # apds = apd_concat(apd_diff(apd_untrained, apd(untrained_net, random_images)))
    # apds = apd_concat(apd_diff(apd_trained, apd(trained_net, random_images)), hist_array=apds)
    # apds = apd_concat(apd_diff(apd(untrained_net, random_images), apd(trained_net, random_images)), hist_array=apds)
    # apds = apd_concat(apd_diff(apd_untrained, apd_trained), hist_array=apds)
    
    # test_write_input_apds(trained_net, images, labels)

    # input_apds = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-0-input-apd_diff-all-to-labels.pyc')
    # xr.concat(input_apds, 'labels').plot(y='inputs_x', x='inputs_y', col='labels', col_wrap=5)
    # plt.show()

    activations = trained_net.pass_forward(images)
    indexes = np.arange(NUM_CASES)
    np.random.shuffle(indexes)
    binned_images = images.groupby_bins('cases', NUM_BATCHES)
    binned_labels = labels_onehot.groupby_bins('cases', NUM_BATCHES)

    pruned_net = test_del_neurons(trained_net, activations, labels, score_threshold=2)
    pruned_output = pruned_net.output_only(pruned_net.pass_forward(test_images))
    print('PRUNED accuracy:', nn.accuracy_sum(pruned_output, test_labels).values,
        'loss:', nn.cost_mean_squared(pruned_output, test_labels).values)
    count = 0
    for newNet, i, l in pruned_net.train_yield(binned_images, binned_labels):
        if count % 100 == 0:
            test_outputs = newNet.output_only(newNet.pass_forward(test_images))
            print('number:', count * BATCH_SIZE,
                'accuracy:', nn.accuracy_sum(test_outputs, test_labels).values,
                'loss:', nn.cost_mean_squared(test_outputs, test_labels).values)
        if count == 200:
            break
        count += 1
        pruned_net = newNet
    retrained_output = pruned_net.output_only(pruned_net.pass_forward(test_images))
    print('RETRAINED accuracy:', nn.accuracy_sum(retrained_output, test_labels).values,
        'loss:', nn.cost_mean_squared(retrained_output, test_labels).values)

    inputs_only = activations[nn.mkey(0, 'post_activation')]
    correct = nn.accuracy(trained_net.output_only(activations), labels_onehot)
    incorrect = np.logical_not(correct)
    # apd_diff(apd_layer_diff(apd_layer(inputs_only),
    #         apd_layer(inputs_only.isel(cases=indexes)))).plot()
    
    # plot_all_apds(apd_trained)
    # plot_all_apds(apd_untrained)
    
    # apd_by_layer = apd_layer_by_label(apd_trained[nn.mkey(1, 'post_activation')], labels)
    # a = apd_layer_neuron_pairs(apd_by_layer)
    # a.plot(x='i', y='labels', col='j', col_wrap=6)
    # print(a)
    # plt.show()
    
    # for i in range(10):
    #     apds = apd_concat(apd_compare(apd_untrained, np.equal(labels, i)), hist_array=apds)
    for i in range(10):
        apds = apd_concat(apd_compare(apd_trained, np.equal(labels, i)), hist_array=apds)
    for key in apds:
        xr.concat(apds[key], dim='training').plot()
        plt.show()
    

test_apd()