import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import apd
import utility
import neural_net as nn

def apply_to_all_layers(net, activations, func, start_layer=1):
    for i in range(start_layer, net.num_layers + 1):
        activation_layer = activations[nn.mkey(i, 'post_activation')]
        yield func(activation_layer)

# gets apd difference for each label to total
def experiment_dapd_by_label(activation_layer, labels):
    dapd = apd.diff_by_label(activation_layer, labels)
    dapd = apd.apd_area(dapd)
    try:
        dapd.unstack('inputs')
        dapd.plot(x='inputs_x', y='inputs_y', col='labels', col_wrap=5)
    except:
        dapd.plot(x='labels', y='inputs')
    plt.show()
    return dapd

# checks how correlated pairs of neurons are
def experiment_dapd_by_neuron_pairs(activation_layer, labels):
    pairs_by_layers = apd.apd_area(apd.diff_by_pairs(activation_layer, labels))
    num_items = activation_layer.sizes['inputs']
    num_pairs = pairs_by_layers.sizes['pairs']
    pairs_cost = apd.cost(pairs_by_layers, dim='labels').assign_coords(pairs=np.arange(num_pairs))
    indexes = pairs_cost.argsort()
    print(pairs_cost.isel(pairs=indexes).isel(pairs=slice(100)))
    #TODO: find an appropriate threshold for pruning pairs, then delete
    pairs_by_layers.plot(x='pairs', y='labels')
    plt.show()

def dead_neuron_indexes(activation_layer, labels, threshold=0.01):
    costs = apd.cost(apd.apd_area(apd.diff_by_label(activation_layer, labels)), 'labels')
    print(costs)
    return np.where(costs < threshold)[0]

def prune_dead_neurons(net, activations, labels, threshold=0.01):
    indexes = [i for i in apply_to_all_layers(net, activations,
        lambda x: dead_neuron_indexes(x, labels, threshold=threshold))]
    print('DELETED neurons:', indexes)
    return net.delete_neurons(activations, indexes)

def test_net(net, images, labels):
    activations = net.pass_forward(images)
    [x for x in apply_to_all_layers(net, activations,
        lambda x: experiment_dapd_by_label(x, labels))]
    # pruned_net = prune_dead_neurons(net, activations, labels)
    # pruned_activations = pruned_net.pass_forward(images)
    # [x for x in apply_to_all_layers(pruned_net, pruned_activations,
    #     lambda x: experiment_dapd_by_label(x, labels))]
    # [x for x in apply_to_all_layers(pruned_net, pruned_activations,
    #     lambda x:experiment_dapd_by_neuron_pairs(x, labels))]

def inputs_activations(inputs, labels):
    dapd = experiment_dapd_by_label(inputs, labels)

def test_experiments():
    untrained_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-0-untrained.pyc')
    trained_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained-without-random.pyc')
    trained_with_random_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained-with-random.pyc')
    trained_with_noise = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained-with-noise.pyc')
    kernel_net = utility.read_object('/home/devin/d/data/src/abstraction/neural_net_v2/models/neuralnet-trained-from-artificial-kernel.pyc')
    images = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    labels_onehot = nn.make_onehot(labels, np.arange(10))
    #TODO: remove dead neurons, prune correlated neuron pairs
    # utility.plot_layer_weights(untrained_net)
    # utility.plot_layer_weights(trained_net)
    # utility.plot_layer_weights(trained_with_random_net)
    utility.plot_layer_weights(trained_with_noise)
    plt.show()
    test_net(kernel_net, images, labels)
    test_net(kernel_net, utility.rotate_images_90_deg(images), labels)

if __name__ == '__main__':
    images = utility.read_idx_images('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('/home/devin/d/data/src/abstraction/mnist-toy-net/data/train-labels.idx1-ubyte')
    # inputs_activations(images, labels)
    test_experiments()
