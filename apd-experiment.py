import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from littlenet import apd
from littlenet import utility
from littlenet import neural_net as nn

def apply_to_all_layers(net, activations, func, start_layer=1):
    for i in range(start_layer, net.num_layers + 1):
        activation_layer = activations[nn.mkey(i, 'post_activation')]
        yield func(activation_layer)

# gets apd difference for each label to total
def experiment_dapd_by_label(activation_layer, labels):
    dapd = apd.diff_by_label(activation_layer, labels)
    dapd = apd.apd_area(dapd)
    try:
        dapd = dapd.unstack('inputs')
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
    return np.where(costs < threshold)[0]

def prune_dead_neurons(net, activations, labels, threshold=0.01):
    indexes = [i for i in apply_to_all_layers(net, activations,
        lambda x: dead_neuron_indexes(x, labels, threshold=threshold))]
    print('DELETED neurons:', indexes)
    return net.delete_neurons(indexes, activations=activations)

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

def build_10x1_input_apd_net(dapd_by_labels):
    net = nn.NeuralNet((784, 10), func_fill=np.zeros)
    dapd = (dapd_by_labels / 2).stack(inputs=('inputs_y', 'inputs_x')).rename(
        labels='neurons').transpose('inputs', 'neurons').reset_index('inputs', drop=True)
    net.matrices[nn.mkey(0, 'weights')] = dapd
    return net

def build_10x10_input_apd_net(dapd_by_labels):
    net = nn.NeuralNet((784, 10, 10), func_fill=np.zeros)
    dapd = (dapd_by_labels / 2).stack(inputs=('inputs_y', 'inputs_x')).rename(
        labels='neurons').transpose('inputs', 'neurons').reset_index('inputs', drop=True)
    net.matrices[nn.mkey(0, 'weights')] = dapd
    return net

def build_stochastic_culled_net_v2(inputs, labels, sizes=(784, 30, 10),
    num_to_search=100, dapd_by_labels=None, dapd_scale_factor=2):
    
    best_weights = []
    best_biases = []
    control_weights = []
    control_biases = []
    worst_weights = []
    worst_biases = []
    best_dapd, worst_dapd = None, None
    # if set, use dapd to screen weights
    if not dapd_by_labels is None:
        best_dapd = apd.cost(dapd_by_labels, dim='labels').stack(
            inputs=('inputs_y', 'inputs_x')).reset_index('inputs', drop=True)
        # normalize dapd
        best_dapd =  best_dapd / np.amax(best_dapd) * dapd_scale_factor
        worst_dapd = dapd_scale_factor - best_dapd
    for neuron in range(sizes[1]):
        print('Generating neuron', neuron, '...')
        control_net = nn.NeuralNet((sizes[0], num_to_search))
        net = control_net
        # control
        control = np.random.randint(num_to_search)
        control_weights.append(net.matrices[nn.mkey(0, 'weights')].isel(neurons=control))
        control_biases.append(net.matrices[nn.mkey(0, 'biases')].isel(neurons=control))
        # best
        if not dapd_by_labels is None:
            net = copy.deepcopy(control_net)
            net.matrices[nn.mkey(0, 'weights')] = net.matrices[nn.mkey(0, 'weights')] * best_dapd
        activations = net.pass_forward(inputs)[nn.mkey(1, 'post_activation')]
        dapd_per_neuron = apd.cost(apd.apd_area(apd.diff_by_label(
            activations, labels, num_buckets=30)), dim='labels')
        best = np.argmax(dapd_per_neuron)
        best_weights.append(net.matrices[nn.mkey(0, 'weights')].isel(neurons=best))
        best_biases.append(net.matrices[nn.mkey(0, 'biases')].isel(neurons=best))
        # worst
        if not dapd_by_labels is None:
            net = copy.deepcopy(control_net)
            net.matrices[nn.mkey(0, 'weights')] = net.matrices[nn.mkey(0, 'weights')] * worst_dapd
            activations = net.pass_forward(inputs)[nn.mkey(1, 'post_activation')]
            dapd_per_neuron = apd.cost(apd.apd_area(apd.diff_by_label(
                activations, labels, num_buckets=30)), dim='labels')
        worst = np.argmin(dapd_per_neuron)
        worst_weights.append(net.matrices[nn.mkey(0, 'weights')].isel(neurons=worst))
        worst_biases.append(net.matrices[nn.mkey(0, 'biases')].isel(neurons=worst))
    net = nn.NeuralNet(sizes)
    best_net = copy.deepcopy(net)
    best_net.matrices[nn.mkey(0, 'weights')] = xr.concat(best_weights, dim='neurons').transpose('inputs', 'neurons')
    best_net.matrices[nn.mkey(0, 'biases')] = xr.concat(best_biases, dim='neurons')
    control_net = copy.deepcopy(net)
    control_net.matrices[nn.mkey(0, 'weights')] = xr.concat(control_weights, dim='neurons').transpose('inputs', 'neurons')
    control_net.matrices[nn.mkey(0, 'biases')] = xr.concat(control_biases, dim='neurons')
    worst_net = copy.deepcopy(net)
    worst_net.matrices[nn.mkey(0, 'weights')] = xr.concat(worst_weights, dim='neurons').transpose('inputs', 'neurons')
    worst_net.matrices[nn.mkey(0, 'biases')] = xr.concat(worst_biases, dim='neurons')
    return best_net, control_net, worst_net

def test_experiments():
    untrained_net = utility.read_object('./models/neuralnet-0-untrained.pyc')
    trained_net = utility.read_object('./models/neuralnet-trained-without-random.pyc')
    trained_with_random_net = utility.read_object('./models/neuralnet-trained-with-random.pyc')
    trained_with_noise = utility.read_object('./models/neuralnet-trained-with-noise.pyc')
    kernel_net = utility.read_object('./models/neuralnet-trained-from-artificial-kernel.pyc')
    images = utility.read_idx_images('./mnist_data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('./mnist_data/train-labels.idx1-ubyte')
    labels_onehot = utility.make_onehot(labels, np.arange(10))
    #TODO: remove dead neurons, prune correlated neuron pairs
    # utility.plot_layer_weights(untrained_net)
    # utility.plot_layer_weights(trained_net)
    # utility.plot_layer_weights(trained_with_random_net)
    utility.plot_layer_weights(trained_with_noise)
    plt.show()
    test_net(kernel_net, images, labels)
    test_net(kernel_net, utility.rotate_images_90_deg(images), labels)

if __name__ == '__main__':
    images = utility.read_idx_images('./mnist_data/train-images.idx3-ubyte')
    labels = utility.read_idx_labels('./mnist_data/train-labels.idx1-ubyte')
    
    # dapd_by_labels = experiment_dapd_by_label(images, labels)
    # utility.write_object(dapd_by_labels, 'dapd_by_labels_0_noise', directory='./models/dapd')
    # dapd_20_noise = experiment_dapd_by_label(utility.random_noise(images, percent_noise=0.2), labels)
    # utility.write_object(dapd_20_noise, 'dapd_by_labels_20_noise', directory='./models/dapd')
    # dapd_50_noise = experiment_dapd_by_label(utility.random_noise(images, percent_noise=0.5), labels)
    # utility.write_object(dapd_50_noise, 'dapd_by_labels_50_noise', directory='./models/dapd')

    dapd_by_labels = utility.read_object('./models/dapd/dapd_by_labels_0_noise.pyc')
    # dapd_20_noise = utility.read_object('./models/dapd/dapd_by_labels_20_noise.pyc')
    # dapd_50_noise = utility.read_object('./models/dapd/dapd_by_labels_50_noise.pyc')
    # utility.write_object(build_10x1_input_apd_net(dapd_by_labels), 'untrained_10x1_label_dapd', directory='./models/experiment_apd_nets')
    # utility.write_object(build_10x10_input_apd_net(dapd_by_labels), 'untrained_10x10_label_dapd', directory='./models/experiment_apd_nets')
    # utility.write_object(build_10x1_input_apd_net(dapd_50_noise), 'untrained_10x1_label_dapd_50_noise', directory='./models/experiment_apd_nets')
    # utility.write_object(build_10x10_input_apd_net(dapd_50_noise), 'untrained_10x10_label_dapd_50_noise', directory='./models/experiment_apd_nets')
    # utility.write_object(build_30x10_probability_apd_net(dapd_by_labels), 'untrained_10x10_label_dapd', directory='./models/experiment_apd_nets')
    
    experiment_dir = './models/experiment_reg_vs_culled_1/'
    num_trials = 5
    for i in range(1, num_trials):
        if i > 1:
            utility.write_object(nn.NeuralNet((784, 30, 10)), 'untrained_30x10_control_' + str(i), directory=experiment_dir)
            best, control, worst = build_stochastic_culled_net_v2(images, labels, num_to_search=100)
            utility.write_object(best, 'untrained_30x10_rand_culled100_best_' + str(i), directory=experiment_dir)
            utility.write_object(worst, 'untrained_30x10_rand_culled100_worst_' + str(i), directory=experiment_dir)
            best, control, worst = build_stochastic_culled_net_v2(images, labels, num_to_search=10)
            utility.write_object(best, 'untrained_30x10_rand_culled10_best_' + str(i), directory=experiment_dir)
            utility.write_object(worst, 'untrained_30x10_rand_culled10_worst_' + str(i), directory=experiment_dir)
        best, control, worst = build_stochastic_culled_net_v2(images, labels, dapd_by_labels=dapd_by_labels, num_to_search=100)
        utility.write_object(best, 'untrained_30x10_semiculled100_best_' + str(i), directory=experiment_dir)
        utility.write_object(worst, 'untrained_30x10_semiculled100_worst_' + str(i), directory=experiment_dir)
        best, control, worst = build_stochastic_culled_net_v2(images, labels, dapd_by_labels=dapd_by_labels, num_to_search=10)
        utility.write_object(best, 'untrained_30x10_semiculled10_best_' + str(i), directory=experiment_dir)
        utility.write_object(worst, 'untrained_30x10_semiculled10_worst_' + str(i), directory=experiment_dir)

    # test_experiments()
