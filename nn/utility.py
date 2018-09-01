"""Helper functions for file I/O and manipulating MNIST dataset"""
import os
import glob
import struct
import pickle
import copy
import math
import numpy as np
import xarray as xr
import neural_net as nn


def normalize_images(np_array):
    """Resizes MNIST image inputs to fit in interval 0.0 to 1.0.

    Arguments:
        np_array {np_array} -- input array

    Returns:
        np_array -- array divided by 255.0
    """

    return np.multiply(np_array, 1/255.0)


def read_idx_images(filename):
    """Imports MNIST images, adapted from
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40.

    Arguments:
        filename {str} -- path to file

    Returns:
        xarray -- images across dims=DIM_CASE, dim=DIM_IN containing
            DIM_IN_X and DIM_IN_Y for image x and y axes
    """

    with open(filename, 'rb') as file:
        _, _, dims = struct.unpack('>HBB', file.read(4))
        shape = tuple(struct.unpack('>I', file.read(4))
                      [0] for d in range(dims))
        images = np.fromstring(file.read(), dtype=np.uint8).reshape(shape)
        images = normalize_images(images)
        images = xr.DataArray(images, dims=(nn.DIM_CASE, nn.DIM_IN_Y, nn.DIM_IN_X),
                              coords={nn.DIM_CASE: np.arange(images.shape[0])})
        return images.stack(inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))


def read_idx_labels(filename):
    """Imports MNIST labels, adapted from
    https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/utils.py.

    Arguments:
        filename {str} -- path to file

    Returns:
        xarray -- labels as int (1 to 10) across dims=DIM_CASE
    """

    with open(filename, 'rb') as file:
        _, _ = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)
        return xr.DataArray(labels, dims=(nn.DIM_CASE),
                            coords={nn.DIM_CASE: np.arange(labels.shape[0])})


def make_onehot(label_xarray, symbols_list):
    """Makes onehot vector: a vector with a single 1 per row, one column per
        unique symbol. Example: symbols_list [0, 2] will produce:
            0 -> [1, 0]
            1 -> [0, 0]
            2 -> [0, 1]
            3 -> [0, 0]

    Arguments:
        label_xarray {xarray} -- array of labels
        symbols_list {list(int)} -- symbols to match in label_xarray

    Returns:
        xarray -- onehot vector of labels encoded along dim=DIM_LABEL
    """

    onehot_array = []
    for symbol in symbols_list:
        onehot_array.append(np.equal(label_xarray, symbol))
    labels = xr.concat(onehot_array, dim=nn.DIM_LABEL).astype(float)
    return labels.transpose(*label_xarray.dims, nn.DIM_LABEL)


def read_object(filename):
    """Convenience function for retrieving object from file using pickle.

    Arguments:
        filename {str} -- path to file

    Returns:
        object -- object in file (uses pickle)
    """

    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        return obj


def read_all_objects(directory, pattern, suffix='.pyc'):
    """Reads multiple files from a directory that match a pattern using read_object().

    Arguments:
        directory {str} -- directory to search
        pattern {str} -- filename pattern, uses glob.glob()

    Keyword Arguments:
        suffix {str} -- filename suffix (default: {'.pyc'})

    Returns:
        list(object) -- list of objects from each file
    """

    objs = []
    for filename in glob.glob(directory + pattern + suffix):
        objs.append((os.path.split(filename)[1], read_object(filename)))
    return objs


def write_object(obj, filename):
    """Convenience function for writing object to file using pickle.

    Arguments:
        obj {object} -- object to write
        filename {str} -- path to file
    """

    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def unstack_copy(images, deep_copy=True):
    """Helper function for image manipulation: unstacks MNIST input
        from 1-dimensional to 2-dimensional form (DIM_IN into DIM_IN_Y, DIM_IN_X)
        optionally makes a copy of the images.

    Arguments:
        images {xarray} -- input images with dim DIM_IN

    Keyword Arguments:
        deep_copy {bool} -- returns a new copy if true (default: {True})

    Returns:
        xarray -- unstacked images
    """

    if deep_copy:
        return copy.deepcopy(images.unstack(nn.DIM_IN))
    return images.unstack(nn.DIM_IN)


def shuffle_pixels(images, deep_copy=True):
    """Function for making noise from images: takes a list of images,
        and shuffles the pixels between the images, preserving their
        original x, y coordinates (like stacking images vertically,
        then shuffling pixels vertically between images).

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels

    Keyword Arguments:
        deep_copy {bool} -- returns a new copy if true (default: {True})

    Returns:
        xarray -- images across DIM_CASE, with pixels shuffled
    """

    image_np_array = unstack_copy(images, deep_copy=deep_copy).transpose(
        nn.DIM_IN_Y, nn.DIM_IN_X, nn.DIM_CASE).values
    random_images = np.empty(image_np_array.shape)
    for coord_x in range(image_np_array.shape[0]):
        for coord_y in range(image_np_array.shape[1]):
            column = image_np_array[coord_x, coord_y, :]
            np.random.shuffle(column)
            random_images[coord_x, coord_y, :] = column
    random_images = xr.DataArray(image_np_array,
                                 dims=(nn.DIM_IN_Y, nn.DIM_IN_X, nn.DIM_CASE)).transpose(
                                     nn.DIM_CASE, nn.DIM_IN_Y, nn.DIM_IN_X).stack(
                                         inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))
    return random_images


def random_noise(images, percent_noise=0.5, noise_stdev=0.1, deep_copy=True):
    """Function for adding random noise to images, where the noise is evenly
        distributed across the interval [0, 1]. Noise is mixed with the image
        via a normally distributed ratio centered around percent_noise and with
        width of noise_stdev.

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels

    Keyword Arguments:
        percent_noise {float} -- Proportion of noise to image (default: {0.5})
        noise_stdev {float} -- width of variation in percent_noise (default: {0.1})
        deep_copy {bool} -- returns a new copy if true (default: {True})

    Returns:
        xarray -- images across DIM_CASE, with added noise
    """

    image_array = unstack_copy(images, deep_copy=deep_copy)
    noise = np.random.rand(*list(image_array.sizes.values()))
    ratio = np.random.randn(*list(image_array.sizes.values()))
    ratio = ratio * noise_stdev + percent_noise
    ratio = np.minimum(np.maximum(ratio, 0), 1)
    image_array = image_array * (1 - ratio) + noise * ratio
    return image_array.stack(inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))


def rotate_images_90_deg(images, num_clockwise=1, deep_copy=True):
    """Rotates images in 90 degree increments.

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels

    Keyword Arguments:
        num_clockwise {int} -- number of clockwise turns, can be negative (default: {1})
        deep_copy {bool} -- [description] (default: {True})

    Returns:
        xarray -- images across DIM_CASE, rotated
    """

    rotate_case = num_clockwise % 4
    if rotate_case == 0:
        if deep_copy:
            return copy.deepcopy(images)
        return images
    if rotate_case == 2:
        return flip_images(flip_images(
            images, dim=nn.DIM_IN_X, deep_copy=deep_copy), dim=nn.DIM_IN_Y, deep_copy=False)
    # flip axes to rotate along angled plane, then flip image to get rotation
    # flip x for counterclockwise, flip y for clockwise
    if rotate_case == 1:
        return flip_images(flip_images_on_angle(images, deep_copy=deep_copy),
                           dim=nn.DIM_IN_X, deep_copy=False)
    # case 3
    return flip_images(flip_images_on_angle(images, deep_copy=deep_copy),
                       dim=nn.DIM_IN_Y, deep_copy=False)


def flip_images_on_angle(images, topright_to_bottomleft=False, deep_copy=True):
    """Flips images along a diagonal plane at a 45 degree angle along image.
        By default the diagonal plane runs top left to bottom right.

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels

    Keyword Arguments:
        topright_to_bottomleft {bool} -- if true, flips along other diagonal plane
            (default: {False})
        deep_copy {bool} -- [description] (default: {True})

    Returns:
        xarray -- images across DIM_CASE, diagonally flipped
    """

    image_array = unstack_copy(images, deep_copy=deep_copy)
    if topright_to_bottomleft:
        # reverse both axes then do angle flip
        image_array = flip_images(flip_images(images, dim=nn.DIM_IN_X,
                                              deep_copy=deep_copy),
                                  dim=nn.DIM_IN_Y, deep_copy=False).unstack(nn.DIM_IN)
    image_array = image_array.rename(inputs_x='y_temp', inputs_y='x_temp')
    image_array = image_array.rename(x_temp=nn.DIM_IN_X, y_temp=nn.DIM_IN_Y)
    return image_array.stack(inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))


def flip_images(images, dim=nn.DIM_IN_Y, deep_copy=True):
    """Flips images along given dimension.

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels

    Keyword Arguments:
        dim {[type]} -- dimension to flip along, vertical by default (default: {nn.DIM_IN_Y})
        deep_copy {bool} -- [description] (default: {True})

    Returns:
        xarray -- images across DIM_CASE, flipped
    """

    image_array = unstack_copy(images, deep_copy=deep_copy)
    image_array = image_array.isel({dim: slice(None, None, -1)})
    return image_array.stack(inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))


def quarter_images(images, deep_copy=True):
    """Shifts image so that corners are in the center, centers are in corners,
        equivalent to cutting image into 4 quarters, then rearranging so
        that top/bottom and left/right switch places. Used to create
        inputs where the varying pixels are in the corners instead of the
        centers.

    Arguments:
        images {xarray} -- xarray of images aross DIM_CASE, each image
            has DIM_IN_Y and DIM_IN_X pixels
        deep_copy {bool} -- [description] (default: {True})

    Returns:
        xarray -- images across DIM_CASE, quartered and rearranged
    """

    image_array = unstack_copy(images, deep_copy=deep_copy)
    x_size = math.floor(image_array.sizes[nn.DIM_IN_X] / 2)
    y_size = math.floor(image_array.sizes[nn.DIM_IN_Y] / 2)
    image_array = image_array.roll(**{nn.DIM_IN_Y: y_size, nn.DIM_IN_X: x_size})
    return image_array.stack(**{nn.DIM_IN: (nn.DIM_IN_Y, nn.DIM_IN_X)})


def square_kernel(kernel_width=4, falloff_width=4):
    """Creates a square kernel function, like an image of a dot with
    blurred edges. Center of kernel is value 1.0, edges are value 0.

    Keyword Arguments:
        kernel_width {int} -- width of center (default: {4})
        falloff_width {int} -- width of border around center with values
            tapering from 1 to 0 (default: {4})

    Returns:
        np_array -- 2-d array containing kernel
    """

    width = kernel_width + falloff_width * 2
    kernel = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            diffx = i - width / 2
            diffy = j - width / 2
            distance = math.sqrt(diffx * diffx + diffy * diffy)
            falloff = distance - kernel_width / 2
            kernel[i, j] = distance
            if falloff <= 0:
                kernel[i, j] = 1
            else:
                kernel[i, j] = max(1 - falloff / falloff_width, 0)
    return kernel


def tile_kernel(np_kernel, fill_dims=(28, 28), stride=(4, 4), tile_dim=nn.DIM_OUT):
    """Creates a set of filters where a 2-d kernel is placed at regular intervals
        across a larger 2-d space. Default values in the filters are 0.

    Arguments:
        np_kernel {np_array} -- array of kernel values

    Keyword Arguments:
        fill_dims {tuple} -- tuple of int sizes of 2-d filters (default: {(28, 28)})
        stride {tuple} -- tuple of int distances by which kernel is
            moved for each filter (default: {(4, 4)})
        tile_dim {str} -- dimension along which to stack filters (default: nn.DIM_OUT)

    Returns:
        xarray -- xarray of filters
    """

    tiles = []
    shape = np_kernel.shape
    kernel = xr.DataArray(np.zeros(fill_dims), dims=(nn.DIM_IN_Y, nn.DIM_IN_X))
    kernel[:shape[0], :shape[1]] = np_kernel
    for i in range(0, fill_dims[0] - shape[0]+1, stride[0]):
        for j in range(0, fill_dims[1] - shape[1] + 1, stride[1]):
            tiles.append(kernel.roll(**{nn.DIM_IN_Y: i, nn.DIM_IN_X: j}))
    return xr.concat(tiles, tile_dim).stack(inputs=(nn.DIM_IN_Y, nn.DIM_IN_X))


def plot_layer_weights(neural_net, layer=0, shape=None):
    """Plots neural net weights for a single layer. If a shape is given, each
        neuron is plotted separately as a 2-d color plot. Otherwise, neurons are
        plotted together as a single 2-d color plot. Must call plt.show() to
        show plots after.

    Arguments:
        neural_net {NeuralNet} -- Neural net to plot

    Keyword Arguments:
        layer {int} -- layer number (default: {0})
        shape {tuple} -- tuple of ints to reshape weights into (default: {None})
    """

    weights = neural_net.matrices[nn.mkey(layer, nn.KEY_WEIGHT)]
    if not shape is None:
        weights = weights.transpose(nn.DIM_IN, nn.DIM_OUT)
        weights = xr.DataArray(weights.values.reshape((*shape, weights.sizes[nn.DIM_OUT])),
                               dims=(nn.DIM_IN_Y, nn.DIM_IN_X, nn.DIM_OUT))
        weights.plot(x=nn.DIM_IN_X, y=nn.DIM_IN_Y, col=nn.DIM_OUT, col_wrap=10)
    else:
        weights.plot()


def compose_params(*params_list):
    """Composes multiple lists into a single list which has every
        permutation of values from each list, same as np.meshgrid().
        Example: if params_list = [1, 2], [a, b], [8.8, 9.9, 1.1]
        output is [1, a, 8.8], [2, a, 8.8], [1, b, 8.8], [2, b, 8.8],
        [1, a, 9.9] ... [2, b, 1.1]

    Returns:
        np_array -- 2-d list of composed parameters
    """

    dims = len(params_list)
    params = np.array(np.meshgrid(*params_list)).transpose()
    return params.reshape((int(params.size / dims), dims))
