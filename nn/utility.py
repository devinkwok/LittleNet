import os
import glob
import struct
import pickle
import copy
import math
import numpy as np
import xarray as xr
import neural_net as nn

def read_all_objects(directory, pattern):
    return [(os.path.split(filename)[1], read_object(filename)) for filename in glob.glob(directory + pattern + '.pyc')]

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        images = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        images = normalize(images)
        images = xr.DataArray(images, dims=('cases', 'inputs_y', 'inputs_x'),
            coords={'cases': np.arange(images.shape[0])})
        return images.stack(inputs=('inputs_y', 'inputs_x'))

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.int8)
        return xr.DataArray(labels, dims=('cases'), coords={'cases': np.arange(labels.shape[0])})

### resize data to fit in range, flatten array
def normalize(np_array):
    return np.multiply(np_array, 1/256.0)

def read_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

def write_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def unstack_copy(images, deep_copy=True):
    if deep_copy:
        return copy.deepcopy(images.unstack('inputs'))
    return images.unstack('inputs')

def shuffle_pixels(images, deep_copy=True):
    image_np_array = unstack_copy(images, deep_copy=deep_copy).transpose(
        'inputs_y', 'inputs_x', 'cases').values
    random_images = np.empty(image_np_array.shape)
    for x in range(image_np_array.shape[0]):
        for y in range(image_np_array.shape[1]):
            column = image_np_array[x, y,:]
            np.random.shuffle(column)
            random_images[x,y,:] = column
    random_images = xr.DataArray(image_np_array,
        dims=('inputs_y', 'inputs_x', 'cases')).transpose(
            'cases', 'inputs_y', 'inputs_x').stack(inputs=('inputs_y', 'inputs_x'))
    return random_images

def random_noise(images, percent_noise=0.5, noise_stdev=0.1, deep_copy=True):
    image_array = unstack_copy(images, deep_copy=deep_copy)
    noise = np.random.rand(*list(image_array.sizes.values()))
    ratio = np.random.randn(*list(image_array.sizes.values()))
    ratio = ratio * noise_stdev + percent_noise
    ratio = np.minimum(np.maximum(ratio, 0), 1)
    image_array = image_array * (1 - ratio) + noise * ratio
    return image_array.stack(inputs=('inputs_y', 'inputs_x'))

def rotate_images_90_deg(images, num_clockwise=1, deep_copy=True):
    rotate_case = num_clockwise % 4
    if rotate_case == 0:
        if deep_copy:
            return copy.deepcopy(images)
        return images
    if rotate_case == 2:
        return flip_images(flip_images(
            images, dim='inputs_x', deep_copy=deep_copy), dim='inputs_y', deep_copy=False)
    #flip axes to rotate along angled plane, then flip image to get rotation
    # flip x for counterclockwise, flip y for clockwise
    if rotate_case == 1:
        return flip_images(flip_images_on_angle(images, deep_copy=deep_copy),
            dim='inputs_x', deep_copy=False)
    # case 3
    return flip_images(flip_images_on_angle(images, deep_copy=deep_copy),
        dim='inputs_y', deep_copy=False)

def flip_images_on_angle(images, switch_axis=None, deep_copy=True):
    image_array = unstack_copy(images, deep_copy=deep_copy)
    if not switch_axis is None:
        # reverse both axes then do angle flip
        image_array = flip_images(flip_images(images, dim='inputs_x',
            deep_copy=deep_copy), dim='inputs_y', deep_copy=False).unstack('inputs')
    image_array = image_array.rename(inputs_x='y_temp', inputs_y='x_temp')
    image_array = image_array.rename(x_temp='inputs_x', y_temp='inputs_y')
    return image_array.stack(inputs=('inputs_y', 'inputs_x'))

def flip_images(images, dim='inputs_y', deep_copy=True):
    image_array = unstack_copy(images, deep_copy=deep_copy)
    image_array = image_array.isel({dim: slice(None, None, -1)})
    return image_array.stack(inputs=('inputs_y', 'inputs_x'))

# shifts images so that corners are in middle, and middle is in corners
def quarter_images(images, deep_copy=True):
    image_array = unstack_copy(images, deep_copy=deep_copy)
    x_size = math.floor(image_array.sizes['inputs_x'] / 2)
    y_size = math.floor(image_array.sizes['inputs_y'] / 2)
    return image_array.roll(inputs_y=y_size, inputs_x=x_size).stack(inputs=('inputs_y', 'inputs_x'))

def square_kernel(kernel_width=4, falloff_width=4):
    width = kernel_width + falloff_width * 2
    kernel = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            diffx = i - width / 2
            diffy = j - width / 2
            distance = math.sqrt(diffx * diffx + diffy * diffy)
            falloff = distance - kernel_width / 2
            kernel[i,j] = distance
            if falloff <= 0:
                kernel[i, j] = 1
            else:
                kernel[i, j] = max(1 - falloff / falloff_width, 0)
    return kernel

def tile_kernel(np_kernel, fill_dims=(28, 28), stride=(4,4)):
    tiles = []
    shape = np_kernel.shape
    kernel = xr.DataArray(np.zeros(fill_dims), dims=('inputs_y', 'inputs_x'))
    kernel[:shape[0],:shape[1]] = np_kernel
    for i in range(0, fill_dims[0] - shape[0]+1, stride[0]):
        for j in range(0, fill_dims[1] - shape[1]+1, stride[1]):
            tiles.append(kernel.roll(inputs_y=i, inputs_x=j))
    return xr.concat(tiles, 'neurons').stack(inputs=('inputs_y', 'inputs_x'))

def plot_layer_weights(neural_net):
    weights = neural_net.matrices[nn.mkey(0, 'weights')]
    weights = xr.DataArray(weights.values.reshape((28, 28, weights.sizes['neurons'])),
        dims=('inputs_y', 'inputs_x', 'neurons'))
    weights.plot(x='inputs_x', y='inputs_y', col='neurons', col_wrap=10)

# similar to np.meshgrid, turns lists into coordinates
def vectorize_params(*params_list):
    dims = len(params_list)
    params = np.array(np.meshgrid(*params_list)).transpose()
    return params.reshape((int(params.size / dims), dims))
