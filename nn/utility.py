import struct
import pickle
import numpy as np
import neural_net as nn
import xarray as xr
import copy

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        images = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        images = normalize(images)
        images = xr.DataArray(images, dims=('cases', 'inputs_x', 'inputs_y'),
            coords={'cases': np.arange(images.shape[0])})
        return images.stack(inputs=('inputs_x', 'inputs_y'))

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

def shuffle_pixels(images):
    image_np_array = copy.deepcopy(images.unstack('inputs').transpose('inputs_x', 'inputs_y', 'cases').values)
    random_images = np.empty(image_np_array.shape)
    for x in range(image_np_array.shape[0]):
        for y in range(image_np_array.shape[1]):
            column = image_np_array[x, y,:]
            np.random.shuffle(column)
            random_images[x,y,:] = column
    random_images = xr.DataArray(image_np_array,
        dims=('inputs_x', 'inputs_y', 'cases')).transpose(
            'cases', 'inputs_x', 'inputs_y').stack(inputs=('inputs_x', 'inputs_y'))
    return random_images