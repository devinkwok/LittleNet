import struct
import pickle
import numpy as np

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.int8)

### resize data to fit in range, flatten array
def normalize(np_array):
    return np.multiply(np_array, 1/256.0)

def onehot_vector(labels, max_label=10):
    output = np.zeros((labels.size, max_label))
    output[np.arange(labels.size), labels] = 1
    return output

def read_tensor(filename):
    with open(filename, 'rb') as f:
        tensor = pickle.load(f)
        return tensor

def write_tensor(filename, tensor):
    with open(filename, 'wb') as f:
        pickle.dump(tensor, f)