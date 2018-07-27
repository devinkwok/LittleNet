import numpy as np
import xarray as xr


ar = xr.Dataset({
    'weights': (['inputs', 'neurons'], np.ones((3, 2))),
    'biases': (['neurons'], np.ones((2)))})
w1 = ar['weights']
b1 = ar['biases']
ip = xr.DataArray(np.ones((3)), dims=('inputs'))
c = xr.dot(ip, w1, dims=('inputs', 'inputs'))
print(c)
print(np.add(c, b1))
# print(c.sizes == {'neurons': 2})
print(w1.sel(inputs=0))
print(w1.sum(dim='inputs'))

# data = xr.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])
data = xr.DataArray(np.zeros((2, 3, 10)), coords={'onehot': [x for x in range(100, 1100, 100)]}, dims=('batches', 'cases', 'onehot'))

training_set = xr.Dataset(
    {'training_inputs': (['batches', 'cases', 'inputs'], np.zeros((2, 3, 4))),
    'training_labels': (['batches', 'cases', 'label_onehot'], np.zeros((2, 3, 2)))},
    coords={'label_onehot': [i for i in range(2)]} )
print(training_set)
print(training_set['training_labels'])
