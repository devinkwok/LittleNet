import numpy as np
import xarray as xr


# ar = xr.Dataset({
#     'weights': (['inputs', 'neurons'], np.ones((3, 2))),
#     'biases': (['neurons'], np.ones((2)))})
# w1 = ar['weights']
# b1 = ar['biases']
# ip = xr.DataArray(np.ones((3)), dims=('inputs'))
# c = xr.dot(ip, w1, dims=('inputs', 'inputs'))
# # print(c)
# # print(np.add(c, b1))
# # print(c.sizes == {'neurons': 2})
# # print(w1.sel(inputs=0))
# # print(w1.sum(dim='inputs'))

# # data = xr.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])
# data = xr.DataArray(np.zeros((2, 3, 10)), coords={'onehot': [x for x in range(100, 1100, 100)]}, dims=('batches', 'cases', 'onehot'))

training_set = xr.Dataset(
    {'training_inputs': (['labels_onehot', 'cases', 'inputs'], np.arange(2*3*4).reshape((2, 3, 4))),
    'training_labels': (['batches', 'cases', 'label_onehot'], np.zeros((2, 3, 2)))},
    coords={'label_onehot': [i for i in range(2)]} )
# # print(training_set)
# # print(training_set['training_labels'])
# aa = w1.rename(inputs='asdf', neurons='badd')
# aa['asdf'] = ('asdf', ['a', 'b', 'c'])
# aa['badd'] = ('badd', [111, 222])
# # print(aa)
# x = None
# for i in aa['badd']:
#     print(i)
# print(aa.sizes['asdf'])

# asdf = {'a': 1, 'b': 2}
# for x in asdf.keys():
#     print(x)

# # print(aa['asdf'])
# # print(x.coords)
# # for property, value in vars(x.coords).items():
# #     print ('asdf', property, ": ", value)
# # [print(x.coords, y.coords) for x,y in aa]

a = xr.DataArray(np.zeros((3, 2, 1, 1)), dims=['batches', 'cases', 'labels_onehot', 'asdf'])
# # a = training_set['training_inputs']
# a = a.rolling(batches=1)
# for x, b in a:
#     print(b.squeeze())

a = a.squeeze(dim='asdf')
print(a.size)