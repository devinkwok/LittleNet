# calculate activation probability distributions
# put into a sql like database
# query APD between 2 sets of data
# example: apd(all, label=0)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import neural_net as nn
import utility

# new stuff
def cost(apds, dim='histogram_buckets'):
    if dim == 'histogram_buckets': # account for half size buckets
        return apd_area(np.square(apds)) / (apds.sizes[dim] - 1)
    else: # regular mean squared error
        return np.square(apds).sum(dim) / apds.sizes[dim]

def triangular_to_linear_index(i, j, n):
    # from https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1

def linear_to_triangular_indexes(k, n):
    # from https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    i = n - 2 - np.floor(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    pairs = xr.concat([i, j], dim='pair_a pair_b')
    return pairs.transpose(*k.dims, 'pair_a pair_b')

def apd_raw(activations, num_buckets=10):
    scaled = activations * (num_buckets - 1)
    histogram = [np.maximum(1 - np.abs(b - scaled), 0) for b in range(num_buckets)]
    return xr.concat(histogram, dim='histogram_buckets').transpose(*activations.dims, 'histogram_buckets')

def apd_area(apds):
    # first and last buckets are half sized, so subtract this from the sum total
    diff = apds.isel(histogram_buckets=0) * 0.5 + apds.isel(histogram_buckets=-1) * 0.5
    return apds.sum(dim='histogram_buckets') - diff

# difference of activation probability distributions
def diff(apd1, apd2):
    return np.absolute(apd1 - apd2)

def merge(apds, *dims_list):
    dims = dims_list
    if not dims_list:
        dims = apds.dims
    dims = list(dims)
    if 'histogram_buckets' in dims:
        dims.remove('histogram_buckets')
    return apds.mean(dim=dims)

def merge_ex(apds, *dims_to_exclude):
    dims = list(apds.dims)
    for d in dims_to_exclude:
        if d in dims:
            dims.remove(d)
    return merge(apds, *dims)

def subset(raw_apd, **dict_dims_to_selected):
    apds = raw_apd
    for dim, selected_bool_arr in dict_dims_to_selected.items():
        apds = apds.isel({dim: np.where(selected_bool_arr)[0]})
    return merge(apds, *dict_dims_to_selected.keys())

def subset_by_label(raw_apd, labels, symbols, dims_to_exclude=['inputs']):
    label_apd = [merge_ex(subset(
        raw_apd, cases=np.equal(labels, i)), *dims_to_exclude) for i in symbols]
    return xr.concat(label_apd, 'labels').transpose('labels', *label_apd[0].dims)

def diff_by_label(activations, labels, symbols=range(10), dims_to_exclude=['inputs'], num_buckets=10):
    raw_apd = apd_raw(activations, num_buckets=num_buckets)
    total_apd = merge_ex(raw_apd, *dims_to_exclude)
    label_apd = subset_by_label(raw_apd, labels, symbols, dims_to_exclude)
    return diff(total_apd, label_apd).transpose('labels', *total_apd.dims)

def diff_by_pairs(activations, labels, symbols=range(10), dim_to_pair='inputs', num_buckets=10):
    num_items = activations.sizes[dim_to_pair]
    apds = apd_raw(activations, num_buckets=num_buckets)
    pairs = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            apd1 = subset_by_label(apds.isel({dim_to_pair: i}), labels, symbols, dims_to_exclude=[dim_to_pair])
            apd2 = subset_by_label(apds.isel({dim_to_pair: j}), labels, symbols, dims_to_exclude=[dim_to_pair])
            pairs.append(diff(apd1, apd2))
    return xr.concat(pairs, 'pairs')