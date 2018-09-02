"""Functions for calculating and combining activation probability distributions (APD)
APDs are non-parametric probability distributions of the intermediate outputs of a
neural network. Currently APDs are modelled as histograms."""

import numpy as np
import xarray as xr
import neural_net as nn

DIM_HIST = 'histogram_buckets'
"""xarray dimension key for histogram buckets"""

DIM_PAIR = 'pairs'
"""xarray dimension key for probabilities of pairs of activations"""


def apd_raw(activations, num_buckets=10):
    """Returns activation probability histogram for every individual activation.
        Activations are in the interval [0., 1.].
        Histogram values are  linearly interpolated between adjacent buckets.
        First and last buckets are half sized (no adjacent bucket to interpolate to).
        Example: for 3 buckets centered at [0., 0.5, 1.], histograms would be
            0. -> [1., 0., 0.]
            0.1 -> [0.8, 0.2, 0.]
            0.75 -> [0., 0.5, 0.5]

    Arguments:
        activations {xarray} -- output values of neurons

    Keyword Arguments:
        num_buckets {int} -- number of histogram buckets, where
            the first and last buckets are half-sized (default: {10})

    Returns:
        xarray -- histograms along dim=DIM_HIST, preserves every individual activation
    """

    scaled = activations * (num_buckets - 1)
    histogram = [np.maximum(1 - np.abs(b - scaled), 0)
                 for b in range(num_buckets)]
    return xr.concat(histogram, dim=DIM_HIST).transpose(*activations.dims, DIM_HIST)


def apd_area(apds):
    """Sums areas of histogram buckets into total probability.

    Arguments:
        apds {xarray} -- array of histograms encoded along dim=DIM_HIST

    Returns:
        xarray -- array with single sum replacing each histogram
    """

    # first and last buckets are half sized, so subtract this from the sum total
    end_bucket_excess = apds.isel(histogram_buckets=0) * 0.5 + \
        apds.isel(histogram_buckets=-1) * 0.5
    return apds.sum(dim=DIM_HIST) - end_bucket_excess


def cost(apds, dim=DIM_HIST):
    """Returns mean squared cost for a set of histograms. Differs from
        neural_net.mean_squared_error() as it accounts for the half sized
        histogram buckets at the beginning and end.

    Arguments:
        apds {xarray} -- array of values to find mean squared error

    Keyword Arguments:
        dim {str} -- dimension along which to sum, if DIM_HIST, sum
            accounts for half sized buckets at ends (default: {DIM_HIST})

    Returns:
        xarray -- mean squared cost along dim
    """

    if dim == DIM_HIST:  # account for half size buckets
        return apd_area(np.square(apds)) / (apds.sizes[dim] - 1)
    else:               # regular mean squared error
        return np.square(apds).sum(dim) / apds.sizes[dim]


def diff(apd1, apd2):
    """Finds difference of two probability distributions. The area sum
        of the difference should be in the interval [0., 2.], since each
        probability difference has an area of 1., and two distributions
        can range from identical to completely divergent.

    Arguments:
        apd1 {xarray} -- xarray with histogram buckets along DIM_HIST
        apd2 {xarray} -- xarray with histogram buckets along DIM_HIST

    Raises:
        ValueError -- Probability distributions must both contain dimension DIM_HIST

    Returns:
        xarray -- absolute difference of apd1 and apd2
    """

    if not DIM_HIST in apd1.dims or not DIM_HIST in apd2.dims:
        raise ValueError(
            'Probability distributions must both contain dimension', DIM_HIST)
    return np.absolute(apd1 - apd2)


def merge(apds, *dims_list):
    """Combines probability distributions along given dimensions.
        Returns the mean of multiple individual probabilities in apds.
        Reduces size of apds for easier manipulation and less memory use.

    Arguments:
        apds {xarray} -- probability distributions
        *dims_list {str} -- dimensions to find means along and remove,
            cannot remove DIM_HIST, use apd_area() or cost() instead

    Returns:
        xarray -- probability distributions with dimensions in dims_list combined
    """

    dims = dims_list
    if not dims_list:
        dims = apds.dims
    dims = list(dims)
    if DIM_HIST in dims:
        dims.remove(DIM_HIST)
    return apds.mean(dim=dims)


def merge_ex(apds, *dims_to_exclude):
    """Same as merge(), but preserving certain dimensions instead of removing
    certain dimensions.

    Arguments:
        apds {xarray} -- probability distributions
        *dims_to_exclude {str} -- dimensions to preserve, all other dimensions
            excluding DIM_HIST are sent to merge()

    Returns:
        xarray -- probability distributions with dimensions in dims_list combined
    """

    dims = list(apds.dims)
    for dimension in dims_to_exclude:
        if dimension in dims:
            dims.remove(dimension)
    return merge(apds, *dims)


def subset(raw_apd, **dict_dims_to_selected):
    """Merges probability distributions for selected indexes. Used to find the
        probability distribution of a subset of values, such as for labels that equal 1.
        Uses np.where() to select values.

    Arguments:
        raw_apd {xarray)} -- probability distributions containing dimensions
            from dict_dims_to_selected
        **dict_dims_to_selected {dict[str:list(bool)]} -- dictionary of dims along
            which to select items, each dim has a boolean array of the same size
            as raw_apd.sizes[dim], which indicates the indexes to select.

    Returns:
        xarray -- subset of probability distributions merged along dims
    """

    apds = raw_apd
    for dim, selected_bool_arr in dict_dims_to_selected.items():
        apds = apds.isel({dim: np.where(selected_bool_arr)[0]})
    return merge(apds, *dict_dims_to_selected.keys())


def subset_by_label(raw_apd, labels, symbols, dims_to_exclude=[nn.DIM_IN]):
    """Divides probability distribution into subsets by label.

    Arguments:
        raw_apd {xarray} -- probability distributions containing dimensions
            nn.DIM_CASE and nn.DIM_LABEL
        labels {xarray} -- list of labels associated with each activation,
            same as used in utility.make_onehot(), do NOT use onehot vector
        symbols {list} -- list of int symbols in labels, same as used in
            utility.make_onehot()

    Keyword Arguments:
        dims_to_exclude {list} -- dimensions to preserve and not merge along
            (default: {[nn.DIM_IN]})

    Returns:
        xarray -- probability distributions for each label along DIM_LABEl
    """

    label_apd = [merge_ex(subset(
        raw_apd, **{nn.DIM_CASE: np.equal(labels, i)}), *dims_to_exclude) for i in symbols]
    return xr.concat(label_apd, nn.DIM_LABEL).transpose(nn.DIM_LABEL, *label_apd[0].dims)


def diff_by_label(activations, labels, symbols=range(10),
                  dims_to_exclude=[nn.DIM_IN], num_buckets=10):
    """Finds difference between probability distributions of each label and
        the total probability distribution.

    Arguments:
        activations {xarray} -- neuron output activations
        labels {xarray} -- list of labels per activation, NOT onehot

    Keyword Arguments:
        symbols {list} -- symbols in labels (default: {range(10)})
        dims_to_exclude {list} -- dimensions to preserve (default: {[nn.DIM_IN]})
        num_buckets {int} -- number of histogram buckets (default: {10})

    Returns:
        xarray -- difference of probability distributions along DIM_LABEl
    """

    raw_apd = apd_raw(activations, num_buckets=num_buckets)
    total_apd = merge_ex(raw_apd, *dims_to_exclude)
    label_apd = subset_by_label(raw_apd, labels, symbols, dims_to_exclude)
    return diff(total_apd, label_apd).transpose(nn.DIM_LABEL, *total_apd.dims)


def triangular_to_linear_index(i, j, size):
    """Turns a triangular 2-d array of x-y pairs to linear indexes,
        for tabulating unique pairs of elements: the combination C(n, 2), code from
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    Arguments:
        i {np_array} -- indexes of first element
        j {np_array} -- indexes of second element
        size {int} -- total number of pairs

    Returns:
        np_array -- indexes of pairs
    """

    return (size*(size-1)/2) - (size-i)*((size-i)-1)/2 + j - i - 1


def linear_to_triangular_indexes(k, size):
    """Turns a linear index of x-y pairs into coordinates for a triangular 2-d array,
        for tabulating unique pairs of elements: the combination C(n, 2), code from
        https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix

    Arguments:
        k {np_array} -- indexes of pair
        size {int} -- total number of pairs

    Returns:
        xarray -- xarray of pairs, where the indexes of the first and second
            elements are encoded along dim=
    """

    i = size - 2 - np.floor(np.sqrt(-8 * k + 4 * size * (size - 1) - 7) / 2.0 - 0.5)
    j = k + i + 1 - size*(size-1)/2 + (size-i)*((size-i)-1)/2
    return np.array([i, j]).transpose().reshape((len(i), 2))


def diff_by_pairs(activations, labels, symbols=range(10), dim_to_pair=nn.DIM_IN, num_buckets=10):
    """Finds difference between probability distributions of pairs of neurons and
    the total probability distribution.

    Arguments:
        activations {xarray} -- neuron output activations
        labels {xarray} -- list of labels per activation, NOT onehot

    Keyword Arguments:
        symbols {list} -- symbols in labels (default: {range(10)})
        dims_to_exclude {list} -- dimensions to preserve (default: {[nn.DIM_IN]})
        num_buckets {int} -- number of histogram buckets (default: {10})

    Returns:
        xarray -- difference of probability distributions by pairs,
            the indexes of the 2 neurons in each pair can be retrieved using
            triangular_to_linear_index() and linear_to_triangular_indexes()
    """

    num_items = activations.sizes[dim_to_pair]
    apds = apd_raw(activations, num_buckets=num_buckets)
    pairs = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            apd1 = subset_by_label(
                apds.isel({dim_to_pair: i}), labels, symbols, dims_to_exclude=[dim_to_pair])
            apd2 = subset_by_label(
                apds.isel({dim_to_pair: j}), labels, symbols, dims_to_exclude=[dim_to_pair])
            pairs.append(diff(apd1, apd2))
    return xr.concat(pairs, DIM_PAIR)
