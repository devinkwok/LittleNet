LittleNet: a simple neural network with exploration tools
=========================================================
An implementation of a simple neural network that trains on the MNIST dataset. I wrote this to get a deeper understanding of neural networks without using existing libraries (Tensorflow, Keras). The code prioritizes being easier to read and extend, rather than brevity and cleverness. Experiments currently being conducted with this network include:

 - [Using activation probabilities to choose initial parameters](./experiment-init-apd.md)
 - [Manually designing initial parameters](./experiment-init-kernel.md)

Requirements
------------
 - __Linux__: As development has been done on Linux, code has not been tested elsewhere. File I/O operations may be broken in Windows.
 - __Python 3__: Because this is a standalone project, it doesn't need to interoperate with other codebases in python 2.
 - __numpy__: For fast array manipulation.
 - __xarray__: This is a convenient wrapper around numpy which gives each dimension names and optional coordinates. Although verbose, named coordinates are easier to reason about and follow in the code, especially when arrays have higher and higher numbers of dimensions. xarray also supports automatic broadcasting to higher numbers of dimensions, which makes batch training and convolutional neural networks easier to implement.
 - __pandas__: Used by xarray for coordinates and indexing.
 - __matplotlib__: Used by xarray for plotting.

Installation
------------
Download LittleNet in desired directory:

```
git clone ./littlenet
cd littlenet
```

If needed, download the MNIST training images and labels to ./mnist_data/:

```
mkdir mnist_data
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output mnist_data/train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output mnist_data/train-labels-idx1-ubyte.gz
```

Install with setup.py:

```
python3 setup.py install
```

Run unit tests if desired:

```
python3 -m unittest discover
```

Run experiment scripts if desired:

```
python3 experiment-init-kernel.py
python3 experiment-init-apd.py
```

Usage
-----
Apart from the neural_net.NeuralNet object, code is written in a functional programming style. Parameters and return types for all functions are in the comments and docstrings. Detailed documentation for xarray can be found at [http://xarray.pydata.org/en/stable/](http://xarray.pydata.org/en/stable/).

 - __./__: Scripts for experiments are in the root directory.
 - __./littlenet__: Contains all of the package code.
 - __./littlenet/neural_net.py__: Contains the actual neural network object neural_net.NeuralNet, as well as some supporting functions.
 - __./littlenet/utility.py__: Contains file I/O and image manipulation functions.
 - __./littlenet/train.py__: Higher level functions for training neural networks.
 - __./littlenet/apd.py__: Functions for calculating activation probability distributions.
 - __./test__: Contains unit tests.
