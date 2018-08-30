import numpy as np

#tensor is a dict with fields
#each field is a np array of of np arrays
# weights are inputs x outputs in size
# biases 1 x outputs in size

def create_tensor(layer_inputs, seed=0, fillFunction=lambda s: np.random.normal(size=s)):
    # np.random.seed(seed)
    tensor = {}
    layers = len(layer_inputs) - 1
    tensor['w'] = np.empty(layers, dtype=object)
    tensor['b'] = np.empty(layers, dtype=object)
    for i, (m, n) in enumerate(zip(layer_inputs[0:],layer_inputs[1:])):
        tensor['w'][i] = fillFunction((m,n))
        tensor['b'][i] = fillFunction((1,n))
    return tensor

def numerical_derivative(inputs, t, labels):
    layers = len(t['w'])
    grad = t_copy_shape(t)
    tDelta = t_copy_shape(t)
    for l in range(layers):
        tDelta['w'][l] = np.copy(t['w'][l])
        tDelta['b'][l] = np.copy(t['b'][l])
    #get value of function
    c = cost(pass_forward(inputs, t)[-1], labels)
    delta = 0.000000001
    #for each parameter, add delta value and get function value to get slope
    for n in ['w', 'b']:
        for l in range(layers):
            for i in np.ndindex(tDelta[n][l].shape):
                parameter = tDelta[n][l][i]
                tDelta[n][l][i] = parameter + delta
                cDelta = cost(pass_forward(inputs, tDelta)[-1], labels)
                grad[n][l][i] = (cDelta - c) / delta
                tDelta[n][l][i] = parameter
    return grad

def visualize_output(t, sample_output):
    layers = len(t['w'])
    # use output as input
    inputs = sample_output
    # for each layer
    for i in range(layers - 1, -1, -1):
        # apply inverse of activation function
        inputs = inverse_activation(inputs)
        # add bias
        inputs = np.subtract(inputs, np.transpose(t['b'][i]))
        # distribute to inputs by {input weight} / {sum of all weights}
        weight_sums = np.sum(t['w'][i], axis=0)
        weight_proportion = np.divide(t['w'][i], weight_sums)
        inputs = np.dot(weight_proportion, inputs)
    # repeat until bottom layer of network
    # return inputs as output
    return inputs

def t_copy_shape(t, fillFunc=lambda s: np.zeros(s)):
    dims = [t['w'][0].shape[0]]
    for b in t['b']:
        dims.append(b.size)
    return create_tensor(tuple(dims), fillFunction=fillFunc)

def t_equal(t1, t2):
    for key in t1:
        for a, b in zip(t1[key], t2[key]):
            if a.shape != b.shape:
                return False
            if not np.all(np.equal(a, b)):
                return False
    return True

#applies function to each numpy array in tensors
# first argument after function must be tensor
def t_eval(function, *tensors):
    tout = t_copy_shape(tensors[0])
    for key in tensors[0]:
        args = []
        for t in tensors:
            if isinstance(t, dict):
                args.append(t[key])
            else:
                args.append(t)
        tout[key] = function(*args)
    return tout

#for debugging
def t_print(t):
    print("TENSOR: ")
    for i in range(len(t['w'])):
        print(i, ' w ', t['w'][i].shape, 'b ', t['b'][i].shape)

### backprop algorithms
    # i: inputs
    # w: weights
    # b: biases
def apply(i, w, b):
    return np.add(np.dot(i, w), b)

    # sigmoid
def activation(i):
    return np.divide(1, np.add(1, np.exp(np.multiply(-1, i))))

    # inverse of sigmoid
def inverse_activation(i):
    return i #TODO

    # sigmoid derivative
def d_activation(i):
    # e = np.exp(i)
    # return np.divide(e, np.square(np.add(e, 1)))
    a = activation(i)
    return np.multiply(a, (np.subtract(1, a))) 
    # return i

### returns input and all intermediary outputs
def pass_forward(input, t):
    layers = len(t['w'])
    output = np.empty(layers + 1, dtype=object)
    output[0] = np.reshape(input, (1, input.size))
    for i in range(layers):
        output[i+1] = activation(apply(output[i], t['w'][i], t['b'][i]))
    return output

### returns intermediary outputs before applying the activation function (for backprop)
def pass_forward_no_activation(input, t):
    layers = len(t['w'])
    records = np.empty(layers + 1, dtype=object)
    records[0] = np.reshape(input, (1, input.size))
    output = records[0]
    for i in range(layers):
        records[i+1] = apply(output, t['w'][i], t['b'][i])
        output = activation(records[i+1])
    return records

def cost(output, labels):
    # return np.divide(np.sum(np.square(np.subtract(labels, output))), len(output))
    return np.sum(np.subtract(labels, output))

def cost_gradient(output, labels):
    return np.subtract(output, labels)
    # return np.multiply(2, np.subtract(output, labels))

#TODO: get input sensitivity
def pass_back_gradient():
    layers = len(t['w'])
    delta = t_copy_shape(t)
    grad = gradient
    for i in range(layers - 1, -1, -1):
        grad = np.reshape(grad, (1, grad.size))
        activation_d = d_activation(input[i+1])
        grad = np.multiply(grad, activation_d)
        delta['b'][i] = grad
        delta['w'][i] = np.multiply(np.transpose(input[i]), grad)
        grad = np.dot(t['w'][i], np.transpose(grad))
    return delta

def pass_back(input, gradient, t):
    layers = len(t['w'])
    delta = t_copy_shape(t) #copy tensor shape into gradient shape
    grad = gradient
    for i in range(layers - 1, -1, -1):
        grad = np.reshape(grad, (1, grad.size))
        activation_d = d_activation(input[i+1])
        grad = np.multiply(grad, activation_d)
        delta['b'][i] = grad

        # TODO: ?? ugly hack: only apply activation if it's not the original inputs
        a = input[i]
        if (i > 0):
            a = activation(a) # TODO: why does the activation need to be applied here?
        # TODO: not doing the ugly hack above should still work
        # a = activation(input[i])
        # TODO: however, not applying the activation makes it quite a lot worse
        # a = input[i]
        delta['w'][i] = np.multiply(np.transpose(a), grad)
        grad1 = np.dot(t['w'][i], np.transpose(grad))
        grad = grad1
    return delta

# batch training, averaging gradient
# learning rate
def train(data, labels, tensor, rate=0.01):
    return t_eval(np.add, tensor, gradient(data, labels, tensor, rate))

def gradient(data, labels, tensor, rate=0.01):
    g = t_copy_shape(tensor)
    for d, l in zip(data, labels):
        outputs = pass_forward_no_activation(d, tensor)
        gradient = cost_gradient(activation(outputs[-1]), l)
        delta = pass_back(outputs, gradient, tensor)
        g = t_eval(np.subtract, g, delta)
    return t_eval(np.multiply, g, rate / len(data))

def train_numerical(data, labels, tensor, rate=0.01):
    g = t_copy_shape(tensor)
    for d, l in zip(data, labels):
        delta = numerical_derivative(d, tensor, l)
        g = t_eval(np.subtract, g, delta)
    return t_eval(np.add, tensor, t_eval(np.multiply, g, rate / len(data)))

def run(data, tensor):
    outputs = []
    for d in data:
        outputs.append(pass_forward(d, tensor)[-1])
    return outputs

def analyze(outputs, labels):
    correct = 0.
    for a, b in zip(outputs, labels):
        if np.argmax(a) == np.argmax(b):
            correct += 1
    # print('asdf', correct, len(outputs))
    return correct / len(outputs)

#gets a random selection of n integers no larger than data_size
def random_batch(data_size, n):
    a = np.random.randint(data_size, size=n)
    # print(a)
    return a

def learn(t, norm_data, onehot, minibatch=100, period=50, epochs=100, tests=30, rate=0.03):
    np.random.seed(0)
    
    for i in range(epochs):
        for j in range(period):
            r = random_batch(norm_data.shape[0], minibatch)
            t = train(norm_data[r,:], onehot[r,:], t, rate)

        np.random.seed(i*period+j)
        test_r = random_batch(norm_data.shape[0], tests)
        results = run(norm_data[test_r,:], t)
        labels = onehot[test_r,:]
        accuracy = analyze(results, labels)
        # for a, b in zip(results[:5], labels[:5]):
        #     print('\n', np.around(a, decimals=2), '\n', b)
        print((str(i * period + j) + ' acc:' + str(accuracy)).ljust(80), end="\r")
    
    return t
