"""
First attempt to train a neural network,
Avoiding biases for now to keep the computation
a little simpler.
"""

import numpy as np


class Network(object):
    def __init__(self, sizes):
        self._layers = len(sizes)
        self._sizes = sizes
        self._weights = [np.random.randn(x, y) for x, y in
                         zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, inp):
        for weight in self._weights:
            inp = sigmoid(inp.dot(weight))
        return inp

    def back_propagate(self, inp, expected):
        # Feed Forward
        zs = []
        activations = [inp]
        a = inp
        for weight in self._weights:
            z = a.dot(weight)
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        errors = (a - expected) ** 2
        error_prime = 2 * (expected - a) 

        # First Iteration
        l_z       = zs[-1]
        l_z_prime = sigmoid_prime(l_z)
        l_a       = activations[-2]
        l_ze      = error_prime * l_z_prime  # [A, B], delta
        d_w       = l_a.transpose().dot(l_ze)

        del_weights = [d_w]

        # Rest of the iterations
        for i in range(2, self._layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            activation = activations[-i-1]
            weight = self._weights[-i+1]
            l_ze = l_ze.dot(weight.T) * sp
            d_w = activation.T.dot(l_ze)
            del_weights.append(d_w)

        for i, weight in enumerate(self._weights):
            del_w = del_weights[-i-1]
            self._weights[i] = weight + del_w
        return self._weights

"""Sigmoid Function"""
def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z) )


"""Derivative of the sigmoid function."""
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


"""
Train for single input and test the same
Baby Step #1
"""
net = Network([4,3,3,2])
print net.feed_forward(np.array([[1,2,3,4]]))

for i in range(10000):
    net.back_propagate(np.array([[1,2,3,4]]), np.array([[1,0.1]]))


print net.feed_forward(np.array([[1,2,3,4]]))

"""
Step 2: train the same for a huge training set & Test
"""
