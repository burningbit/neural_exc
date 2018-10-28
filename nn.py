"""
ToDo:
Refactor code, rename variables.

First attempt to train a neural network,
Added SGD & Biases, to enhance accuracy.
Using biases to shift the result
"""

import numpy as np
import random
import time
import os
import sys

class Network(object):
    """
    If network receives biases & wights from pre-trained network
    It initializes the aforementioned with provided values
    """
    def __init__(self, sizes, biases=None, weights=None):
        self._layers = len(sizes)
        self._sizes = sizes
        self._biases = biases
        if biases == None:
            self._biases = [np.random.randn(1, y) for y in sizes[1:]]
        self._weights = weights
        if weights == None:
            self._weights = [np.random.randn(x, y) for x, y in
                         zip(sizes[:-1], sizes[1:])]
            
    def feed_forward(self, inp):
        for weight, bias in zip(self._weights, self._biases):
            inp = sigmoid(np.dot(inp, weight) + bias)
        return inp

    """
    An attempt to implement Stochastic Gradient Descent
    """
    def SGD(self, training_data, epochs=60, mini_batch_size=10, eta=0.261):
        total_iterations = (len(training_data) / mini_batch_size) * epochs
        for _ in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[p:p+mini_batch_size] for p
                            in xrange(0, len(training_data), mini_batch_size)]
            l_mini_batch = len(mini_batches)
            # Update network in batches
            for idx, mini_batch in enumerate(mini_batches):
                del_weights = [np.zeros(w.shape) for w in self._weights]
                del_biases = [np.zeros(b.shape) for b in self._biases]
                for inp, expected in mini_batch:
                    d_w, d_b = self.back_propagate(inp, expected)
                    del_weights = [dbw+dw for dbw, dw in zip(del_weights,
                                                             d_w)]
                    del_biases = [dbb+db for dbb, db in zip(del_biases,
                                                              d_b)]
                self._weights = [w - (eta/len(mini_batch)) * dw for w, dw in
                                 zip(self._weights, del_weights)]
                self._biases = [b - (eta/len(mini_batch)) * db for b, db in
                                zip(self._biases, del_biases)]
                progress(_ * l_mini_batch + idx, total_iterations)
        return self._weights, self._biases

    def back_propagate(self, inp, expected):
        # Feed Forward
        zs = []
        activations = [inp]
        a = inp
        for bias, weight in zip(self._biases, self._weights):
            z = np.dot(a, weight) + bias
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        error_prime = (a - expected) 

        # First Iteration
        l_z       = zs[-1]
        l_z_prime = sigmoid_prime(l_z)
        l_a       = activations[-2]
        l_ze      = error_prime * l_z_prime  # [A, B], delta
        d_w       = np.dot(l_a.T, l_ze)

        del_weights = [d_w]
        del_biases = [l_ze]

        # Rest of the iterations
        for i in range(2, self._layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            activation = activations[-i-1]
            weight = self._weights[-i+1]
            l_ze = np.dot(l_ze, weight.T) * sp
            d_w = np.dot(activation.T, l_ze)
            del_weights.append(d_w)
            del_biases.append(l_ze)

        # ToDo: Refactor by initializing and updating in place \
        # to avoid reverse
        del_weights.reverse()
        del_biases.reverse()
        return (del_weights, del_biases)
    
"""Sigmoid Function"""
def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z) )


"""Derivative of the sigmoid function."""
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# Print iterations progress
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 2)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
