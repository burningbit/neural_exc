"""
ToDo: Refactor Code
"""
import os
import numpy as np
from mnist import MNIST
from nn import Network


resource_path = 'res'

mndata = MNIST(os.path.join(resource_path, 'dataset'))

# Training data
images, labels = mndata.load_testing()
images = images
labels = labels
# Initialize network
network = Network([784, 16, 16, 10])

# Prepare data
result_labels = map(lambda x: np.array([[0] * x + [1] + [0] * (10-x-1)]),
                     labels)
image_inps = map(lambda x: np.array([x]), images)

# Train Network
weights, biases = network.SGD(zip(image_inps, result_labels))

# Store data in the provided path.
cache_path = os.path.join(resource_path, 'cache')
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
# Save weights
np.save(os.path.join(cache_path, 'weights.v'), weights)
# np.save(os.path.join(cache_path, 'biases.v'), biases)

print "Running test cases ..."
# Load test data
# images, labels = mndata.load_testing()
trues = 0
for i in range(len(images)):
    result_vector = network.feed_forward(np.array([images[i]]))
    result_vector = result_vector
    result_list = list(result_vector[0])
    result = result_list.index(max(result_list))
    if result == labels[i]: trues += 1

print "Accuracy: {}%".format(float(trues)/len(images) * 100)
