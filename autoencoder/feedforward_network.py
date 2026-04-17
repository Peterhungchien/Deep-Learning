# I have heard stuffs like gradient descent, computation graph,
# and backward propagation, but I currently do not have
# any ideas on implementing them.
# So I will just start with a simple feedforward network
# with random weights and biases to understand the
# structure of a network.

import numpy as np
from collections.abc import Callable


# There are many kinds of layers out there,
# here I start with the simplest one,
# a dense layer that accepts a vector as input
# and returns the activated output vector.
class DenseLayer:
    input_size: int
    output_size: int
    activation_func: Callable[[np.ndarray], np.ndarray]
    # Here weights also include the bias,
    # an extra constant column is added to the input vector to accommodate the bias.
    weights: np.ndarray

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_func: Callable[[np.ndarray], np.ndarray],
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.weights = np.random.rand(output_size, input_size + 1) * 0.01

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        # concat the input vector with a constant 1 to accommodate the bias.
        input_with_bias = np.concatenate((input_vector, np.array([1])))
        preactivation: np.ndarray = np.dot(self.weights, input_with_bias)
        activation = self.activation_func(preactivation)
        return activation


# Built on top of this,
# a simplified version of neural network
# is just an ordered collection of layers.
class NeuralNetwork:
    layers: list[DenseLayer]

    def __init__(self, layers: list[DenseLayer]):
        self.layers = layers

    # Make the network callable, so that we can just call it with an input vector to get the output.
    def __call__(self, input_vector: np.ndarray) -> np.ndarray:
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        return output


# Also define two activation functions.
# Since we are not building a computation graph here,
# just write them as functions.


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# Create a simple network with one hidden layer and one output layer.
test_input = np.array([0.5, 0.2, 0.1])
network = NeuralNetwork([DenseLayer(3, 4, ReLU), DenseLayer(4, 1, sigmoid)])
print(network(test_input))
