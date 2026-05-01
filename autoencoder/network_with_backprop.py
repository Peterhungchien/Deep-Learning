from typing import Protocol
import numpy as np
from collections.abc import Callable

# In the previous version,
# I implemented a neural network
# class with feedforward method only.
# Now I will add backpropagation method
# to the class to make it trainable,
# and will verify it by comparing the result
# with the one calculated by numerical differentiation.

# Since we need the gradient of activation functions,
# a bare function definition is not enough,
# we need to define a class with a gradient method.
# It turns out that PyTorch splits a layer we defined
# last time (a linear map with an activation function)
# into two separate layers: a linear layer and an activation layer.
# This allows us to decouple the activation function from the linear layer,
# offering more composability and flexibility
# in designing the network architecture.


class Layer(Protocol):
    def forward(self, input_vector: np.ndarray) -> np.ndarray: ...
    def backward(
        self, input_vector: np.ndarray, output_gradient: np.ndarray
    ) -> np.ndarray: ...


class ReLU:
    input: np.ndarray | None

    def __init__(self):
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.max(0, input)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Linear:
    input: np.ndarray | None
    weight: np.ndarray
    bias: np.ndarray
    grad_weight: np.ndarray
    grad_bias: np.ndarray

    def __init__(self, input_dim: int, output_dim: int):
        self.input = None
        self.weight = np.random.randn(input_dim, output_dim)
        self.grad_weight = np.zeros((input_dim, output_dim))
        self.bias = np.random.randn(output_dim)
        self.grad_bias = np.zeros(output_dim)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.weight + self.bias

    def backward(self, grad_output):
        # In this script we assume the input data has
        # the shape (num_sample, num_feature),
        # otherwise there would be a dimension mismatch
        # and we need to handle it by summing over the
        # extra axes manually.
        # In PyTorch, the autograd module takes care
        # of this automatically.
        self.grad_weight = self.input @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weight.T


class NeuralNetwork:
    layers: list[Layer]
    loss_function: Callable[[np.ndarray, np.ndarray], float]

    def __init__(
        self,
        layers: list[Layer],
        loss_function: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.layers = layers
        self.loss_function = loss_function

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        return output


test_input = np.array([0.5, 0.2, 0.1])
