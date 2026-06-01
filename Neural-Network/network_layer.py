import numpy as np

class Layer:
    def __init__(self, inputs, neurons, activation):
        self.weights = np.random.randn(inputs, neurons) * 0.1
        self.bias = np.zeros((1, neurons))
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.weighed_sum = x @ self.weights + self.bias
        self.output = self.activation.forward(self.weighed_sum)
        return self.output

    def backward(self, gradient, learning_rate):
        gradient = self.activation.backward(gradient)
        d_weights = self.input.T @ gradient
        d_bias = np.sum(gradient, axis=0, keepdims=True)
        gradient_input = gradient @ self.weights.T
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return gradient_input