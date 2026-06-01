import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            out = self.forward(x)
            loss = np.mean((y - out) ** 2)
            gradient = 2 * (out - y) / len(y)
            self.backward(gradient, learning_rate)
            if epoch % 1000 == 0:
                print(f'epoch {epoch} | Loss: {round(loss, 4)}')
        return out