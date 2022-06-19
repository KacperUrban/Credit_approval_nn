import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Tanh(Activation):
    def __init__(self):
        self.tanh = lambda x: np.tanh(x)
        self.tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(self.tanh, self.tanh_prime)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # przejście "w przód"
            output = predict(network, x)

            # obliczanie błędu sieci
            error += loss(y, output)

            # wsteczna propagacja
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")


def encoding_y_full(y):
    for i in range(0, 690):
        if y.iloc[i] == '+':
            y.iloc[i] = 1
        elif y.iloc[i] == "-":
            y.iloc[i] = 0
    return y


# Wczytanie danych za pomocą biblioteki pandas
credit = pd.read_csv("crx.data", header=None)

y_full = credit.iloc[:, 15]
X_full = credit.drop(columns=15)

print(y_full)
print(X_full)
