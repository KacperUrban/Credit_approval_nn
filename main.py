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
        self.bias = self.bias - learning_rate * output_gradient
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
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, s_d = 0.07, s_i = 1.05, er = 1.04, verbose = True):
    previous_error = 0
    tmp = 0
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
        if tmp > 0:
            if error > er * previous_error:
                learning_rate = learning_rate * s_d
            elif error < previous_error:
                learning_rate = learning_rate * s_i
        tmp += 1
        previous_error = error


def test(network, X_t, y_test):
    error = 0
    sum = 0
    accuracy = 0
    for x, y in zip(X_t, y_test):
        # przejście "w przód"
        output = predict(network, x)

        # obliczanie błędu sieci
        error += mse(y, output)

        # obliczanie dokładności sieci
        if output > 0.5:
            output = 1
        else:
            output = 0

        if output == y:
            sum += 1
    error /= len(X_t)
    accuracy = sum / len(X_t) * 100
    return  f"Accuracy={accuracy}, error={error}"


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

y_full_encoded = encoding_y_full(y_full)

ordinal = OrdinalEncoder()
X_full_encoded = ordinal.fit_transform(X_full)

min_max_scaler = MinMaxScaler()

X_full_encoded_scaled = min_max_scaler.fit_transform(X_full_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_full_encoded_scaled, y_full_encoded, test_size=0.2, random_state=42)
X_tr = np.reshape(X_train, (len(X_train), 15, 1))
X_t = np.reshape(X_test, (len(X_test), 15, 1))
network = [
    Dense(15, 10),
    Tanh(),
    Dense(10, 5),
    Tanh(),
    Dense(5, 1),
    Sigmoid()
]

# train
train(network, mse, mse_prime, X_tr, y_train, epochs=200, learning_rate=0.1, er= 1.02)

#test
print(test(network, X_t, y_test))