import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Ustawienie stałej wielkości okna wykresu
plt.rcParams["figure.figsize"] = [8, 5]


class Dense:
    """ Klasa odpowiedzialna za tworzenie warstw gęstych"""

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
    """Funkcja bazowa do tworzenia funkcji aktywacji"""

    def __init__(self, activation, activation_prime):
        """Inicjalizacja funkcji aktywacji oraz funkcj obliczającej jej pochodną. """
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Zaimplentownie wstecznej propagacji. Zmiana learning rate została użyta po to, aby można
           było tą funkcje używać w sposób uniwersalny."""
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    """ Sigmoidalna funkcja aktywacji"""

    def __init__(self):
        def sigmoid(x):
            """Funkcja init zwraca obliczoną wartość funkcji aktywacji"""
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            """Funkcja init zwraca obliczoną wartość funkcji aktywacji"""
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


def mse(y_true, y_pred):
    """Funkcja odpowiedzialana za obliczanie błędu średniokwardatowego"""
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """Funkcja odpowiedzialana za obliczanie pochodnej błędu średniokwardatowego"""
    return 2 * (y_pred - y_true) / np.size(y_true)


def predict(network, input):
    """Funkcja oblicza wartośc przewidywaną. Wywołuje funkcje forward dla każdej z warstw."""
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, s_d=0.07,
          s_i=1.05, er=1.04, verbose=True):
    """Funkcja jest odpowiedzialna za uczenie sieci neuronowej"""
    previous_error = 0
    tmp = 0
    errors = list()
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # przejście "w przód"
            output = predict(network, x)

            # obliczanie błędu
            error += loss(y, output)

            # wsteczna propagacja
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # Obliczenie błędu sieci
        error /= len(x_train)
        errors.append(error)
        # Stworzenie możliwości wyświetlania. Opcje można wyłączyć
        if verbose:
            print(f"Epoka = {e + 1}/{epochs}, Błąd={error}")
        # Zastosowanie adaptacyjnego współczynnika uczenia
        if tmp > 0:
            if error > er * previous_error:
                learning_rate = learning_rate * s_d
            elif error < previous_error:
                learning_rate = learning_rate * s_i
        tmp += 1
        previous_error = error
    return errors, epochs


def test(network, X_t, y_test):
    """Funkcja jest odpowiedzialna za testowanie sieci neuronowej"""
    error = 0
    sum = 0
    accuracy = 0
    for x, y in zip(X_t, y_test):
        # przejście "w przód"
        output = predict(network, x)

        # obliczanie błędu
        error += mse(y, output)

        # obliczanie dokładności sieci
        if output > 0.5:
            output = 1
        else:
            output = 0

        if output == y:
            sum += 1

    # obliczanie błędu dla sieci
    error /= len(X_t)
    accuracy = sum / len(X_t) * 100
    return accuracy, error


def encoding_y_full(y):
    """Funkcja koduje wartośc y z minusa i plusa na wartości zero oraz jeden"""
    for i in range(0, 690):
        if y.iloc[i] == '+':
            y.iloc[i] = 1
        elif y.iloc[i] == "-":
            y.iloc[i] = 0
    return y


# Wczytanie danych za pomocą biblioteki pandas
credit = pd.read_csv("crx.data", header=None)

# Podział na dane wejściowe oraz wyjściowe
y_full = credit.iloc[:, 15]
X_full = credit.drop(columns=15)

y_full_encoded = encoding_y_full(y_full)

# Zmienianie cech kategorialnych na wartości liczbowe
ordinal = OrdinalEncoder()
X_full_encoded = ordinal.fit_transform(X_full)

min_max_scaler = MinMaxScaler()

# Zamienianie wartości liczbowe na wartości pomiędzy zero a jeden
X_full_encoded_scaled = min_max_scaler.fit_transform(X_full_encoded)

# Podział na zestaw uczący oraz zestaw testowy
X_train, X_test, y_train, y_test = train_test_split(X_full_encoded_scaled, y_full_encoded, test_size=0.2,
                                                    random_state=42)

# Stworzenie odpowiednich wymiarów danych wejściowych
X_tr = np.reshape(X_train, (len(X_train), 15, 1))
X_t = np.reshape(X_test, (len(X_test), 15, 1))
network = [
    Dense(15, 10),
    Sigmoid(),
    Dense(10, 5),
    Sigmoid(),
    Dense(5, 1),
    Sigmoid()
]

# Trenowanie
errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=20, learning_rate=0.1, er=1.01)

# Rysowanie wykresu błędu w stosunku do epok
plt.plot(np.linspace(1, epoch + 1, num=epoch), errors)
plt.title("Błąd podczas trenowania sieci w stosunku do epok")
plt.ylabel("Błąd")
plt.xlabel("Epoki")
plt.grid()
plt.show()

# Testowanie nauczone sieci
test_accuracy, test_error = test(network, X_t, y_test)
print(f"Dokładność = {test_accuracy}, Błąd = {test_error}")

# Testowanie hiperparametrów
# Test nr. 1 - zmiana współczynnika uczenia
lr_list = [0.1, 0.2, 0.3, 0.4]
accuracy_list = list()
for i in range(0, 4):
    network = [
        Dense(15, 10),
        Sigmoid(),
        Dense(10, 5),
        Sigmoid(),
        Dense(5, 1),
        Sigmoid()
    ]

    # Trenowanie
    errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100, learning_rate=lr_list[i])
    test_accuracy, test_error = test(network, X_t, y_test)
    accuracy_list.append(test_accuracy)
plt.plot(lr_list, accuracy_list)
plt.title("Dokładność sieci w stosunku do różnych wartości współczynnika uczenia")
plt.ylabel("Dokładność")
plt.xlabel("Współczynnik uczenia")
plt.grid()
plt.show()
# Test nr. 2 - zmiana dopuszczalnego błędu
er_list = [1.01, 1.04, 1.05, 1.07]
accuracy_list = list()
for i in range(0, 4):
    network = [
        Dense(15, 10),
        Sigmoid(),
        Dense(10, 5),
        Sigmoid(),
        Dense(5, 1),
        Sigmoid()
    ]

    # Trenowanie
    errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100, er=er_list[i])
    test_accuracy, test_error = test(network, X_t, y_test)
    accuracy_list.append(test_accuracy)
plt.plot(er_list, accuracy_list)
plt.title("Dokładność sieci w stosunku do różnych wartości dopuszczalnego błędu")
plt.ylabel("Dokładność")
plt.xlabel("Błąd dopuszczalny")
plt.grid()
plt.show()
# Test nr. 3 - zmiana współczynnika zmniejszenia współczynnika uczenia
s_d_list = [0.01, 0.04, 0.06, 0.08]
accuracy_list = list()
for i in range(0, 4):
    network = [
        Dense(15, 10),
        Sigmoid(),
        Dense(10, 5),
        Sigmoid(),
        Dense(5, 1),
        Sigmoid()
    ]

    # Trenowanie
    errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100, s_d=s_d_list[i])
    test_accuracy, test_error = test(network, X_t, y_test)
    accuracy_list.append(test_accuracy)
plt.plot(s_d_list, accuracy_list)
plt.title("Dokładność sieci w stosunku do różnych współczynnika zmniejszania")
plt.ylabel("Dokładność")
plt.xlabel("Współczynnik zmniejszania")
plt.grid()
plt.show()
# Test nr. 4 - zmiana współczynnika zwiększania współczynnika uczenia
s_i_list = [1.02, 1.04, 1.06, 1.08]
accuracy_list = list()
for i in range(0, 4):
    network = [
        Dense(15, 10),
        Sigmoid(),
        Dense(10, 5),
        Sigmoid(),
        Dense(5, 1),
        Sigmoid()
    ]

    # Trenowanie
    errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100, s_i=s_i_list[i])
    test_accuracy, test_error = test(network, X_t, y_test)
    accuracy_list.append(test_accuracy)
plt.plot(s_i_list, accuracy_list)
plt.title("Dokładność sieci w stosunku do różnych wartości współczynnika zwiększania")
plt.ylabel("Dokładność")
plt.xlabel("Współczynnik zwiększania")
plt.grid()
plt.show()
# eksperyment nr. 1 - sprawdzanie dokładnośći w zależności od ilości neuronów w warstwach

accuracies = np.zeros((10, 10))

for i in range(1, 11):
    for j in range(1, 11):
        network = [
            Dense(15, i),
            Sigmoid(),
            Dense(i, j),
            Sigmoid(),
            Dense(j, 1),
            Sigmoid()
        ]
        errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100)
        accuracy, _ = test(network, X_tr, y_train)
        accuracies[i - 1][j - 1] = accuracy

x = np.linspace(1, 11, 10)
y = np.linspace(1, 11, 10)

# Zamiana wektora na macierz współrzędnych
X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
ax.set_title('Dokładność w zależności od ilości neuronów w warstwie pierwszej i drugiej')
ax.plot_surface(X, Y, accuracies, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Warstwa 1', labelpad=20)
ax.set_ylabel('Warstwa 2', labelpad=20)
ax.set_zlabel('Dokładność', labelpad=20)

plt.show()

# eksperyment nr. 2 - sprawdzanie wpływu współczynnika zmniejszania i zwiększania współczynnika uczenia

accuracies = np.zeros((8, 8))
s_d_list = [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
s_i_list = [1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.3, 1.5]
for i in range(0, 8):
    for j in range(0, 8):
        network = [
            Dense(15, 10),
            Sigmoid(),
            Dense(10, 5),
            Sigmoid(),
            Dense(5, 1),
            Sigmoid()
        ]
        errors, epoch = train(network, mse, mse_prime, X_tr, y_train, epochs=100, s_i=s_i_list[i], s_d=s_d_list[j])
        accuracy, _ = test(network, X_tr, y_train)
        accuracies[i - 1][j - 1] = accuracy

# Zamiana wektora na macierz współrzędnych
X, Y = np.meshgrid(s_d_list, s_i_list)
ax = plt.axes(projection='3d')
ax.set_title('Dokładność w zależności wartości współczynnika zmniejszania i zwiększania')
ax.plot_surface(X, Y, accuracies, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Współczynnik zmniejszania', labelpad=20)
ax.set_ylabel('Współczynnik zwiększenia', labelpad=20)
ax.set_zlabel('Dokładność', labelpad=20)

plt.show()