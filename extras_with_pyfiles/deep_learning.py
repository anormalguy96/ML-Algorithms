import math
import random


### -----------------
### 1. PERCEPTRON
### -----------------

class Perceptron:
    def __init__(self, input_dim, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = [0.0] * input_dim
        self.b = 0.0

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return self.activation(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                for j in range(len(self.w)):
                    self.w[j] += self.lr * error * xi[j]
                self.b += self.lr * error


### -------------------------------
### 2. MULTI-LAYER PERCEPTRON (MLP)
### -------------------------------

def relu(x):
    return [max(0, xi) for xi in x]

def softmax(z):
    max_z = max(z)
    exp_z = [math.exp(x - max_z) for x in z]
    sum_exp = sum(exp_z)
    return [x / sum_exp for x in exp_z]

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w1 = [[random.uniform(-1, 1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

    def forward(self, x):
        self.z1 = [sum(wij * xj for wij, xj in zip(row, x)) + bj for row, bj in zip(self.w1, self.b1)]
        self.a1 = relu(self.z1)
        self.z2 = [sum(wij * aj for wij, aj in zip(row, self.a1)) + bj for row, bj in zip(self.w2, self.b2)]
        self.a2 = softmax(self.z2)
        return self.a2

    def predict(self, x):
        probs = self.forward(x)
        return probs.index(max(probs))

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                self.forward(xi)

                y_true = [0] * len(self.a2)
                y_true[yi] = 1

                dL_dz2 = [a2i - yi for a2i, yi in zip(self.a2, y_true)]
                for i in range(len(self.w2)):
                    for j in range(len(self.w2[i])):
                        self.w2[i][j] -= self.lr * dL_dz2[i] * self.a1[j]
                    self.b2[i] -= self.lr * dL_dz2[i]

                dL_da1 = [sum(dL_dz2[i] * self.w2[i][j] for i in range(len(self.w2))) for j in range(len(self.a1))]
                dL_dz1 = [da1 * (1 if z1i > 0 else 0) for da1, z1i in zip(dL_da1, self.z1)]

                for i in range(len(self.w1)):
                    for j in range(len(self.w1[i])):
                        self.w1[i][j] -= self.lr * dL_dz1[i] * xi[j]
                    self.b1[i] -= self.lr * dL_dz1[i]


### -------------------------------
### 3. CONVOLUTIONAL BASICS (1D)
### -------------------------------

class SimpleConv1D:
    def __init__(self, kernel):
        self.kernel = kernel

    def convolve(self, x):
        k_len = len(self.kernel)
        out = []
        for i in range(len(x) - k_len + 1):
            val = sum(x[i + j] * self.kernel[j] for j in range(k_len))
            out.append(val)
        return out


### ---------------------------
### 4. SIMPLE RNN CELL
### ---------------------------

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.Wx = [[random.uniform(-1, 1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.Wh = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b = [0.0] * hidden_dim

    def step(self, x, h_prev):
        h_new = []
        for i in range(self.hidden_dim):
            wx_sum = sum(self.Wx[i][j] * x[j] for j in range(len(x)))
            wh_sum = sum(self.Wh[i][j] * h_prev[j] for j in range(self.hidden_dim))
            h = math.tanh(wx_sum + wh_sum + self.b[i])
            h_new.append(h)
        return h_new

    def forward(self, sequence):
        h = [0.0] * self.hidden_dim
        for x in sequence:
            h = self.step(x, h)
        return h