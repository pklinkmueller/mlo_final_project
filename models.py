import numpy as np
from descent_algorithms import DescentAlgorithm
from abc import ABCMeta, abstractmethod


class Model:
    @abstractmethod
    def __init__(self, descent: DescentAlgorithm, lr: float, num_iter: int, batch_size: int):
        self.lr = lr
        self.w = np.empty([1])
        self.descent = descent
        self.num_iter = num_iter
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def grad(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError


class LogisticRegression(Model):
    def __init__(self, descent: DescentAlgorithm, lr: float, num_iter: int, batch_size: int):
        super().__init__(descent, lr, num_iter, batch_size)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return -y * np.log(h) - (1 - y) * np.log(1 - h)

    def grad(self, X, y):
        h = self.predict(X)
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = train(X, y, self, int(self.num_iter / 10))

    def predict(self, X: np.ndarray):
        return self.__sigmoid(np.dot(X, self.w))


def train(X: np.ndarray, y: np.ndarray, model: Model, print_iter: int):
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    start_idx = 0
    perm_idx = np.random.permutation(n)
    idx = range(n)

    for i in range(model.num_iter):
        stop_idx = min(start_idx + model.batch_size, n)
        batch_idx = idx[int(start_idx):int(stop_idx)]
        bX = X[perm_idx[batch_idx], :]
        bY = y[perm_idx[batch_idx], :]
        bh = model.predict(bX)
        grad = model.grad(bX, bY)
        w = model.descent.update(w, grad)
        start_idx = stop_idx % n
        if i % print_iter == 0:
            print('Iter: {:8} batch loss: {:.3f}'.format(i, model.loss(bh, bY)))

    return w

