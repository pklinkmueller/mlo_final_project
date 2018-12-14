import numpy as np
from descent_algorithms import DescentAlgorithm, LearningRate
from abc import ABCMeta, abstractmethod


class Model:
    @abstractmethod
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int, rel_conv: float):
        self.lr = lr
        self.w = np.empty([1])
        self.descent = descent
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.rel_conv = rel_conv

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
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
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int, rel_conv: float):
        super().__init__(descent, lr, num_iter, batch_size, rel_conv)
        self.X = None
        self.y = None

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y):
        h = self.predict(X)
        return np.dot(-y.T, np.log(h)) - np.dot((1 - y).T,np.log(1 - h))

    def grad(self, X, y):
        h = self.predict(X)
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray, non_zero_init: bool = False):
        self.X = X
        self.y = y
        self.w = np.random.rand(X.shape[1], 1)
        loss_data = train(X, y, self, int(self.num_iter / 10), self.rel_conv, non_zero_init)
        return loss_data

    def predict(self, X: np.ndarray):
        return self.__sigmoid(np.dot(X, self.w))


class SVM(Model):
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, c: float, num_iter: int, batch_size: int, rel_conv: float):
        super().__init__(descent, lr, num_iter, batch_size, rel_conv)
        # self.kernel = kernel.get_kernel()
        self.c = c
        self.w = np.empty([0])

    def loss(self, X, y):
        n = X.shape[0]
        l = 1 - np.multiply(y, np.dot(X, self.w))
        l[l < 0] = 0
        return np.dot(self.w.T, self.w) + self.c * (1/n) * np.sum(l)

    def grad(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        grad = np.zeros(self.w.shape)
        for i in range(n):
            if y[i] * np.dot(X[i], self.w) > 1:
                grad += (1/n)*self.w - self.c*y[i]*X[i].reshape(X[i].shape[0],1)
            else:
                grad += (1/n)*self.w
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, non_zero_init: bool = False):
        self.X = X
        self.y = y
        self.w = np.random.rand(X.shape[1], 1)
        loss_data = train(X, y, self, int(self.num_iter / 10), self.rel_conv, non_zero_init)
        return loss_data

    def predict(self, X: np.ndarray):
        return np.sign(X.dot(self.w))


def train(X: np.ndarray, y: np.ndarray, model: Model, print_iter: int, rel_conv: float, non_zero_init: bool = False) \
        -> np.ndarray:
    if non_zero_init:
        model.w = np.ones((X.shape[1], 1))
        model.w /= X.shape[1]
    else:
        model.w = np.zeros((X.shape[1], 1))
    n = X.shape[0]
    start_idx = 0
    perm_idx = np.random.permutation(n)
    idx = range(n)
    loss_data = np.zeros((model.num_iter, 1))
    for i in range(model.num_iter):
        stop_idx = min(start_idx + model.batch_size, n)
        batch_idx = idx[int(start_idx):int(stop_idx)]
        bX = X[perm_idx[batch_idx], :]
        bY = y[perm_idx[batch_idx], :]
        model.w = model.descent.update(model, bX, bY)
        start_idx = stop_idx % n
        loss_data[i] = model.loss(X, y)
        if i % print_iter == 0:
            print('Iter: {:8} train loss: {:.3f}'.format(i, float(loss_data[i])))
        if (i > 0) and ((abs(loss_data[i] - loss_data[i-1]) / loss_data[i]) < rel_conv):
            print('Converged at {} iterations.'.format(i))
            loss_data = loss_data[0:i]
            break
    return loss_data



