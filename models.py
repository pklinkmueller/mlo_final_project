import numpy as np
from descent_algorithms import DescentAlgorithm, LearningRate
from abc import ABCMeta, abstractmethod


class Model:
    @abstractmethod
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int):
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

# class LinearRegression(Model):
#     def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int):
#         super().__init__(descent, lr, num_iter, batch_size)

#     @staticmethod
#     def __sigmoid(z):
#         return 1 / (1 + np.exp(-z))

#     def loss(self, h, y):
#         return np.sum((h - y)**2) / y.shape[0]

#     def grad(self, X, y):
#         h = self.predict(X)
#         return np.dot(X.T, (h - y)) / y.shape[0]

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         self.w = np.random.rand(X.shape[1], 1)
#         print(self.w.shape)
#         loss_data = train(X, y, self, int(self.num_iter / 10))
#         return loss_data

#     def predict(self, X: np.ndarray):
#         return self.__sigmoid(np.dot(X, self.w))

class LogisticRegression(Model):
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int):
        super().__init__(descent, lr, num_iter, batch_size)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return np.dot(-y.T, np.log(h)) - np.dot((1 - y).T,np.log(1 - h))

    def grad(self, X, y):
        h = self.predict(X)
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.random.rand(X.shape[1], 1)
        loss_data = train(X, y, self, int(self.num_iter / 10))
        return loss_data

    def predict(self, X: np.ndarray):
        return self.__sigmoid(np.dot(X, self.w))


def train(X: np.ndarray, y: np.ndarray, model: Model, print_iter: int) -> np.ndarray:
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
        bh = model.predict(bX)
        model.w = model.descent.update(model, X, y)
        start_idx = stop_idx % n
        loss_data[i] = model.loss(model.predict(X), y)
        if i % print_iter == 0:
            print('Iter: {:8} train loss: {:.3f}'.format(i, float(loss_data[i])))

    return loss_data


