import numpy as np
import time
import datetime
from descent_algorithms import DescentAlgorithm
from learning_rates import LearningRate
from abc import ABCMeta, abstractmethod

"""
Abstract Class for Classification Models

"""
class Model:
    @abstractmethod
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int, rel_conv: float):
        self.lr = lr
        self.w = np.empty([1])
        self.descent = descent
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.rel_conv = rel_conv
        self.X = np.ndarray
        self.y = np.ndarray
        self.time = datetime.time()

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

"""
Logistic Regression Model

"""

class LogisticRegression(Model):
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, num_iter: int, batch_size: int, rel_conv: float):
        super().__init__(descent, lr, num_iter, batch_size, rel_conv)

    """
    Calculates sigmoid of input

    Parameters:

    z (np.ndarray) : numpy array

    Returns:
        np.ndarray : returns sigmoid calculation
    """

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))
    """
    Calculate Logistic Loss

    Parameters:
        h (np.ndarray) : numpy array predictions
        y (np.ndarray) : numpy array ground truths
    Returns:
        float: loss value
    """
    def loss(self, X, y):
        h = self.predict(X)
        return np.dot(-y.T, np.log(h)) - np.dot((1 - y).T,np.log(1 - h))
    """
    Calculate Gradient

    Parameters:
        X (np.ndarray) : numpy array data
        y (np.ndarray) : numpy array ground truths
    Returns:
        float: gradient
    """
    def grad(self, X, y):
        h = self.predict(X)
        return np.dot(X.T, (h - y)) / y.shape[0]
    """
    Fit the data and get weights

    Parameters:
        X (np.ndarray) : numpy array input data
        y (np.ndarray) : numpy array ground truths
    Returns:
        np_array: loss/iteration
    """
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.w = np.random.rand(X.shape[1], 1)
        start_time = time.time()
        loss_data = train(X, y, self, int(self.num_iter / 10), self.rel_conv)
        self.time = time.time()-start_time
        print('Runtime:{a:.5f} secs.'.format(a=self.time))
        return loss_data
    """
    Output prediction

    Parameters:
        X (np.ndarray) : input
    Returns:
        float: predicted value
    """
    def predict(self, X: np.ndarray):
        return self.__sigmoid(np.dot(X, self.w))

    # Return whether the prediction was labeled 0 or 1.
    def predict_label(self, X: np.ndarray):
        return np.round(self.predict(X))


class SVM(Model):
    def __init__(self, descent: DescentAlgorithm, lr: LearningRate, c: float, num_iter: int, batch_size: int,
                 rel_conv: float):
        super().__init__(descent, lr, num_iter, batch_size, rel_conv)
        # self.kernel = kernel.get_kernel()
        self.c = c
        self.w = np.empty([0])

    def loss(self, X, y):
        n = X.shape[0]
        loss = 1 - np.multiply(y, np.dot(X, self.w))
        loss[loss < 0] = 0
        return self.c * np.dot(self.w.T, self.w) + (1/n) * np.sum(loss)

    def grad(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        grad = self.c * self.w
        for i in range(n):
            if y[i] * np.dot(X[i], self.w) <= 0:
                grad += -y[i]*X[i].reshape(X[i].shape[0], 1)
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.w = np.random.rand(X.shape[1], 1)
        loss_data = train(X, y, self, int(self.num_iter / 10), self.rel_conv)
        return loss_data

    def predict(self, X: np.ndarray):
        h = X.dot(self.w)
        h[h >= 0] = 1
        h[h < 0] = -1
        return h


def train(X: np.ndarray, y: np.ndarray, model: Model, print_iter: int, rel_conv: float) \
        -> np.ndarray:
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
            print('Converged in {} iterations.'.format(i))
            loss_data = loss_data[0:i]
            break
    return loss_data



