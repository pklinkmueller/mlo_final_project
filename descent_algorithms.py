import numpy as np
from abc import ABCMeta, abstractmethod
from math import sqrt
from random import randint


class LearningRate:
    @abstractmethod
    def get_rate(self):
        raise NotImplementedError


class FixedRate(LearningRate):
    def __init__(self, eta: float):
        self.eta = eta
    
    def get_rate(self):
        return self.eta


class ExpDecayRate(LearningRate):
    def __init__(self, eta: float, gamma: float):
        self.eta = eta
        self.gamma = gamma
        self.iter = 0

    def get_rate(self):
        self.eta *= (1. / (1. + self.gamma * self.iter))
        self.iter += 1
        return self.eta


class DescentAlgorithm:
    @abstractmethod
    def update(self, model, X, y):
        raise NotImplementedError


class GradientDescent(DescentAlgorithm):
    def update(self, model, X, y):
        return model.w - model.lr.get_rate() * model.grad(X, y)


class StochasticVarianceReducedGradientDescent(DescentAlgorithm):
    def update(self, model, X, y):
        w_est_curr = model.w
        mean_est = np.zeros(w_est_curr.shape)
        n = X.shape[0]
        
        model.w = self.w_est
        for i in range(n):
            sample = X[i, :].reshape(X.shape[1],1).T
            mean_est += model.grad(sample, y[i])
        mean_est /= n
        
        w_0 = w_est_curr
        model.w = w_0
        for t in range(2*n):
            i_t = randint(0, n-1)
            sample = X[i_t, :].reshape(X.shape[1],1).T
            model.w = w_0
            local_grad = model.grad(sample, y[i_t])
            model.w = w_est_curr
            global_grad = model.grad(sample, y[i_t])
            w_0 = w_0 - model.lr.get_rate() * (local_grad - global_grad + mean_est)
        self.w_est = w_0
        model.w = w_0
        
        return self.w_est


class NesterovAcceleratedDescent(DescentAlgorithm):
    def __init__(self):
        self.yt = 0
        self.lam = 0

    def update(self, model, X, y):
        lam_next = (1 + sqrt(1 + 4 * self.lam**2)) / 2
        mu = (1 - self.lam) / lam_next
        # Update Rule
        y_new = model.w - model.lr.get_rate() * model.grad(X, y)
        w_new = (1 - mu) * y_new + mu * self.yt
        self.yt = y_new
        # Update internal sequence
        self.lam = lam_next
        return w_new


