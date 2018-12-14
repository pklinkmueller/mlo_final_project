import numpy as np
from abc import ABCMeta, abstractmethod
from math import sqrt
from random import randint

class DescentAlgorithm:
    @abstractmethod
    def update(self, model, X, y):
        raise NotImplementedError

class GradientDescent(DescentAlgorithm):
    def update(self, model, X, y):
        return model.w - model.lr.get_rate() * model.grad(X, y)

class StochasticVarianceReducedGradientDescent(DescentAlgorithm):
    def update(self, model, X, y):
        w_est = model.w
        mean_est = np.zeros(w_est.shape)
        n = model.X.shape[0]
        k = model.X.shape[1]

        for i in range(n):
            sample = model.X[i, :].reshape(1,k)
            mean_est += model.grad(sample, model.y[i])
        mean_est /= n

        w_t = w_est
        for t in range(2*n):
            i_t = randint(0, n-1)
            sample = model.X[i_t, :].reshape(1,k)
            model.w = w_t
            local_grad = model.grad(sample, model.y[i_t])
            model.w = w_est
            global_grad = model.grad(sample, model.y[i_t])
            w_t = w_t - model.lr.get_rate() * (local_grad - global_grad + mean_est)
        model.w = w_t

        return w_t

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

class MirrorDescent(DescentAlgorithm):
    #using bregman divergence 
    def __init__(self):
        self.wpos = None
        self.wneg = None
        self.initialized = False
    def update(self, model, X, y):
        if(not self.initialized):
            self.wpos = np.ones((X.shape[1], 1))
            self.wneg = -1*self.wpos
            self.initialized = True
        self.wpos = (self.wpos*np.exp(-model.lr.get_rate() * model.grad(X, y)))
        self.wneg = (self.wneg*np.exp(model.lr.get_rate() * model.grad(X, y)))
        return self.wpos + self.wneg



