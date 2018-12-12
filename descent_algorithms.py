import numpy as np
from abc import ABCMeta, abstractmethod
from math import sqrt

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
        
    def rate(self):
        self.eta *= (1. / (1. + self.gamma * self.iter))
        self.iter += 1
        return self.eta

class DescentAlgorithm:
    @abstractmethod
    def update(self, w: np.ndarray, grad: np.ndarray):
        raise NotImplementedError

class GradientDescent(DescentAlgorithm):
    def __init__(self, eta: LearningRate):
        self.eta = eta

    def update(self, w: np.ndarray, grad: np.ndarray):
        return w - self.eta.get_rate() * grad

class StochasticVarianceReducedGradientDescent(DescentAlgorithm):
    def __init__(self, eta: LearningRate, w_est: np.ndarray):
        self.eta = eta
        self.w_est = w_est
    
    def update(self, w: np.ndarray, grad: np.ndarray):
        raise NotImplementedError

class NesterovAcceleratedDescent(DescentAlgorithm):
    def __init__(self, eta: LearningRate, y0: np.ndarray):
        self.eta = eta
        self.yt = y0
        self.lam = 0

    def update(self, w, grad):
        lam_next = (1 + sqrt(1 + 4 * self.lam**2)) / 2
        mu = (1 - self.lam) / lam_next

        # Update Rule
        y_new = w - self.eta.get_rate() * grad
        w_new = y_new + mu * (y_new - self.yt)
        self.yt = y_new

        # Update internal sequence
        self.lam = lam_next
        return w_new


