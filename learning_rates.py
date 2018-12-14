import numpy as np
from abc import ABCMeta, abstractmethod

class LearningRate:
    @abstractmethod
    def get_rate(self):
        raise NotImplementedError

class FixedRate(LearningRate):
    def __init__(self, eta: float = 0.01):
        self.eta = eta

    def get_rate(self):
        return self.eta

class SqrtDecayRate(LearningRate):
    def __init__(self, eta: float = 0.01, gamma: float = 10.):
        self.eta = eta
        self.gamma = gamma
        self.iter = -1

    def get_rate(self):
        self.iter += 1
        return self.eta * (1. / self.gamma * np.sqrt(self.iter))

class ExpDecayRate(LearningRate):
    def __init__(self, eta: float = 0.01, gamma: float = 0.0001):
        self.eta = eta
        self.gamma = gamma
        self.iter = -1

    def get_rate(self):
        self.iter += 1
        return self.eta * (1. / (1. + (self.gamma * self.iter)))
