import numpy as np
from abc import ABCMeta, abstractmethod
from math import sqrt

class DescentAlgorithm:
    @abstractmethod
    def update(self, w: np.ndarray, grad: np.ndarray):
        raise NotImplementedError


class GradientDescent(DescentAlgorithm):
    def __init__(self, eta: float):
        self.eta = eta

    def update(self, w: np.ndarray, grad: np.ndarray):
        return w - self.eta * grad


class NesterovAcceleratedDescent(DescentAlgorithm):
    def __init__(self, eta: float, y0: np.ndarray):
        self.eta = eta
        self.yt = y0
        self.lam = 0

    def update(self, w, grad):
        lam_next = (1 + sqrt(1 + 4 * self.lam**2)) / 2
        mu = (1 - self.lam) / lam_next

        # Update Rule
        y_new = w - self.eta * grad
        w_new = y_new + mu * (y_new - self.yt)
        self.yt = y_new

        # Update internal sequence
        self.lam = lam_next
        return w_new


