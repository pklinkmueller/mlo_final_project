"""
ML-From-Scratch
https://github.com/eriklindernoren/ML-From-Scratch/
"""
import numpy as np
from abc import abstractmethod


class Kernel:
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_kernel(self, **kwargs):
        raise NotImplementedError


class LinearKernel(Kernel):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def get_kernel(self):
        def f(x1, x2):
            return np.inner(x1, x2)
        return f


class PolynomialKernel(Kernel):
    def __init__(self, power, coef):
        super(PolynomialKernel, self).__init__()
        self.power = power
        self.coef = coef

    def get_kernel(self):
        def f(x1, x2):
            return (np.inner(x1, x2) + self.coef)**self.power
        return f


class RadialBasisKernel(Kernel):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def get_kernel(self):
        def f(x1, x2):
            distance = np.linalg.norm(x1 - x2) ** 2
            return np.exp(-self.gamma * distance)
        return f

