from abc import ABCMeta, abstractmethod

class LossFunction:
    @abstractmethod
    def loss(self):
        raise NotImplementedError
    @abstractmethod
    def grad(self):
        raise NotImplementedError

class LogisticLoss(LossFunction):
    def loss(self, X, Y):
        pass
    def grad(self, X, Y):
        pass