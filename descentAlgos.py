from lossFunctions import *

class DescentAlgorithm:
    @abstractmethod
    def update(self):
        raise NotImplementedError

class GradientDescent(LossFunction):
    def update(self, x, y, w, loss, eta):
        pass