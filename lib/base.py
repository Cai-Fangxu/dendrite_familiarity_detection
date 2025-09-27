from abc import ABC, abstractmethod

class Neuron(ABC):
    
    def __init__(self, hyperparameters):
        self.hyper = hyperparameters
        self.forward = self._forward
        self.update = self._update

    @abstractmethod
    def init_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def _update(self, *args, **kwargs):
        pass
        

class Xs_Generator(ABC):
    
    def __init__(self, hyperparameters):
        self.hyper = hyperparameters
        self.generate = self._generate

    @abstractmethod
    def _generate(self, *args, **kwargs):
        pass