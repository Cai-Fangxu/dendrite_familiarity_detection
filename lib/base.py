from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import jax

Hyper = TypeVar('Hyper')
HyperDyn = TypeVar('HyperDynamic') # Hyperparameters that are fixed within one simulation, but can change between simulations. They are dynamic, and changing them won't trigger jax.jit recompilation.
State = TypeVar('State')

class Neuron(ABC, Generic[Hyper, State, HyperDyn]):
    def __init__(self, hyperparameters: Hyper):
        self.hyper = hyperparameters
        self.forward = self._forward
        self.update = self._update

    @abstractmethod
    def init_state(self, rng: jax.Array, hyperD: HyperDyn, *args, **kwargs) -> State:
        pass

    @abstractmethod
    def _forward(self, hyperD: HyperDyn, state: State, x, *args, **kwargs):
        pass

    @abstractmethod
    def _update(self, hyperD: HyperDyn, state: State, x, *args, **kwargs) -> State:
        pass
        

class Xs_Generator(ABC):
    def __init__(self):
        self.generate = self._generate

    @abstractmethod
    def _generate(self, rng: jax.Array, n_patterns: int, *args, **kwargs):
        pass

# class Driver(ABC, Generic[Hyper, State]):
#     def __init__(self, rng: jax.Array, neuron: Neuron[Hyper], initial_state: State, n_init_steps: int):
#         self.neuron = neuron
#         self.hyper = neuron.hyper

#         rng, subkey = jax.random.split(rng)
#         self.state = self._init_state(subkey, initial_state, n_init_steps)
#         self.record = []

#     @abstractmethod
#     def _init_state(self, rng, state: State, n_steps: int, *args, **kwargs) -> State:
#         pass

#     @abstractmethod
#     def _update_state(self, state: State, *args, **kwargs) -> State:
#         pass

#     @abstractmethod
#     def _recorder(self, state: State, *args, **kwargs):
#         pass

#     @abstractmethod
#     def run(self, *args, **kwargs):
#         pass