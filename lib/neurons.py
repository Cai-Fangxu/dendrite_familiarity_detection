import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from .base import Neuron


# region neurontype1

@dataclass
class HyperConfig1:
    nd: int # number of dendrites
    ns: int # number of synapses per dendrite
    n: float # (average) number of selected dendrites for each update
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate

@dataclass
class HyperDynConfig1:
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate

@dataclass
class State1:
    w: jnp.ndarray # shape (nd, ns)

class Neuron_Decay_Int(Neuron[HyperConfig1, State1, HyperDynConfig1]): 
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def init_state(self, rng, hyperD: HyperDynConfig1): 
        w = jax.random.normal(rng, (self.hyper.nd, self.hyper.ns)) / jnp.sqrt(self.hyper.ns)
        return State1(w=w)
    
    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jax.nn.relu(u - hyperD.b) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
    @partial(jax.jit, static_argnames=('self',))
    def _update(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        ns, n = self.hyper.ns, self.hyper.n
        beta, bk = hyperD.beta, hyperD.bk
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        updated_overlap, updated_idx = jax.lax.top_k(overlaps, n)
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk-updated_overlap, 0.)[:, None] * ( x[updated_idx]/x_len_updated2*(bk-(1-beta/ns)*updated_overlap)[:, None]- beta/ns*w[updated_idx] ) # dim = (n, ns)
        w = w.at[updated_idx].set(updated_w)
        new_state = State1(w=w)
        return new_state
    
class Neuron_Decay_NonInt(Neuron_Decay_Int):
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    def _update(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        n, nd, ns = self.hyper.n, self.hyper.nd, self.hyper.ns
        bk, beta = hyperD.bk, hyperD.beta
        bla = jnp.sqrt(2.)*jax.scipy.special.erfinv(1-2*n/nd) # bias - lambda
        x_len2 = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)**2
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        mask = jnp.heaviside(bk-overlaps, 0)*jnp.heaviside(overlaps-bla, 0.) # dim = (nd, )
        w = w + mask[:, None] * ( x/x_len2*(bk-(1-beta/ns)*overlaps)[:, None] - beta/ns*w ) # dim = (nd, ns)
        new_state = State1(w=w)
        return new_state

# endregion neurontype1

